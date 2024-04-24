# Distill (LLM + GNN) --> MLP, for regression tasks

from itertools import chain
from config import cfg, update_cfg
import argparse
from config_parser import parser_add_main_args
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import GCN
from torch_geometric.datasets import MoleculeNet
import torch_geometric.transforms as T
from torchvision import transforms as visionT
from torch_geometric.loader import DataLoader, DenseDataLoader
from torchmetrics import AUROC
from torchmetrics.regression import MeanSquaredError as MSE
from utils import init_path, set_seed, change_dtype, ToDense, valid_smiles_filter, change_target
from splitter import scaffold_split, random_scaffold_split, random_split
from functools import partial
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from lm_models import BertClassifier, BertClaInfModel
from loader import split_loader, batch_loader, model_loader
from dataset import CustomMoleculeNet
import pandas as pd
LOG_FREQ = 50


def get_best_lm_path(dataname, split_method, seed):
    if split_method == 'random_scaffold':
        best_lm_path = {
            'bace': f'prt_lm/bace/random_scaffold/roberta-base_20_0.6_{seed}_0_True',
            'bbbp': f'prt_lm/bbbp/random_scaffold/roberta-base_10_0.6_{seed}_0_True',
            'clintox': f'prt_lm/clintox/random_scaffold/roberta-base_20_0.3_{seed}_1_True',
            'freesolv': f'prt_lm/freesolv/random_scaffold/microsoft/deberta-base_20_0.6_{seed}_0_True', 
            'esol': f'prt_lm/esol/random_scaffold/microsoft/deberta-base_20_0.6_{seed}_0_True', 
            'lipo': f'prt_lm/lipo/random_scaffold/roberta-base_5_0.3_{seed}_0_True',
        }
    return best_lm_path[dataname]



class DistillTrainer():
    def __init__(self, args):
        self.seed = args.seed
        set_seed(self.seed)
        print(self.seed)
        
        self.device = args.device
        self.dataset_name = args.dataname
        self.target_task = args.target_task
        self.split_method = args.split_method
        
        # Pretained-GNN
        self.gnn_model_name = args.gnn_name.lower()
        self.gnn_num_layers = args.gnn_layers
        self.gnn_hidden_dim = args.gnn_hidden
        
        # Distilliation Model
        self.distill_model_name = args.distill_name.lower()
        self.distill_num_layers = args.distill_layers
        self.distill_hidden_dim = args.distill_hidden 
        
        self.dropout = args.distill_dropout
        self.lr = args.distill_lr
        self.weight_decay = args.distill_wd
        self.epochs = args.distill_epochs
        self.max_nodes = args.distill_max_nodes
        self.batch_size = args.distill_batch_size
        self.alpha = args.alpha
        self.beta = args.beta
        
        assert self.dataset_name in ['esol', 'lipo', 'freesolv']
        
           
        # Load LM predictions and Customized Dataset
        tsfm = partial(change_target, self.target_task)
        lm_path = get_best_lm_path(self.dataset_name, self.split_method, self.seed)
        dataset = CustomMoleculeNet(name=self.dataset_name, root='./dataset/', lm_path=lm_path, transform=tsfm, pre_filter=valid_smiles_filter)
        print("There are {} graphs in the dataset.".format(len(dataset)))
        self.dataset = dataset
        self.num_graphs = len(dataset)
        self.num_features = dataset.x.shape[1]
        
        
        # Load pretrained GNN model
        gnn_path = f"gnn_output/{self.dataset_name}/{self.split_method}_{self.gnn_model_name}_{self.gnn_hidden_dim}_{self.seed}.pt"
        print(f'GNN Model is loaded from {gnn_path}')
        self.gnn_model = GCN(in_channels=self.num_features,
                            hidden_channels=self.gnn_hidden_dim,
                            out_channels=1,
                            num_layers=self.gnn_num_layers,
                            dropout=self.dropout).to(self.device)
        self.gnn_model.load_state_dict(torch.load(gnn_path))
        
        
        # Load split, dataloader, model
        assert self.split_method in ['random', 'scaffold', 'random_scaffold']
        train_idx, val_idx, test_idx = split_loader(dataset=dataset, 
                                                    split_method=self.split_method,
                                                    frac_train=cfg.dataset.train_prop, 
                                                    frac_val=cfg.dataset.val_prop, 
                                                    frac_test=cfg.dataset.test_prop,
                                                    seed=self.seed)     
        self.train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
        self.val_dataset = torch.utils.data.Subset(self.dataset, val_idx)
        self.test_dataset = torch.utils.data.Subset(self.dataset, test_idx)
          
        self.train_loader, self.val_loader, self.test_loader = batch_loader(self.train_dataset, 
                                                                            self.val_dataset, 
                                                                            self.test_dataset, 
                                                                            self.distill_model_name, 
                                                                            self.batch_size)
        self.model = model_loader(model_name=self.distill_model_name,
                                in_channels=self.num_features,
                                hidden_channels=self.distill_hidden_dim,
                                out_channels=1,
                                num_layers=self.distill_num_layers,
                                dropout=self.dropout,
                                max_nodes=self.max_nodes).to(self.device)
        # self.rep_trans = nn.Linear(self.distill_hidden_dim, self.distill_hidden_dim).to(self.device)
        # self.gnn_rep_trans = nn.Linear(self.distill_hidden_dim, self.distill_hidden_dim).to(self.device)
        self.rep_trans = nn.Identity().to(self.device)
        self.gnn_rep_trans = nn.Identity().to(self.device)
        self.lm_rep_trans = nn.Linear(768, self.distill_hidden_dim).to(self.device)
        
        combined_params = chain(self.model.parameters(), 
                                # self.rep_trans.parameters(),
                                # self.gnn_rep_trans.parameters(),
                                self.lm_rep_trans.parameters(),
                                )
        self.optimizer = torch.optim.Adam(combined_params, lr=self.lr, weight_decay=self.weight_decay)
                
        self.loss = nn.MSELoss(reduction='mean')
        self.gnn_loss = nn.MSELoss(reduction='mean')
        self.lm_loss = nn.MSELoss(reduction='mean')
        self.gnn_rep_loss = nn.MSELoss(reduction='mean')
        self.lm_rep_loss = nn.MSELoss(reduction='mean')
        

    def _forward(self, data):
        return self.model(x=data.x, edge_index=data.edge_index, batch=data.batch)

    def _train(self):
        self.model.train()
        train_evaluate = MSE(squared=False).to(self.device)
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            logits, reps = self._forward(data)
            gnn_logits, gnn_reps = self.gnn_model(data.x, data.edge_index, data.batch)
            lm_logits, lm_reps = data.lm_predictions, data.lm_embeddings  # |G| x 1,  |G| x 768
            
            l0, l1, l2, g_rep = reps  # N x hidden,  |G| x hidden
            gnn_l0, gnn_l1, gnn_l2, gnn_g_rep = gnn_reps  # N x hidden,  |G| x hidden
            
            # l0_trans = self.rep_trans(l0)
            # l1_trans = self.rep_trans(l1)
            # l2_trans = self.rep_trans(l2)
            g_rep_trans = self.rep_trans(g_rep)
            
            # gnn_l0_trans = self.gnn_rep_trans(gnn_l0)
            # gnn_l1_trans = self.gnn_rep_trans(gnn_l1)
            # gnn_l2_trans = self.gnn_rep_trans(gnn_l2)
            gnn_g_rep_trans = self.gnn_rep_trans(gnn_g_rep)
            lm_g_rep_trans = self.lm_rep_trans(lm_reps)
            
            gnn_rep_loss = self.gnn_rep_loss(input=g_rep_trans, target=gnn_g_rep_trans) # representation alignment between MLP and GNN
            lm_rep_loss = self.lm_rep_loss(input=g_rep_trans, target=lm_g_rep_trans)    # representation alignment between MLP and LM
            
            y_loss = self.loss(input=logits.squeeze(), target=data.y)         # scalar alignment between mlp and y
            # gnn_loss = self.gnn_loss(input=logits, target=gnn_logits) # scalar alignment between mlp and gnn
            # lm_loss = self.lm_loss(input=logits, target=lm_logits)    # scalar alignment between mlp and lm
            
            loss = y_loss + self.alpha * gnn_rep_loss + self.beta * lm_rep_loss
    
            train_evaluate.update(preds=logits.squeeze(), target=data.y)
            loss.backward()
            self.optimizer.step()
        train_metric = train_evaluate.compute()
        return train_metric, loss.item(), y_loss.item(), gnn_rep_loss.item(), lm_rep_loss.item()
             

    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        
        val_evaluate = MSE(squared=False).to(self.device)
        for data in self.val_loader:
            data = data.to(self.device)
            logits, _ = self._forward(data)
            logits = logits.squeeze()
            target_y = data.y
            val_evaluate.update(preds=logits, target=target_y)
        val_metric = val_evaluate.compute()
        
        test_evaluate = MSE(squared=False).to(self.device)
        for data in self.test_loader:
            data = data.to(self.device)
            logits, _ = self._forward(data)
            logits = logits.squeeze()
            target_y = data.y
            test_evaluate.update(preds=logits, target=target_y)
        test_metric = test_evaluate.compute()            
        return val_metric, test_metric
    

    def train(self):
        best_val_metric = 1e8
        best_test_metric = 1e8
        for epoch in range(self.epochs + 1):
            train_metric, loss, y_loss, gnn_rep_loss, lm_rep_loss = self._train()
            val_rmse, test_rmse = self._evaluate()
            
            if val_rmse < best_val_metric:
                best_val_metric = val_rmse
                best_test_metric = test_rmse
                # torch.save(self.model.state_dict(), self.ckpt)
                
            if epoch % LOG_FREQ == 0:
                print(f'Epoch: {epoch}, Loss: {loss:.3f} = y_loss: {y_loss:.3f} + gnn_rep_loss: {gnn_rep_loss:.3f} + lm_rep_loss: {lm_rep_loss:.3f}')
                print(f'                TrainRMSE: {train_metric:.4f}, ValRMSE: {val_rmse:.4f}, TestRMSE: {test_rmse:.4f}')
                print(f'                BestValRMSE: {best_val_metric:.4f}, BestTestRMSE: {best_test_metric:.4f}')
                print()       
        return best_val_metric, best_test_metric


def run(args):
    print("seed: ", args.seed) 
    trainer = DistillTrainer(args)
    best_val_auc, best_test_auc = trainer.train()
    return best_test_auc
            
        
def search_hyper_params(trial: optuna.Trial):
    hidden = trial.suggest_categorical("hidden", [args.distill_hidden])
    alpha = trial.suggest_categorical("alpha", [0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    beta = trial.suggest_categorical("beta", [0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.3, 0.5, 0.7])
    lr = trial.suggest_categorical("lr", [0.05, 0.01, 0.005, 0.001, 0.0005])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 5e-4])
    
    args.gnn_hidden = hidden
    args.distill_hidden = hidden
    args.alpha = alpha
    args.beta = beta
    args.distill_dropout = dropout
    args.distill_lr = lr
    args.distill_wd = weight_decay  
    return run(args)



if __name__ == '__main__':
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*') 
    print('-'*80)
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    args = parser.parse_args()
    
    device = args.device
    args.gnn_hidden = args.distill_hidden
    
    name = f'optuna_search/{args.dataname}/mode{args.mode}/{args.split_method}_{args.distill_hidden}_{args.seed}'
    print("Study Name: ", name)
    
    study = optuna.create_study(direction="minimize",
                                study_name=name,
                                storage="sqlite:///" + name + ".db",
                                load_if_exists=True)
    study.optimize(search_hyper_params, n_trials=1000)
    
    df = study.trials_dataframe()
    df.to_csv(f'{name}.csv', index=False)
    
    print("best params ", study.best_params)
    print("best valf1 ", study.best_value)
    print('\n')

