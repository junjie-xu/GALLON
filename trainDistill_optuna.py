# import pandas as pd
# from config import cfg, update_cfg
import argparse
from config_parser import parser_add_main_args
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import GCN, DiffPool
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
import optuna
LOG_FREQ = 50

def get_best_lm_path(dataname, split_method, seed):
    if split_method == 'random_scaffold':
        best_lm_path = {
            'bace': f'prt_lm/bace/random_scaffold/roberta-base_20_0.6_{seed}_0_True.pred',
            'bbbp': f'prt_lm/bbbp/random_scaffold/roberta-base_10_0.6_{seed}_0_True.pred',
            'clintox': f'prt_lm/clintox/random_scaffold/roberta-base_20_0.3_{seed}_1_True.pred',      
        }
    elif split_method == 'random':
        best_lm_path = {
            'bace': f'prt_lm/bace/random/roberta-base_20_0.3_{seed}_0_True.pred',
            'bbbp': f'prt_lm/bbbp/random/roberta-base_10_0.3_{seed}_0_True.pred',
            'clintox': f'prt_lm/clintox/random/roberta-base_20_0.3_{seed}_1_True.pred',      
        }
    return best_lm_path[dataname]


class DistillTrainer():
    def __init__(self, args, seed):
        self.seed = seed
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
        
        
        if self.dataset_name in ['esol', 'lipo', 'freesolv']:
            self.metrics = 'rmse'
        else:
            self.metrics = 'auc'
        
        
        # Load LM predictions and Customized Dataset
        # lm_path = f'prt_lm/{self.dataset_name}/{self.target_task}/{self.split_method}/{args.lm_name}_{args.lm_epochs}_{args.lm_warmup_epochs}_{args.seed}.pred'
        # to_dense = ToDense(self.max_nodes)
        # tsfm = visionT.Compose([to_dense, partial(change_dtype, self.target_task)]) if self.distill_model_name == 'diffpool' else partial(change_dtype, self.target_task)
        
        lm_path = get_best_lm_path(self.dataset_name, self.split_method, self.seed)
        tsfm = partial(change_target, self.target_task) if self.metrics == 'rmse' else partial(change_dtype, self.target_task)
        dataset = CustomMoleculeNet(name=self.dataset_name, root='./dataset/', lm_path=lm_path, transform=tsfm, pre_filter=valid_smiles_filter)
        
        self.dataset = dataset
        self.num_graphs = len(dataset)
        self.num_classes = dataset.y.max().long().item() + 1
        self.num_features = dataset.x.shape[1]
        
        
        # Load pretrained GNN model
        gnn_path = f"gnn_output/{self.dataset_name}/{self.split_method}/{self.gnn_model_name}_{self.gnn_hidden_dim}_{self.seed}.pt"
        print(f'GNN Model is loaded from {gnn_path}')
        self.gnn_model = GCN(in_channels=self.num_features,
                            hidden_channels=self.gnn_hidden_dim,
                            out_channels=self.num_classes,
                            num_layers=self.gnn_num_layers,
                            dropout=self.dropout).to(self.device)
        self.gnn_model.load_state_dict(torch.load(gnn_path))
        
        
        # Load split, dataloader, model
        assert self.split_method in ['random', 'scaffold', 'random_scaffold']
        train_idx, val_idx, test_idx = split_loader(dataset=dataset, 
                                                    split_method=self.split_method,
                                                    frac_train=args.train_prop, 
                                                    frac_val=args.val_prop, 
                                                    frac_test=args.test_prop,
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
                                out_channels=self.num_classes,
                                num_layers=self.distill_num_layers,
                                dropout=self.dropout,
                                max_nodes=self.max_nodes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                
         
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of parameters: {trainable_params}")
        self.loss_func = nn.NLLLoss()
        self.ce_gnn = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.ce_lm = nn.KLDivLoss(reduction='batchmean', log_target=True)
        

    def _forward(self, data):
        if self.distill_model_name == 'diffpool':
            logits = self.model(x=data.x, adj=data.adj)
        else:
            logits = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch)
        return logits

    def _train(self):
        self.model.train()
        train_evaluate = AUROC(task="multiclass", num_classes=self.num_classes)
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            target_y = data.y.T.squeeze() if self.distill_model_name == 'diffpool' else data.y
            
            logits = self._forward(data)
            gnn_logits = self.gnn_model(data.x, data.edge_index, data.batch)
            
            ls_logits = F.log_softmax(logits, dim=1)
            ls_gnn_logits = F.log_softmax(gnn_logits, dim=1)
            ls_lm_logits = F.log_softmax(data.lm_prediction, dim=1)
            
            y_loss = self.loss_func(input=ls_logits, target=target_y) 
            gnn_loss = self.ce_gnn(input=ls_logits, target=ls_gnn_logits) 
            lm_loss = self.ce_lm(input=ls_logits, target=ls_lm_logits)
            
            loss = y_loss + self.alpha * gnn_loss + self.beta * lm_loss
                    
            train_evaluate.update(preds=logits, target=target_y)
            loss.backward()
            self.optimizer.step()
        train_auc = train_evaluate.compute()
        return train_auc, loss.item(), y_loss.item(), gnn_loss.item(), lm_loss.item()
             

    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        
        val_evaluate = MSE(squared=False).to(self.device) if self.metrics == 'rmse' else AUROC(task="multiclass", num_classes=self.num_classes)
        for data in self.val_loader:
            data = data.to(self.device)
            logits = self._forward(data)
            logits = logits.squeeze() if self.metrics == 'rmse' else logits
            target_y = data.y
            val_evaluate.update(preds=logits, target=target_y)
        val_metric = val_evaluate.compute()
        
        test_evaluate = MSE(squared=False).to(self.device) if self.metrics == 'rmse' else AUROC(task="multiclass", num_classes=self.num_classes)
        for data in self.test_loader:
            data = data.to(self.device)
            logits = self._forward(data)
            logits = logits.squeeze() if self.metrics == 'rmse' else logits
            target_y = data.y
            test_evaluate.update(preds=logits, target=target_y)
        test_metric = test_evaluate.compute()            
        return val_metric, test_metric, logits
    

    def train(self):
        if self.metrics == 'rmse':
            best_val_metric = 1e8
            best_test_metric = 1e8
            for epoch in range(self.epochs + 1):
                loss, train_rmse = self._train()
                val_rmse, test_rmse, _ = self._evaluate()
                
                if val_rmse < best_val_metric:
                    best_val_metric = val_rmse
                    best_test_metric = test_rmse
                    torch.save(self.model.state_dict(), self.ckpt)
                    
                if epoch % LOG_FREQ == 0:
                    print(f'Epoch: {epoch}, Loss: {loss:.4f}, TrainRMSE: {train_rmse:.4f}, ValRMSE: {val_rmse:.4f}, TestRMSE: {test_rmse:.4f}')
                    print(f'                BestValRMSE: {best_val_metric:.4f}, BestTestRMSE: {best_test_metric:.4f}')  
        else:
            best_val_metric = 0
            best_test_metric = 0
            for epoch in range(self.epochs + 1):
                loss, train_auc = self._train()
                val_auc, test_auc, _ = self._evaluate()
                
                if val_auc > best_val_metric:
                    best_val_metric = val_auc
                    best_test_metric = test_auc
                    torch.save(self.model.state_dict(), self.ckpt)
                    
                if epoch % LOG_FREQ == 0:
                    print(f'Epoch: {epoch}, Loss: {loss:.4f}, TrainAuc: {train_auc:.4f}, ValAuc: {val_auc:.4f}, TestAuc: {test_auc:.4f}')
                    print(f'                BestValAuc: {best_val_metric:.4f}, BestTestAuc: {best_test_metric:.4f}')
        return best_val_metric, best_test_metric, self.model

    # @ torch.no_grad()
    # def eval_and_save(self):
    #     val_acc, test_acc, logits = self._evaluate()
    #     print(f'[{self.distill_model_name}] ValAuc: {val_acc:.4f}, TestAuc: {test_acc:.4f}\n')
    #     res = {'val_auc': val_acc.detach().cpu().numpy(), 'test_auc': test_acc.detach().cpu().numpy()}
    #     return logits, res


# # Used for interate mumtiple seeds
# def run(args):
#     seeds = [0, 1, 2, 3, 4]
#     all_acc = []
#     print(seeds) 
#     for seed in seeds:
#         trainer = DistillTrainer(args, seed)
#         best_val_auc, best_test_auc, _ = trainer.train()
#         # _, acc = trainer.eval_and_save()
#         all_acc.append(best_test_auc)
#     return sum(all_acc) / len(all_acc)


def run(args):
    print("seed: ", args.seed) 
    trainer = DistillTrainer(args, args.seed)
    best_val_auc, best_test_auc, _ = trainer.train()
    return best_test_auc


def search_hyper_params(trial: optuna.Trial):
    hidden = trial.suggest_categorical("hidden", [args.distill_hidden])
    alpha = trial.suggest_categorical("alpha", [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    beta = trial.suggest_categorical("beta", [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.3, 0.5, 0.7])
    lr = trial.suggest_categorical("lr", [0.05, 0.01, 0.005, 0.001])
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
    
    name = f'optuna_search/{args.dataname}/{args.split_method}_{args.distill_hidden}_{args.seed}'
    print("Study Name: ", name)
    
    study = optuna.create_study(direction="maximize",
                                study_name=name,
                                storage="sqlite:///" + name + ".db",
                                load_if_exists=True)
    study.optimize(search_hyper_params, n_trials=500)
    
    df = study.trials_dataframe()
    df.to_csv(f'{name}.csv', index=False)
    
    print("best params ", study.best_params)
    print("best valf1 ", study.best_value)
    print('\n')

