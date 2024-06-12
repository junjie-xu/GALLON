# Distill (LLM + GNN) --> MLP, for regression tasks

from itertools import chain
from config import cfg, update_cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models import GCN
from torchmetrics import AUROC
from torchmetrics.regression import MeanSquaredError as MSE
from utils import , set_seed, change_dtype, valid_smiles_filter, change_target
from functools import partial
from loader import split_loader, batch_loader, model_loader
from dataset import CustomMoleculeNet


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
    elif split_method == 'scaffold':
        best_lm_path = {
            'bace': f'prt_lm/bace/scaffold/roberta-base_20_0.6_{seed}_0_True',
            'bbbp': f'prt_lm/bbbp/scaffold/roberta-base_20_0.3_{seed}_0_True',
            'clintox': f'prt_lm/clintox/scaffold/roberta-base_5_0.3_{seed}_1_True',
            'hiv': f'prt_lm/hiv/scaffold/roberta-base_5_0.6_{seed}_0_True',
            'esol': f'prt_lm/esol/scaffold/roberta-base_5_0.6_{seed}_0_True', 
            'freesolv': f'prt_lm/freesolv/scaffold/roberta-base_5_0.6_{seed}_0_True', 
            'lipo': f'prt_lm/lipo/scaffold/roberta-base_10_0.3_{seed}_0_True',
        }
    return best_lm_path[dataname]



class DistillTrainer():
    def __init__(self, cfg, seed):
        self.seed = seed
        set_seed(self.seed)
        print(self.seed)
        self.device = cfg.device
        self.dataset_name = cfg.dataset.name
        self.target_task = cfg.dataset.target_task
        self.split_method = cfg.dataset.split_method
        
        # Pretained-GNN
        self.gnn_model_name = cfg.gnn.model.name.lower()
        self.gnn_num_layers = cfg.gnn.model.num_layers
        self.gnn_hidden_dim = cfg.gnn.model.hidden_dim
        
        # Distilliation Model
        self.distill_model_name = cfg.distill.model.name.lower()
        self.distill_num_layers = cfg.distill.model.num_layers
        self.distill_hidden_dim = cfg.distill.model.hidden_dim
        
        self.dropout = cfg.distill.train.dropout
        self.lr = cfg.distill.train.lr
        self.weight_decay = cfg.distill.train.weight_decay
        self.epochs = cfg.distill.train.epochs
        self.max_nodes = cfg.distill.model.max_nodes
        self.batch_size = cfg.distill.train.batch_size
        self.alpha = cfg.distill.train.alpha
        self.beta = cfg.distill.train.beta
        
        
        if self.dataset_name in ['esol', 'lipo', 'freesolv']:
            self.metrics = 'rmse'
        else:
            self.metrics = 'auc'
        
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
        self.gnn_model.load_state_dict(torch.load(gnn_path, map_location=self.device))
        
        
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
        self.rep_trans = nn.Linear(self.distill_hidden_dim, self.distill_hidden_dim).to(self.device)
        self.gnn_rep_trans = nn.Linear(self.distill_hidden_dim, self.distill_hidden_dim).to(self.device)
        self.rep_trans = nn.Identity().to(self.device)
        self.gnn_rep_trans = nn.Identity().to(self.device)
        self.lm_rep_trans = nn.Linear(768, self.distill_hidden_dim).to(self.device)
        
        combined_params = chain(self.model.parameters(), 
                                self.rep_trans.parameters(),
                                self.gnn_rep_trans.parameters(),
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
            # gnn_loss = self.ce_gnn(input=logits, target=gnn_logits) # scalar alignment between mlp and gnn
            # lm_loss = self.ce_lm(input=logits, target=lm_logits)    # scalar alignment between mlp and lm
            
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
        if self.metrics == 'rmse':
            best_val_metric = 1e8
            best_test_metric = 1e8
            for epoch in range(self.epochs + 1):
                train_metric, loss, y_loss, gnn_loss, lm_loss = self._train()
                val_rmse, test_rmse = self._evaluate()
                
                if val_rmse < best_val_metric:
                    best_val_metric = val_rmse
                    best_test_metric = test_rmse
                    
                if epoch % 50 == 0:
                    print(f'Epoch: {epoch}, Loss: {loss:.3f} = y_loss: {y_loss:.3f} + gnn_rep_loss: {gnn_loss:.3f} + lm_rep_loss: {lm_loss:.3f}')
                    print(f'            TrainRMSE: {train_metric:.4f}, ValRMSE: {val_rmse:.4f}, TestRMSE: {test_rmse:.4f}')
                    print(f'            BestValRMSE: {best_val_metric:.4f}, BestTestRMSE: {best_test_metric:.4f}')
                    print()
                    
        return best_val_metric, best_test_metric



def run(cfg):
    seeds = cfg.seed
    all_acc = []
    print(seeds) 
    for seed in seeds:
        trainer = DistillTrainer(cfg, seed)
        best_val_metric, best_test_metric = trainer.train()
        res = {'val_metric': best_val_metric.detach().cpu().numpy(), 'test_metric': best_test_metric.detach().cpu().numpy()}
        all_acc.append(res)
        print("-"*100, '\n')
        
    if len(all_acc) > 1:     
        df = pd.DataFrame(all_acc)
        print(df)
        for k, v in df.items():
            print(f"{k}: {v.mean():.4f}±{v.std():.4f}")
            
        

if __name__ == '__main__':
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    cfg = update_cfg(cfg)
    run(cfg)

