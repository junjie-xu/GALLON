# import pandas as pd
from config import cfg, update_cfg
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
from utils import init_path, set_seed, change_dtype, get_valid_smiles, ToDense, valid_smiles_filter
from splitter import scaffold_split, random_scaffold_split, random_split
from functools import partial
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from lm_models import BertClassifier, BertClaInfModel
from loader import split_loader, batch_loader, model_loader
from dataset import CustomMoleculeNet
import pandas as pd
LOG_FREQ = 10




class DistillTrainer():
    def __init__(self, cfg):
        self.seed = cfg.seed
        set_seed(cfg.seed)
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
        
           
        # Load LM predictions and Customized Dataset
        lm_path = f'prt_lm/{self.dataset_name}/{self.split_method}/{cfg.lm.model.name}_{cfg.lm.train.epochs}_{cfg.lm.train.warmup_epochs}_{self.seed}_{self.target_task}_{cfg.lm.train.diagram}.pred'
        to_dense = ToDense(self.max_nodes)
        tsfm = visionT.Compose([to_dense, partial(change_dtype, self.target_task)]) if self.distill_model_name == 'diffpool' else partial(change_dtype, self.target_task)
        dataset = CustomMoleculeNet(name=self.dataset_name, root='./dataset/', lm_path=lm_path, transform=tsfm, pre_filter=valid_smiles_filter)
        
        self.dataset = dataset
        self.num_graphs = len(dataset)
        self.num_classes = dataset.y.max().long().item() + 1
        self.num_features = dataset.x.shape[1]
        
        
        # Load pretrained GNN model
        gnn_path = f"gnn_output/{self.dataset_name}/{self.split_method}/{self.gnn_model_name}_{self.gnn_num_layers}_{self.gnn_hidden_dim}.pt"
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
                                out_channels=self.num_classes,
                                num_layers=self.distill_num_layers,
                                dropout=self.dropout,
                                max_nodes=self.max_nodes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                
         
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")
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
            
            loss = y_loss + 0.1 * gnn_loss + 0.1 * lm_loss
                    
            train_evaluate.update(preds=logits, target=target_y)
            loss.backward()
            self.optimizer.step()
        train_auc = train_evaluate.compute()
        return train_auc, loss.item(), y_loss.item(), gnn_loss.item(), lm_loss.item()
             

    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        
        val_evaluate = AUROC(task="multiclass", num_classes=self.num_classes)
        for data in self.val_loader:
            data = data.to(self.device)
            logits = self._forward(data)
            target_y = data.y.T.squeeze() if self.distill_model_name == 'diffpool' else data.y
            val_evaluate.update(preds=logits, target=target_y)
        val_auc = val_evaluate.compute()
        
        test_evaluate = AUROC(task="multiclass", num_classes=self.num_classes)
        for data in self.test_loader:
            data = data.to(self.device)
            logits = self._forward(data)
            target_y = data.y.T.squeeze() if self.distill_model_name == 'diffpool' else data.y
            test_evaluate.update(preds=logits, target=target_y)
        test_auc = test_evaluate.compute()            
        return val_auc, test_auc, logits
    

    def train(self):
        best_val_auc = 0
        best_test_auc = 0
        for epoch in range(self.epochs+1):
            # t0, es_str = time.time(), ''
            train_auc, loss, y_loss, gnn_loss, lm_loss = self._train()
            val_auc, test_auc, _ = self._evaluate()
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = test_auc

            if epoch % LOG_FREQ == 0:
                print(f'Epoch: {epoch}, Loss: {loss:.4f}, y_loss: {y_loss:.4f}, gnn_loss: {gnn_loss:.4f}, lm_loss: {lm_loss:.4f}')
                print(f'                TrainAuc: {train_auc:.4f}, ValAuc: {val_auc:.4f}, TestAuc: {test_auc:.4f}, BestValAuc: {best_val_auc:.4f}, BestTestAuc: {best_test_auc:.4f}')
        return self.model

    @ torch.no_grad()
    def eval_and_save(self):
        val_acc, test_acc, logits = self._evaluate()
        print(f'[{self.distill_model_name}] ValAuc: {val_acc:.4f}, TestAuc: {test_acc:.4f}\n')
        res = {'val_auc': val_acc.detach().cpu().numpy(), 'test_auc': test_acc.detach().cpu().numpy()}
        return logits, res



def run(cfg):
    seeds = cfg.seed
    all_acc = []
    print(seeds) 
    for seed in seeds:
        cfg.seed = seed
        trainer = DistillTrainer(cfg)
        trainer.train()
        _, acc = trainer.eval_and_save()
        all_acc.append(acc)
        print("-"*100, '\n')

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        for k, v in df.items():
            print(f"{k}: {v.mean()*100:.2f}±{v.std()*100:.2f}")
            
        

if __name__ == '__main__':
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*') 
    
    cfg = update_cfg(cfg)
    run(cfg)

