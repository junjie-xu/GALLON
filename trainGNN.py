# import pandas as pd
from config import cfg, update_cfg
import time
import torch
import numpy as np
from models import GCN, DiffPool
from torch_geometric.datasets import MoleculeNet
import torch_geometric.transforms as T
from torchvision import transforms as visionT
from torch_geometric.loader import DataLoader, DenseDataLoader
from torchmetrics import AUROC
from utils import init_path, set_seed, change_dtype, get_valid_smiles, ToDense, valid_smiles_filter
from functools import partial
from loader import split_loader, batch_loader, model_loader
import pandas as pd
LOG_FREQ = 10




class GNNTrainer():
    def __init__(self, cfg):
        self.seed = cfg.seed
        set_seed(cfg.seed)
        print(self.seed)
        
        self.device = cfg.device
        self.dataset_name = cfg.dataset.name
        self.target_task = cfg.dataset.target_task
        self.split_method = cfg.dataset.split_method
        
        self.gnn_model_name = cfg.gnn.model.name.lower()
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers
        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.weight_decay = cfg.gnn.train.weight_decay
        self.epochs = cfg.gnn.train.epochs
        self.max_nodes = cfg.gnn.model.max_nodes
        self.batch_size = cfg.gnn.train.batch_size

        
        to_dense = ToDense(self.max_nodes)
        tsfm = visionT.Compose([to_dense, partial(change_dtype, self.target_task)]) if self.gnn_model_name == 'diffpool' else partial(change_dtype, self.target_task)
        dataset = MoleculeNet(name=self.dataset_name, root='./dataset/', transform=tsfm, pre_filter=valid_smiles_filter)
    
        self.dataset = dataset
        self.num_graphs = len(self.dataset)
        self.num_classes = dataset.y.max().long().item() + 1
        self.num_features = dataset.x.shape[1]
        print(self.dataset, f'has {self.num_graphs} graphs.')
        
        train_idx, valid_idx, test_idx = split_loader(dataset=self.dataset, 
                                                    split_method=self.split_method,
                                                    frac_train=cfg.dataset.train_prop, 
                                                    frac_val=cfg.dataset.val_prop, 
                                                    frac_test=cfg.dataset.test_prop,
                                                    seed=self.seed)
        self.train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
        self.val_dataset = torch.utils.data.Subset(self.dataset, valid_idx)
        self.test_dataset = torch.utils.data.Subset(self.dataset, test_idx)
        
        self.train_loader, self.val_loader, self.test_loader = batch_loader(self.train_dataset, 
                                                                            self.val_dataset, 
                                                                            self.test_dataset, 
                                                                            self.gnn_model_name, 
                                                                            self.batch_size)
        self.model = model_loader(model_name=self.gnn_model_name,
                                    in_channels=self.num_features,
                                    hidden_channels=self.hidden_dim,
                                    out_channels=self.num_classes,
                                    num_layers=self.num_layers,
                                    dropout=self.dropout,
                                    max_nodes=self.max_nodes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_func = torch.nn.CrossEntropyLoss()

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")
        self.ckpt = f"gnn_output/{self.dataset_name}/{self.split_method}/{self.gnn_model_name}_{self.hidden_dim}_{self.seed}.pt"
        print(f'Model will be saved: {self.ckpt}', '\n')
        

    def _forward(self, data):
        if self.gnn_model_name == 'diffpool':
            logits = self.model(data.x, data.adj)
        else:
            logits = self.model(data.x, data.edge_index, data.batch)
        return logits

    def _train(self):
        self.model.train()
        
        train_evaluate = AUROC(task="multiclass", num_classes=self.num_classes)
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            logits = self._forward(data)
            target_y = data.y.T.squeeze() if self.gnn_model_name == 'diffpool' else data.y
            loss = self.loss_func(input=logits, target=target_y)
            train_evaluate.update(preds=logits, target=target_y)
            loss.backward()
            self.optimizer.step()
        train_auc = train_evaluate.compute()
        return loss.item(), train_auc
             

    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        
        val_evaluate = AUROC(task="multiclass", num_classes=self.num_classes)
        for data in self.val_loader:
            data = data.to(self.device)
            logits = self._forward(data)
            target_y = data.y.T.squeeze() if self.gnn_model_name == 'diffpool' else data.y
            val_evaluate.update(preds=logits, target=target_y)
        val_auc = val_evaluate.compute()
        
        test_evaluate = AUROC(task="multiclass", num_classes=self.num_classes)
        for data in self.test_loader:
            data = data.to(self.device)
            logits = self._forward(data)
            target_y = data.y.T.squeeze() if self.gnn_model_name == 'diffpool' else data.y
            test_evaluate.update(preds=logits, target=target_y)
        test_auc = test_evaluate.compute()            
        return val_auc, test_auc, logits
    

    def train(self):
        best_val_acc = 0
        best_test_acc = 0
        for epoch in range(self.epochs + 1):
            loss, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), self.ckpt)
                
            if epoch % LOG_FREQ == 0:
                print(f'Epoch: {epoch}, Loss: {loss:.4f}, TrainAuc: {train_acc:.4f}, ValAuc: {val_acc:.4f}, TestAuc: {test_acc:.4f}')
                print(f'                BestValAuc: {best_val_acc:.4f}, BestTestAuc: {best_test_acc:.4f}')
        print(f'[{self.gnn_model_name}] model saved: {self.ckpt}, with best_val_acc:{best_val_acc:.4f} and corresponding test_acc:{best_test_acc:.4f}', '\n')
        return self.model

    @ torch.no_grad()
    def eval_and_save(self):
        self.model.load_state_dict(torch.load(self.ckpt))
        print(f'[{self.gnn_model_name}] model saved: {self.ckpt}')
        val_acc, test_acc, logits = self._evaluate()
        print(f'[{self.gnn_model_name}] ValAuc: {val_acc:.4f}, TestAuc: {test_acc:.4f}\n')
        res = {'val_auc': val_acc.detach().cpu().numpy(), 'test_auc': test_acc.detach().cpu().numpy()}
        return logits, res




def run_train_gnn(cfg):
    seeds = cfg.seed
    all_acc = []
    print(seeds) 
    for seed in seeds:
        cfg.seed = seed
        trainer = GNNTrainer(cfg)
        trainer.train()
        _, acc = trainer.eval_and_save()
        all_acc.append(acc)
        print("-"*100, '\n')

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        for k, v in df.items():
            print(f"{k}: {v.mean()*100:.2f}±{v.std()*100:.2f}")
            
        path = f'prt_results/prt_gnn_results/{cfg.dataset.name}/{cfg.dataset.split_method}/{cfg.gnn.model.name}_{cfg.gnn.model.hidden_dim}.csv'
        df.to_csv(path, index=False)



if __name__ == '__main__':
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*') 
    cfg = update_cfg(cfg)
    run_train_gnn(cfg)

