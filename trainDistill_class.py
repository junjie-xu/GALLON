from config import cfg, update_cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GCN
import torch_geometric.transforms as T
from torchmetrics import AUROC
from torchmetrics.regression import MeanSquaredError as MSE
from utils import init_path, set_seed, change_dtype, valid_smiles_filter, change_target
from splitter import scaffold_split, random_scaffold_split, random_split
from functools import partial
from loader import split_loader, batch_loader, model_loader
from dataset import CustomMoleculeNet



# Change to the optimal pretrained LM path !!!!
def get_best_lm_path(dataname, split_method, seed):
    if split_method == 'random_scaffold':
        best_lm_path = {
            'bace': f'prt_lm/bace/random_scaffold/roberta-base_20_0.6_{seed}_0_True',
            'bbbp': f'prt_lm/bbbp/random_scaffold/roberta-base_10_0.6_{seed}_0_True',
            'clintox': f'prt_lm/clintox/random_scaffold/roberta-base_20_0.3_{seed}_1_True',
            'hiv': f'prt_lm/hiv/random_scaffold/roberta-base_5_0.6_{seed}_0_True'
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
        print('seed: ', self.seed)
        set_seed(self.seed)
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
        lm_path = get_best_lm_path(self.dataset_name, self.split_method, self.seed)
        tsfm = partial(change_target, self.target_task) if self.metrics == 'rmse' else partial(change_dtype, self.target_task)
        dataset = CustomMoleculeNet(name=self.dataset_name, root='./dataset/', lm_path=lm_path, transform=tsfm, pre_filter=valid_smiles_filter)
        
        
        self.dataset = dataset
        self.num_graphs = len(dataset)
        self.num_classes = dataset.y.max().long().item() + 1
        self.num_features = dataset.x.shape[1]
        
        
        # Load pretrained GNN model
        gnn_path = f"gnn_output/{self.dataset_name}/{self.split_method}_{self.gnn_model_name}_{self.gnn_hidden_dim}_{self.seed}.pt"
        print(f'GNN Model is loaded from {gnn_path}')
        self.gnn_model = GCN(in_channels=self.num_features,
                            hidden_channels=self.gnn_hidden_dim,
                            out_channels=self.num_classes,
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
            
            logits, reps = self._forward(data)
            gnn_logits, gnn_reps = self.gnn_model(data.x, data.edge_index, data.batch)
            lm_logits, lm_reps = data.lm_predictions, data.lm_embeddings  # |G| x 1,  |G| x 768
            
            ls_logits = F.log_softmax(logits, dim=1)
            ls_gnn_logits = F.log_softmax(gnn_logits, dim=1)
            ls_lm_logits = F.log_softmax(lm_logits, dim=1)
            
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
            logits, _ = self._forward(data)
            preds = logits.squeeze() if self.metrics == 'rmse' else F.softmax(logits, dim=1)
            target_y = data.y
            val_evaluate.update(preds=preds, target=target_y)
        val_metric = val_evaluate.compute()
        
        test_evaluate = MSE(squared=False).to(self.device) if self.metrics == 'rmse' else AUROC(task="multiclass", num_classes=self.num_classes)
        for data in self.test_loader:
            data = data.to(self.device)
            logits, _ = self._forward(data)
            preds = logits.squeeze() if self.metrics == 'rmse' else F.softmax(logits, dim=1)
            target_y = data.y
            test_evaluate.update(preds=preds, target=target_y)
        test_metric = test_evaluate.compute()            
        return val_metric, test_metric, logits
    

    def train(self):
        if self.metrics == 'auc':
            best_val_metric = 0
            best_test_metric = 0
            for epoch in range(self.epochs + 1):
                train_metric, loss, y_loss, gnn_loss, lm_loss = self._train()
                val_auc, test_auc, _ = self._evaluate()
                
                if val_auc > best_val_metric:
                    best_val_metric = val_auc
                    best_test_metric = test_auc
                    # torch.save(self.model.state_dict(), self.ckpt)
                    
                if epoch % 50 == 0:
                    print(f'Epoch: {epoch}, Loss: {loss:.3f} = y_loss: {y_loss:.3f} + gnn_rep_loss: {gnn_loss:.3f} + lm_rep_loss: {lm_loss:.3f}')
                    print(f'                TrainAUC: {train_metric:.4f}, ValAUC: {val_auc:.4f}, TestAUC: {test_auc:.4f}')
                    print(f'                BestValAUC: {best_val_metric:.4f}, BestTestAUC: {best_test_metric:.4f}')
                    print()   
        return best_val_metric, best_test_metric, self.model


def run(cfg):
    seeds = cfg.seed
    print(seeds) 
    for seed in seeds:
        trainer = DistillTrainer(cfg, seed)
        trainer.train()
        print("-"*100, '\n')
            
        

if __name__ == '__main__':
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*') 
    cfg = update_cfg(cfg)
    run(cfg)

