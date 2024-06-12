import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from lm_models import BertClassifier, BertClaInfModel, BertRegInfModel, BertRegressor
from utils import init_path, get_gpt_response, set_seed, valid_smiles_filter, get_claude_response
from torch_geometric.datasets import MoleculeNet
from dataset import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from loader import split_loader
from config import cfg, update_cfg
import pandas as pd

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
SCALE = 100



def compute_metrics_auc(p):
    labels = p.label_ids
    pred = p.predictions.argmax(-1)
    auc = roc_auc_score(y_true=labels, y_score=pred)
    return {'auc': auc}


def compute_metrics_rmse(p):
    pred, labels = p
    rmse = np.sqrt(((pred - labels) ** 2).mean())
    return {"rmse": rmse}


class LMTrainer():
    def __init__(self, cfg):
        self.seed = cfg.seed
        set_seed(cfg.seed)
        print(self.seed)
        
        self.dataset_name = cfg.dataset.name
        self.target_task = cfg.dataset.target_task 
        self.model_name = cfg.lm.model.name
        self.feat_shrink = cfg.lm.model.feat_shrink

        self.weight_decay = cfg.lm.train.weight_decay
        self.dropout = cfg.lm.train.dropout
        self.att_dropout = cfg.lm.train.att_dropout
        self.cla_dropout = cfg.lm.train.cla_dropout
        self.batch_size = cfg.lm.train.batch_size
        self.epochs = cfg.lm.train.epochs
        self.warmup_epochs = cfg.lm.train.warmup_epochs
        self.eval_patience = cfg.lm.train.eval_patience
        self.grad_acc_steps = cfg.lm.train.grad_acc_steps
        self.lr = cfg.lm.train.lr
        self.split_method = cfg.dataset.split_method
        self.diagram = cfg.lm.train.diagram
        self.structure = cfg.lm.train.structure
        
        if self.dataset_name in ['esol', 'lipo', 'freesolv']:
            self.metrics = 'rmse'
        else:
            self.metrics = 'auc'
        
        
        self.output_dir = f'lm_output/{self.dataset_name}/{self.target_task}/{self.split_method}/{self.model_name}_{self.epochs}_{self.warmup_epochs}_{self.seed}'
        self.ckpt_dir = f'prt_lm/{self.dataset_name}/{self.split_method}/{self.model_name}_{self.epochs}_{self.warmup_epochs}_{self.seed}_{self.target_task}_{self.diagram}_{self.structure}'


        # Load and Preprocess data
        data = MoleculeNet(name=self.dataset_name, root='./dataset/', pre_filter=valid_smiles_filter)
        data.y = data.y[:, self.target_task]
        if self.metrics == 'auc':
            data.y = data.y.long()
        self.data = data
        self.n_labels = data.y.max().item() + 1
        self.num_nodes = data.num_nodes = len(data)    
        
        self.train_idx, self.val_idx, self.test_idx = split_loader(dataset=self.data, 
                                                                    split_method=self.split_method,
                                                                    frac_train=cfg.dataset.train_prop, 
                                                                    frac_val=cfg.dataset.val_prop, 
                                                                    frac_test=cfg.dataset.test_prop,
                                                                    seed=self.seed)
        
        # Construct new dataset
        responses = []
        for mol in data:
            if self.dataset_name in []:
                prompt, response = get_claude_response(self.dataset_name, mol.smiles, diagram=self.diagram)
            # elif self.dataset_name in ['bace', 'bbbp']:
            #     prompt, response = get_claude_response_with_model(self.dataset_name, 'sonnet', mol.smiles, diagram=self.diagram)
            else:
                prompt, response = get_gpt_response(self.dataset_name, mol.smiles, diagram=self.diagram, structure=self.structure)
            response = f'The original SMILES string: {mol.smiles} \n' + response
            responses.append(response)
        print("Load LLM response done.")
            
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        X = tokenizer(responses, padding=True, truncation=True)
        dataset = Dataset(X, data.y.tolist())
        self.dataset = dataset
        
        self.train_dataset = torch.utils.data.Subset(dataset, self.train_idx)
        self.val_dataset = torch.utils.data.Subset(dataset, self.val_idx)
        self.test_dataset = torch.utils.data.Subset(dataset, self.test_idx)
        

        # Define pretrained tokenizer and model
        bert_model = AutoModel.from_pretrained(self.model_name)
        
        if self.metrics == 'auc':
            self.model = BertClassifier(bert_model, n_labels=self.n_labels, feat_shrink=self.feat_shrink)
        else:
            self.model = BertRegressor(bert_model, feat_shrink=self.feat_shrink)

        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")

    # @time_logger
    def train(self):
        # Define training parameters
        eq_batch_size = self.batch_size * 4
        train_steps = self.num_nodes // eq_batch_size + 1
        eval_steps = self.eval_patience // eq_batch_size
        warmup_steps = int(self.warmup_epochs * train_steps)

        # Define Trainer
        args = TrainingArguments(
            output_dir=self.output_dir,
            # save_strategy="no",
            do_train=True,
            do_eval=True,
            eval_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.grad_acc_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size*8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=1,
            fp16=True,
            metric_for_best_model=self.metrics,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics_auc if self.metrics == 'auc' else compute_metrics_rmse,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        self.trainer.train()
        torch.save(self.model.state_dict(), init_path(f"{self.ckpt_dir}.ckpt"))
        print(f'LM saved to {self.ckpt_dir}.ckpt')

    # @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        emb = np.memmap(init_path(f"{self.ckpt_dir}.emb"),
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, self.feat_shrink if self.feat_shrink else 768))
        if self.metrics == 'auc':
            pred = np.memmap(init_path(f"{self.ckpt_dir}.pred"),
                            dtype=np.float16,
                            mode='w+',
                            shape=(self.num_nodes, self.n_labels))
            inf_model = BertClaInfModel(self.model, emb, pred, feat_shrink=self.feat_shrink)
        else:
            pred = np.memmap(init_path(f"{self.ckpt_dir}.pred"),
                            dtype=np.float16,
                            mode='w+',
                            shape=(self.num_nodes))
            inf_model = BertRegInfModel(self.model, emb, pred, feat_shrink=self.feat_shrink)
            
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            # save_strategy="no",
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.dataset)

        if self.metrics == 'auc':

            def eval(msk):
                labels = self.data.y[msk]
                y_true = labels.view(-1).detach().cpu().numpy()
                
                y_pred = torch.softmax(torch.from_numpy(pred[msk]).float(), dim=1)
                y_pred = y_pred[:, 1]
                y_pred = y_pred.detach().cpu().numpy()     
                auc = roc_auc_score(y_true=y_true, y_score=y_pred)
                return auc
            
            train_acc = eval(self.train_idx)
            val_acc = eval(self.val_idx)
            test_acc = eval(self.test_idx)
            print(f'[LM] TrainAuc: {train_acc:.4f}, ValAuc: {val_acc:.4f}, TestAuc: {test_acc:.4f}\n')
            return {'TrainAuc': train_acc, 'ValAuc': val_acc, 'TestAuc': test_acc}
        
        else:
            def eval(msk):
                labels = self.data.y[msk]
                y_true = labels.view(-1).detach().cpu().numpy()
                
                y_pred = torch.from_numpy(pred[msk]).float()
                y_pred = y_pred.view(-1).detach().cpu().numpy()
                
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                return rmse
            
            train_rmse = eval(self.train_idx)
            val_rmse = eval(self.val_idx)
            test_rmse = eval(self.test_idx)
            print(f'[LM] TrainRMSE: {train_rmse:.4f}, ValRMSE: {val_rmse:.4f}, TestRMSE: {test_rmse:.4f}\n')
            return {'TrainRMSE': train_rmse, 'ValRMSE': val_rmse, 'TestRMSE': test_rmse}

        

def run(cfg):
    seeds = cfg.seed
    all_acc = []
    print(seeds)
    for seed in seeds:
        cfg.seed = seed
        trainer = LMTrainer(cfg)
        trainer.train()
        acc = trainer.eval_and_save()
        all_acc.append(acc)
        print("-"*100, '\n')

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        for k, v in df.items():
            print(f"{k}: {v.mean()*SCALE:.2f}±{v.std()*SCALE:.2f}")
        
        path = f'prt_results/prt_lm_results/{cfg.dataset.name}/{cfg.lm.model.name}_{cfg.dataset.split_method}_{cfg.lm.train.epochs}_{cfg.lm.train.warmup_epochs}_{cfg.dataset.target_task}_{cfg.lm.train.diagram}_{cfg.lm.train.structure}.csv'
        df.to_csv(path, index=False)


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)


