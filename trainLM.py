import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from lm_models import BertClassifier, BertClaInfModel
# from core.data_utils.dataset import Dataset
# from core.data_utils.load import load_data
from utils import init_path, time_logger, get_gpt_response, set_seed, change_dtype, get_valid_smiles, get_target_label, valid_smiles_filter
# from gnn_utils import Evaluator
from torch_geometric.datasets import MoleculeNet
from functools import partial
from dataset import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from splitter import index_to_mask, rand_train_test_idx
from loader import split_loader
from config import cfg, update_cfg
import pandas as pd

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# def transform(mol, dataname, tokenizer):
#     prompt, response = get_llm_response(dataname, mol.smiles)
#     response = tokenizer(response, padding=True, truncation=True, max_length=1024)
#     mol.response = response
#     return mol


# def compute_metrics(p):
#     pred, labels = p
#     pred = np.argmax(pred, axis=1)
#     accuracy = accuracy_score(y_true=labels, y_pred=pred)
#     return {"accuracy": accuracy}


def compute_metrics(p):
    labels = p.label_ids
    pred = p.predictions.argmax(-1)
    auc = roc_auc_score(y_true=labels, y_score=pred)
    return {'auc': auc}


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
        
        
        self.output_dir = f'lm_output/{self.dataset_name}/{self.target_task}/{self.split_method}/{self.model_name}_{self.epochs}_{self.warmup_epochs}_{self.seed}'
        self.ckpt_dir = f'prt_lm/{self.dataset_name}/{self.split_method}/{self.model_name}_{self.epochs}_{self.warmup_epochs}_{self.seed}_{self.target_task}_{self.diagram}'


        # Load and Preprocess data
        data = MoleculeNet(name=self.dataset_name, root='./dataset/', pre_filter=valid_smiles_filter)
        data.y = data.y[:, self.target_task].long()  
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
            prompt, response = get_gpt_response(self.dataset_name, mol.smiles, diagram=cfg.lm.train.diagram)
            responses.append(response)
            
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        X = tokenizer(responses, padding=True, truncation=True)
        dataset = Dataset(X, data.y.tolist())
        self.dataset = dataset
        
        self.train_dataset = torch.utils.data.Subset(dataset, self.train_idx)
        self.val_dataset = torch.utils.data.Subset(dataset, self.val_idx)
        self.test_dataset = torch.utils.data.Subset(dataset, self.test_idx)
        

        # Define pretrained tokenizer and model
        bert_model = AutoModel.from_pretrained(self.model_name)
        self.model = BertClassifier(bert_model,
                                    n_labels=self.n_labels,
                                    feat_shrink=self.feat_shrink)

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
            # dataloader_drop_last=True,
            metric_for_best_model='auc',
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
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
        pred = np.memmap(init_path(f"{self.ckpt_dir}.pred"),
                         dtype=np.float16,
                         mode='w+',
                         shape=(self.num_nodes, self.n_labels))

        inf_model = BertClaInfModel(self.model, emb, pred, feat_shrink=self.feat_shrink)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.dataset)


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



def run(cfg):
    # seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
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
            print(f"{k}: {v.mean()*100:.2f}±{v.std()*100:.2f}")
        
        path = f'prt_lm_results/{cfg.dataset.name}/{cfg.dataset.split_method}/{cfg.lm.model.name}_{cfg.lm.train.epochs}_{cfg.lm.train.warmup_epochs}_{cfg.dataset.target_task}_{cfg.lm.train.diagram}.csv'
        df.to_csv(path, index=False)


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)

