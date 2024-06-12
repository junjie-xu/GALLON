# Multi-modal Knowledge Distillation for Molecular Property Prediction
An official PyTorch implementation of the paper [Multi-modal Knowledge Distillation for Molecular Property Prediction].




## Contribution

+ We propose utilizing multimodal molecular data to learn representations and extract prior knowledge from powerful and pretrained large language models by leveraging their multimodality capabilities.

+ We introduce a framework that synergizes GNNs and LLMs, leveraging their complementary strengths. This framework distills the advantages of GNN and LLM into an MLP, aiming to capture the most effective representations for molecular structures.

+ We conduct extensive experiments to demonstrate the superiority of our approach in distilling GNN and LLM knowledge into an MLP, which outperforms both GNNs and LLMs across various datasets, achieving greater efficiency and an even smaller model size.



## Installation
Create a new virtual environment.
```
conda create --name molecule python=3.11
```

Optinal 1: Install via environment.yml
```
conda env create -f environment.yml
```

Optional 2: Install manually.
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install rdkit transformers torchmetrics openai anthropic 
```

## Running Experiments

1. To generate molecular diagrams:
```
python preprocess/visualize.py --dataname $dataname
```

2. To get responses from LLM, use one of the following commands:
```
python preprocess/query_gpt.py --dataname $dataname
python preprocess/claude.py --dataname $dataname
```
To access our responses, unzip gpt_response and claude_response in this folder.

3. To pretrain LM, 
```
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0 python -m trainLM dataset.name $dataname lm.train.diagram True dataset.target_task 0 dataset.split_method scaffold lm.train.warmup_epochs 0.3 lm.train.epochs 5
```

3. To pretrain GNN, 
```
python trainGNN.py dataset.name $dataname dataset.split_method scaffold gnn.model.hidden_dim 32 device cuda:0
```

4. To distill from LLM and GNN to MLP for classification tasks:
```
python trainDistill_class.py dataset.name bace device cuda:0 dataset.split_method scaffold
```
For regression:
```
python trainDistill_regression.py dataset.name esol device cuda:0 dataset.split_method scaffold
```





