import torch
from models import GCN
import optuna
import argparse
from config_parser import parser_add_main_args
from rdkit import Chem
from torch_geometric.datasets import MoleculeNet
from utils import valid_smiles_filter
import os
import numpy as np
from easydict import EasyDict
import simplejson
import shutil


dataname = 'bbbp'
dataset = MoleculeNet(name=dataname, root='./dataset/', pre_filter=valid_smiles_filter)
max_i = len(dataset)
LEN = 2050

# for i in range(LEN):
#     full_path = os.path.join(folder_path, str(i)+'.json')
#     if os.path.isfile(full_path):
#         print(f"Found file: {full_path}")




i = 0 # index of filtered dataset
j = 0 # index of full dataset

while i < max_i:
    target_path = os.path.join('./llm_response', dataname, 'no_diagram', str(i)+'.json')
    while True:
        source_path = os.path.join('./llm_response', dataname+'_old', 'no_diagram', str(j)+'.json')
        
        if os.path.exists(source_path):
            with open(source_path, 'r') as json_file:
                response_loaded = EasyDict(simplejson.load(json_file))
            prompt = response_loaded['Prompt']
        else:
            j += 1
            continue
        
        if dataset[i].smiles in prompt:
            shutil.copy2(source_path, target_path)
            i += 1
            j += 1
            break
        else:
            j += 1
            
            
    

