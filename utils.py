import os
import numpy as np
from easydict import EasyDict
import simplejson
import numpy as np
import torch
import random
from rdkit import Chem
import torch
import errno



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def change_dtype(task_idx, data):
    data.x = data.x.to(torch.float32)
    data.y = data.y[:, task_idx].to(torch.int64)
    return data

def change_target(task_idx, data):
    data.x = data.x.to(torch.float32)
    data.y = data.y[:, task_idx]
    return data


def valid_smiles_filter(data):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(data.smiles)
    return mol is not None


def get_target_label(task_idx, data):
    data.y = data.y[:, task_idx].long()
    return data


# def get_valid_smiles(dataset):
#     from rdkit import RDLogger
#     RDLogger.DisableLog('rdApp.*') 
#     dataset_mask = torch.ones(len(dataset), dtype=torch.bool)
#     for i, mol in enumerate(dataset):
#         m = Chem.MolFromSmiles(mol.smiles)
#         if not m:
#             dataset_mask[i] = False         
#     dataset = dataset[dataset_mask]
#     return dataset


def get_gpt_response(dataname, smiles, diagram=True, structure=True):
    mapping_path = os.path.join('./diagram', dataname, 'mapping.npy')
    mapping = np.load(mapping_path, allow_pickle=True).item()
        
    if diagram and structure:
        json_path = os.path.join('./gpt_response', dataname, str(mapping[smiles]) + '.json')
    elif (not diagram) and structure:
        json_path = os.path.join('./gpt_response', dataname, 'no_diagram', str(mapping[smiles]) + '.json')
    elif diagram and (not structure):
        json_path = os.path.join('./gpt_response', dataname, 'no_structure', str(mapping[smiles]) + '.json')
        
    with open(json_path, 'r') as json_file:
        response_loaded = EasyDict(simplejson.load(json_file))

    prompt = response_loaded['Prompt']
    response = response_loaded['Response']   #.replace("'", "\"")
    response = EasyDict(simplejson.loads(response))
    return prompt, response.choices[0].message.content


def get_claude_response(dataname, smiles, diagram=True):
    mapping_path = os.path.join('./diagram', dataname, 'mapping.npy')
    mapping = np.load(mapping_path, allow_pickle=True).item()
        
    if diagram:
        json_path = os.path.join('./claude_response', dataname, str(mapping[smiles]) + '.json')
        
    with open(json_path, 'r') as json_file:
        response_loaded = EasyDict(simplejson.load(json_file))

    prompt = response_loaded['Prompt']
    response = response_loaded['Response']
    response = EasyDict(simplejson.loads(response))
    return prompt, response.content[0]['text']


def get_claude_response_with_model(dataname, modelname, smiles, diagram=True):
    mapping_path = os.path.join('./diagram', dataname, 'mapping.npy')
    mapping = np.load(mapping_path, allow_pickle=True).item()
        
    if diagram:
        json_path = os.path.join('./claude_response', dataname, modelname, str(mapping[smiles]) + '.json')
       
    with open(json_path, 'r') as json_file:
        response_loaded = EasyDict(simplejson.load(json_file))

    prompt = response_loaded['Prompt']
    response = response_loaded['Response']
    response = EasyDict(simplejson.loads(response))
    return prompt, response.content[0]['text']


def mkdir_p(path, log=True):
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file





