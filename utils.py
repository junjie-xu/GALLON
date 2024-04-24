import os
import numpy as np
import time
import datetime
# import pytzv
from easydict import EasyDict
import simplejson
import numpy as np
import torch
import random
from rdkit import Chem
from typing import Optional
import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform



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


def get_valid_smiles(dataset):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*') 
    dataset_mask = torch.ones(len(dataset), dtype=torch.bool)
    for i, mol in enumerate(dataset):
        m = Chem.MolFromSmiles(mol.smiles)
        if not m:
            dataset_mask[i] = False         
    dataset = dataset[dataset_mask]
    return dataset


def get_gpt_response(dataname, smiles, diagram=True):
    mapping_path = os.path.join('./diagram', dataname, 'mapping.npy')
    mapping = np.load(mapping_path, allow_pickle=True).item()
        
    if diagram:
        json_path = os.path.join('./gpt_response', dataname, str(mapping[smiles]) + '.json')
    else:
        json_path = os.path.join('./gpt_response', dataname, 'no_diagram', str(mapping[smiles]) + '.json')
        
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


def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
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



def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='America/New_York', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret
    return wrapper




# @functional_transform('to_dense')
class ToDense(BaseTransform):
    def __init__(self, num_nodes: Optional[int] = None):
        self.num_nodes = num_nodes

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None

        orig_num_nodes = data.num_nodes
        if self.num_nodes is None:
            num_nodes = orig_num_nodes
        else:
            assert orig_num_nodes <= self.num_nodes
            num_nodes = self.num_nodes

        # if data.edge_attr is None:
        #     edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.float)
        # else:
        #     edge_attr = data.edge_attr
        edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.float)

        size = torch.Size([num_nodes, num_nodes] + list(edge_attr.size())[1:])
        adj = torch.sparse_coo_tensor(data.edge_index, edge_attr, size)
        data.adj = adj.to_dense()
        data.edge_index = None
        data.edge_attr = None

        data.mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.mask[:orig_num_nodes] = 1

        if data.x is not None:
            size = [num_nodes - data.x.size(0)] + list(data.x.size())[1:]
            # print("size: ", size)
            data.x = torch.cat([data.x, data.x.new_zeros(size)], dim=0)

        if data.pos is not None:
            size = [num_nodes - data.pos.size(0)] + list(data.pos.size())[1:]
            data.pos = torch.cat([data.pos, data.pos.new_zeros(size)], dim=0)

        # if data.y is not None and (data.y.size(0) == orig_num_nodes):
        #     print("1: ", data.y.shape)
        #     size = [num_nodes - data.y.size(0)] + list(data.y.size())[1:]
        #     # print("size: ", size)
        #     data.y = torch.cat([data.y, data.y.new_zeros(size)], dim=0)
        #     print("2: ", data.y.shape)

        return data

    def __repr__(self) -> str:
        if self.num_nodes is None:
            return f'{self.__class__.__name__}()'
        return f'{self.__class__.__name__}(num_nodes={self.num_nodes})'


