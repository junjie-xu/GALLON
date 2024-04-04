
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.datasets import MoleculeNet
import numpy as np
from utils import get_valid_smiles


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])



class CustomMoleculeNet(MoleculeNet):
    def __init__(self, root, name, lm_path, transform=None, pre_transform=None, pre_filter=None):
        super(CustomMoleculeNet, self).__init__(root, name, transform, pre_transform, pre_filter)
        l = super().__len__()
        c = self.data.y.max().long().item() + 1      
        lm_predictions = torch.from_numpy(np.memmap(lm_path, dtype=np.float16, shape=(l, c)))
        print(f'Logits of LM is loaded from {lm_path}')
        self.lm_predictions = lm_predictions.unsqueeze(1)
            
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.lm_prediction = self.lm_predictions[idx]
        return data
    


