import random
from collections import defaultdict
from itertools import compress

import numpy as np
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedKFold



def generate_scaffold(smiles, include_chirality=False):
    """ Obtain Bemis-Murcko scaffold from smiles
    :return: smiles of scaffold """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    return scaffold


def scaffold_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
        examples with null value in specified task column of the data.y tensor
        prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
        task_idx is provided
    :param frac_train, frac_valid, frac_test: fractions
    :param return_smiles: return SMILES if Ture
    :return: train, valid, test slices of the input dataset obj. """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    smiles_list = dataset.smiles

    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    # train_dataset = dataset[torch.tensor(train_idx)]
    # valid_dataset = dataset[torch.tensor(valid_idx)]
    # test_dataset = dataset[torch.tensor(test_idx)]  
    return train_idx, valid_idx, test_idx


def random_scaffold_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/
        chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
        examples with null value in specified task column of the data.y tensor
        prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
        task_idx is provided
    :param frac_train, frac_valid, frac_test: fractions, floats
    :param seed: seed
    :return: train, valid, test slices of the input dataset obj
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    rng = np.random.RandomState(seed)
    smiles_list = dataset.smiles
    
    non_null = np.ones(len(dataset)) == 1
    smiles_list = list(compress(enumerate(smiles_list), non_null))

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx, valid_idx, test_idx = [], [], []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    # train_dataset = dataset[torch.tensor(train_idx)]
    # valid_dataset = dataset[torch.tensor(valid_idx)]
    # test_dataset = dataset[torch.tensor(test_idx)]  
    return train_idx, valid_idx, test_idx


def random_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    num_mols = len(dataset)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[:int(frac_train * num_mols)]
    valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols) + int(frac_train * num_mols)]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    # train_dataset = dataset[torch.tensor(train_idx)]
    # valid_dataset = dataset[torch.tensor(valid_idx)]
    # test_dataset = dataset[torch.tensor(test_idx)]
    return train_idx, valid_idx, test_idx


# def cv_random_split(dataset, fold_idx=0,
#                     frac_train=0.9, frac_valid=0.1,
#                     seed=0, smiles_list=None):
#     """
#     :return: train, valid, test slices of the input dataset obj. If
#     smiles_list != None, also returns ([train_smiles_list],
#     [valid_smiles_list], [test_smiles_list]) """

#     np.testing.assert_almost_equal(frac_train + frac_valid, 1.0)

#     skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

#     labels = [data.y.item() for data in dataset]

#     idx_list = []

#     for idx in skf.split(np.zeros(len(labels)), labels):
#         idx_list.append(idx)
#     train_idx, val_idx = idx_list[fold_idx]

#     train_dataset = dataset[torch.tensor(train_idx)]
#     valid_dataset = dataset[torch.tensor(val_idx)]

#     return train_dataset, valid_dataset


def rand_train_test_idx(label, train_prop, valid_prop, test_prop):
    """ randomly splits label into train/valid/test splits """
    labeled_nodes = torch.where(label != -1)[0]

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)
    test_num = int(n * test_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:train_num + valid_num + test_num]

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return {'train': train_idx.numpy(), 'valid': valid_idx.numpy(), 'test': test_idx.numpy()}


def index_to_mask(splits_lst, num_nodes):
    train_mask = torch.zeros((num_nodes), dtype=torch.bool)
    val_mask = torch.zeros((num_nodes), dtype=torch.bool)
    test_mask = torch.zeros((num_nodes), dtype=torch.bool)

    train_mask[splits_lst['train']] = True
    val_mask[splits_lst['valid']] = True
    test_mask[splits_lst['test']] = True

    return train_mask, val_mask, test_mask