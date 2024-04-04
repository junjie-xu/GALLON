from torch_geometric.datasets import MoleculeNet
from rdkit import Chem
from rdkit.Chem import Draw
import os
import numpy as np


def valid_smiles_filter(data):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(data.smiles)
    return mol is not None


dataname = 'hiv'

dataset = MoleculeNet(name=dataname, root='./dataset/', pre_filter=valid_smiles_filter)


dictionary = {}
for i, mol in enumerate(dataset):
    dictionary[mol.smiles] = i
# print(dictionary)

mapping_path = os.path.join('./diagram', dataname, 'mapping.npy')
np.save(mapping_path, dictionary)

dictionary = np.load(mapping_path, allow_pickle=True).item()
# print(read_dictionary)

    
    



image_size = (500, 500)  # Width, Height in pixels
# options = Draw.DrawingOptions()
# options.bondLineWidth = 10.0
# options.atomLabelFontSize = 12


for i, mol in enumerate(dataset):
    m = Chem.MolFromSmiles(mol.smiles)
    if m:
        file_path = os.path.join('./diagram', dataname, str(dictionary[mol.smiles]) + '.png')
        Draw.MolToFile(m, file_path, size=image_size, imageType='png')
    else:
        print(f"No. {i}'s molecule has invalid SMILES string.'")



