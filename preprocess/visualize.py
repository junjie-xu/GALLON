import os
import argparse
import numpy as np
from torch_geometric.datasets import MoleculeNet
from rdkit import Chem
from rdkit.Chem import Draw


def valid_smiles_filter(data):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(data.smiles)
    return mol is not None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataname", type=str, default='bace')    
    args = parser.parse_args()
    
    dataname = args.dataname

    dataset = MoleculeNet(name=dataname, root='./dataset/', pre_filter=valid_smiles_filter)


    dictionary = {}
    for i, mol in enumerate(dataset):
        dictionary[mol.smiles] = i
    # print(dictionary)

    mapping_path = os.path.join('./diagram', dataname, 'mapping.npy')
    np.save(mapping_path, dictionary)

    dictionary = np.load(mapping_path, allow_pickle=True).item()


    image_size = (500, 500)  # Width, Height in pixels
    # options = Draw.DrawingOptions()
    # options.bondLineWidth = 10.0
    # options.atomLabelFontSize = 12


    for i, mol in enumerate(dataset):
        m = Chem.MolFromSmiles(mol.smiles)
        if m:
            file_path = os.path.join('./diagram', dataname, str(dictionary[mol.smiles]) + '.png')
            Draw.MolToFile(m, file_path, size=image_size, imageType='png')
            print(f"No. {i}'s molecule saved.'")
        else:
            print(f"No. {i}'s molecule has invalid SMILES string.'")

