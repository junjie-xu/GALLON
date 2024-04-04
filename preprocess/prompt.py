from torch_geometric.datasets import MoleculeNet
import os
import numpy as np
from torch_geometric.utils.smiles import x_map, e_map
from rdkit.Chem import GetPeriodicTable



task_mapping = {
    # Regression
    'esol': ("Based on these inputs, please analyze the propoerty of the molecule (e.g. Functional groups, Structural   characteristics), and which proporties of the molecule can affect its water solubility? Also try to guess its solubility. "
    ),
    
    'lipo': ("Lipophilicity is an important feature of drug molecules that affects both membrane permeability and solubility. " 
             "Based on the following inputs, please analyze the molecule and give some details of factors that can affect " 
             "octanol/water distribution coeffcient (logD at pH 7.4). Then make your guess or prediction about its lipophilicity. "
             ),
    
    'freesolv': ("Free Solvation Database (FreeSolv) provides experimental and calculated hydration free energy of small molecules in water. Based on the following inputs, please analyze the propoerty of the molecule (e.g. Functional groups, Structural characteristics) with focusing on its hydration free energy. Then make your guess or prediction about its hydration free energy. "
             ),
    
    # Classification
    'bbbp': ("As a membrane separating circulating blood and brain extracellular ﬂuid, "
            "the blood-brain barrier blocks most drugs, hormones, and neurotransmitters. "
            "Based on these inputs, please analyze the propoerty of the molecule (e.g. Functional groups, Structural characteristics) and analyze if the molecule is permeable to the blood-brain barrier?"
            ),
    
    'hiv': ("The HIV dataset tests the ability to inhibit HIV replication for over 40,000 compounds. Based on the following inputs, please analyze the propoerty of the molecule (e.g. Functional groups, Structural characteristics) with focusing on its ability to inhibit HIV replication. Then make your guess or prediction (active or inactive)."
            ), 
    
    'clintox': ("Could you analyze the given molecule based on the provided inputs and detail the factors influencing its potential for clinical trial toxicity or non-toxicity? Additionally, please assess factors that might impact its FDA approval status."
            ),
    
    'bace': ("The BACE dataset provides and binary label binding results for a set of inhibitors of human β-secretase 1 (BACE-1)"  
             "Based on the following inputs, please analyze the propoerty of the molecule (e.g. Functional groups, Structural characteristics) and analyze the binding results for a set of inhibitors of human beta-secretase (BACE-1)?"
    ),
}


def describe_structure(mol, stereo=False, conjugate=False):
    description = ""
    pt = GetPeriodicTable()
    for i in range(0, len(mol.edge_index[0])):
        src, tgt = mol.edge_index[0][i], mol.edge_index[1][i]
        src_name, tgt_name = pt.GetElementName(mol.x[src][0].item()), pt.GetElementName(mol.x[tgt][0].item())
        bond_type = e_map['bond_type'][mol.edge_attr[i][0]]
        
        if stereo:
            pass
            # TODO: stereo bond
        
        if conjugate:
            bond_conj = "conjugated" if e_map['is_conjugated'][mol.edge_attr[i][2]] else "non-conjugated"
            bond_conj = " and " + bond_conj 
        else:
            bond_conj = ""
                  
        dcp = f"Atom {src} ({src_name}) is connected with Atom {tgt} ({tgt_name}) by a(n) {bond_type}{bond_conj} bond. \n"
        description += dcp
    return description



def construct_prompt(dataname, mol, diagram=False, structure=False):
    prompt = ""
    prompt += ( "Assuming you are an expert in chemistry, molecules, and AI, "
                "I have a dataset of molecules for which I'd like to predict a specific property. "
                "Given a molecule, I can provide you with the SMILES "
                "(Simplified molecular-input line-entry system) string of the molecule, ")
    if diagram:
        prompt +=  "and a diagram representing its structure"
    if structure:
        prompt += "and the adjacency matrix depicting its graph structure. \n "
    
    prompt += task_mapping[dataname]  
    prompt += " Here are the details. \n "
    
    prompt += "SMILES: " + mol.smiles + "\n "
    if diagram:
        prompt += "Molecule Diagram: Attached is the image of the molecule's diagram. \n "
    if structure:
        prompt += "Graph Structure: Here is a list of connected atoms with their bond types. \n" \
                    + describe_structure(mol) + "\n "
    
    # print(len(prompt))
    # print(prompt)
    
    return prompt


if __name__ == "__main__":

    dataname = 'bbbp'
    dataset = MoleculeNet(name=dataname, root='./dataset/')

    mapping_path = os.path.join('./diagram', dataname, 'mapping.npy')
    mapping_dict = np.load(mapping_path, allow_pickle=True).item()
    print(dataset[0], dataset[0].y)


    for i, mol in enumerate(dataset):
        prompt = construct_prompt(dataname, mol, diagram=False, structure=True)
        print(prompt)
        break




