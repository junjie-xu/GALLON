import os
import json
import base64
import argparse
import numpy as np
import anthropic
from prompt import construct_prompt
from torch_geometric.datasets import MoleculeNet
from rdkit import Chem

model_map = {
    'haiku': "claude-3-haiku-20240307",
    'sonnet': "claude-3-sonnet-20240229",
    'opus': "claude-3-opus-20240229",
}


def valid_smiles_filter(data):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(data.smiles)
    return mol is not None


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def query(args, model_name):
    dataname = args.dataname
    model = model_map[model_name]
    dataset = MoleculeNet(name=dataname, root='./dataset/', pre_filter=valid_smiles_filter)
    
    mapping_path = os.path.join('./diagram', dataname, 'mapping.npy')
    mapping_dict = np.load(mapping_path, allow_pickle=True).item()
    print(dataset, '\n')
    
    client = anthropic.Anthropic()

    for _, mol in enumerate(dataset[args.begin:]):
        print("-"*80)
        index = mapping_dict[mol.smiles]
        print(f"index: {index}, \n {mol}")
        text_prompt = construct_prompt(dataname, mol, args.diagram, args.structure)
        
       
        image_path = os.path.join('./diagram', dataname, str(index) + '.png')
        
        if args.diagram and os.path.exists(image_path):
            base64_image = encode_image(image_path)
            message = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt}, 
                            {"type": "image",
                                "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,},
                            },  
                        ],
                    }
                ],
            )  
        
        else:
            message = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "user",
                     "content": [{"type": "text", "text": text_prompt}],
                    }
                ],
            ) 
                   
            
        print(text_prompt)
        print(message.content[0].text)
        print(message.usage)
        
        generated_text = message.model_dump_json()
        full_text ={
            'Prompt': text_prompt,
            'Response': generated_text
        }
        
        if args.diagram:
            json_path = os.path.join('./claude_response', dataname, model_name, str(index) + '.json')
        else:
            json_path = os.path.join('./claude_response', dataname, model_name, 'no_diagram', str(index) + '.json')
        
        with open(json_path, 'w') as json_file:
            json.dump(full_text, json_file)
        print(f"Claude response of molecule {index} stored in {json_path}. \n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--dataname", type=str, default='lipo')
    parser.add_argument('--diagram', action='store_true', help='default is false')
    parser.add_argument('--structure', action='store_true', help='default is false')
    
    args = parser.parse_args()
    print(args)
    query(args, 'haiku')


