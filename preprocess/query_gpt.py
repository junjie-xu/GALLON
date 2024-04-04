import os
import json
import base64
import argparse
import numpy as np
from openai import OpenAI
from prompt import construct_prompt
from torch_geometric.datasets import MoleculeNet
from rdkit import Chem



def valid_smiles_filter(data):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(data.smiles)
    return mol is not None


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def query(args):
    dataname = args.dataname
    dataset = MoleculeNet(name=dataname, root='./dataset/', pre_filter=valid_smiles_filter)
    
    mapping_path = os.path.join('./diagram', dataname, 'mapping.npy')
    mapping_dict = np.load(mapping_path, allow_pickle=True).item()
    print(dataset, '\n')
    
    client = OpenAI()

    for _, mol in enumerate(dataset[args.begin:]):
        print("-"*80)
        index = mapping_dict[mol.smiles]
        print(f"index: {index}, \n {mol}")
        text_prompt = construct_prompt(dataname, mol, args.diagram, args.structure)
        
       
        image_path = os.path.join('./diagram', dataname, str(index) + '.png')
        if args.diagram and os.path.exists(image_path):
            base64_image = encode_image(image_path)
            response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"
                    },
                    },
                ],
                }
            ],
            max_tokens=700,
            )
        else:
            response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                ],
                }
            ],
            max_tokens=700,
            )          
            
            
        print(text_prompt)
        print(response.choices[0].message.content)
        
        generated_text = response.model_dump_json()
        # full_text = f"Prompt: {text_prompt}\nResponse: {generated_text}"
        full_text ={
            'Prompt': text_prompt,
            'Response': generated_text
        }
        
        if args.diagram:
            json_path = os.path.join('./llm_response', dataname, str(index) + '.json')
        else:
            json_path = os.path.join('./llm_response', dataname, 'no_diagram', str(index) + '.json')
        
        with open(json_path, 'w') as json_file:
            json.dump(full_text, json_file)
        print(f"GPT response of molecule {index} stored in {json_path}. \n\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--dataname", type=str, default='lipo')
    parser.add_argument('--diagram', action='store_true', help='default is false')
    parser.add_argument('--structure', action='store_true', help='default is false')
    
    args = parser.parse_args()
    print(args)

    query(args)


