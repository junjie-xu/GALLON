import os
import json
from easydict import EasyDict
import simplejson
import numpy as np



dataname = 'clintox'
index = 0

json_path = os.path.join('./llm_response', dataname, str(index) + '.json')

with open(json_path, 'r') as json_file:
    response_loaded = EasyDict(simplejson.load(json_file))

prompt = response_loaded['Prompt']
response = response_loaded['Response']#.replace("'", "\"")
response = EasyDict(simplejson.loads(response))


print(prompt)
print(response)
# print(response.choices)
# print(response.choices[0])
# print(response.choices[0].message.content)



def get_llm_response(dataname, smiles):
    mapping_path = os.path.join('./diagram', dataname, 'mapping.npy')
    mapping = np.load(mapping_path, allow_pickle=True).item()
    

    json_path = os.path.join('./llm_response', dataname, str(mapping[smiles]) + '.json')
    with open(json_path, 'r') as json_file:
        response_loaded = EasyDict(simplejson.load(json_file))

    prompt = response_loaded['Prompt']
    response = response_loaded['Response']#.replace("'", "\"")
    response = EasyDict(simplejson.loads(response))
    
    return prompt, response
    