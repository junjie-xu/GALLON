import torch
import pandas as pd
from config import cfg


# for i in ['5', '10', '20']:
#     for j in ['0.3', '0.6']:
#         path = f'prt_lm_results/clintox/random/roberta-base_{i}_{j}_1_False.csv'
#         print(path)
#         df = pd.read_csv(path)
#         print(df)

#         for k, v in df.items():
#             print(f"{k}: {v.mean()*100:.2f}±{v.std()*100:.2f}")
        
#         print()
        



path = f'prt_lm_results/clintox/random/roberta-base_20_0.3_1_True.csv'
print(path)
df = pd.read_csv(path)
print(df)

for k, v in df.items():
    print(f"{k}: {v.mean()*100:.2f}±{v.std()*100:.2f}")

        
        
