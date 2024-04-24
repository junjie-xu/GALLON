import optuna
import numpy as np

best_all = []
best_gnn_only = []
best_lm_only = []

for seed in [0,1,2,3,4]:

    name = f'optuna_search/lipo/mode1/random_scaffold_64_{seed}'
    print("Study Name: ", name)

    study = optuna.create_study(direction="minimize",
                                study_name=name,
                                storage="sqlite:///" + name + ".db",
                                load_if_exists=True)

    df = study.trials_dataframe()
    sorted_df = df.sort_values(by='value', ascending=True)
    
    # print(sorted_df.head())
    
    all, gnn_only, lm_only = None, None, None
    for index, row in sorted_df.iterrows():
        # print(row['value'], row['params_alpha'], row['params_beta'])
        if (row['params_alpha'] == 0) and (row['params_beta'] != 0) and lm_only is None:
            lm_only = row['value']
        if (row['params_beta'] == 0) and (row['params_alpha'] != 0) and gnn_only is None:
            gnn_only = row['value']
        if (row['params_alpha'] != 0) and (row['params_beta'] != 0) and all is None:
            all = row['value']
            
        if all and gnn_only and lm_only:
            break
        
    print(all, gnn_only, lm_only)
        
    best_all.append(all)
    best_gnn_only.append(gnn_only)
    best_lm_only.append(lm_only)
    

best_all = np.array(best_all)
best_gnn_only = np.array(best_gnn_only)
best_lm_only = np.array(best_lm_only)

scale = 1

print(f"Best All: {best_all.mean()*scale:.2f}±{best_all.std()*scale:.2f}")
print(f"Best GNN Only: {best_gnn_only.mean()*scale:.2f}±{best_gnn_only.std()*scale:.2f}")
print(f"Best LM Only: {best_lm_only.mean()*scale:.2f}±{best_lm_only.std()*scale:.2f}")
print()



    
    
