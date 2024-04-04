import optuna
# import pandas as pd



name = 'optuna_search/bbbp_random_scaffold_32'
print("Study Name: ", name)

study = optuna.create_study(direction="maximize",
                            study_name=name,
                            storage="sqlite:///" + name + ".db",
                            load_if_exists=True)

df = study.trials_dataframe()
print(df)

# df.to_csv(f'{name}.csv', index=False)


print("best params ", study.best_params)
print("best valf1 ", study.best_value)
print('\n')
    
    
