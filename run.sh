
nohup python -u trainDistill_optuna.py --dataname bbbp --distill_hidden 32 --split_method random_scaffold --device cuda:0  > optuna_search/out_bbbp_32_random_scaffold.file 2>&1 &
tail -f optuna_search/out_bbbp_32_random_scaffold.file

nohup python -u trainDistill_optuna.py --dataname bbbp --distill_hidden 64 --split_method random_scaffold --device cuda:0  > optuna_search/out_bbbp_64_random_scaffold.file 2>&1 &
tail -f optuna_search/out_bbbp_64_random_scaffold.file



nohup python -u trainDistill_optuna.py --dataname bbbp --distill_hidden 32 --split_method random --device cuda:0  > optuna_search/out_bbbp_32_random.file 2>&1 &
tail -f optuna_search/out_bbbp_32_random.file

nohup python -u trainDistill_optuna.py --dataname bbbp --distill_hidden 64 --split_method random --device cuda:0  > optuna_search/out_bbbp_64_random.file 2>&1 &
tail -f optuna_search/out_bbbp_64_random.file





nohup python -u trainDistill_optuna.py --dataname bace --distill_hidden 32 --split_method random_scaffold --device cuda:1  > optuna_search/out_bace_32_random_scaffold.file 2>&1 &
tail -f optuna_search/out_bace_32_random_scaffold.file

nohup python -u trainDistill_optuna.py --dataname bace --distill_hidden 64 --split_method random_scaffold --device cuda:1  > optuna_search/out_bace_64_random_scaffold.file 2>&1 &
tail -f optuna_search/out_bace_64_random_scaffold.file


nohup python -u trainDistill_optuna.py --dataname bace --distill_hidden 32 --split_method random --device cuda:1  > optuna_search/out_bace_32_random.file 2>&1 &
tail -f optuna_search/out_bace_32_random.file

nohup python -u trainDistill_optuna.py --dataname bace --distill_hidden 64 --split_method random --device cuda:1  > optuna_search/out_bace_64_random.file 2>&1 &
tail -f optuna_search/out_bace_64_random.file






nohup python -u trainDistill_optuna.py --dataname clintox --distill_hidden 32 --split_method random_scaffold --target_task 1 --device cuda:2  > optuna_search/out_clintox_32_random_scaffold.file 2>&1 &
tail -f optuna_search/out_clintox_32_random_scaffold.file

nohup python -u trainDistill_optuna.py --dataname clintox --distill_hidden 64 --split_method random_scaffold --target_task 1 --device cuda:2  > optuna_search/out_clintox_64_random_scaffold.file 2>&1 &
tail -f optuna_search/out_clintox_64_random_scaffold.file



nohup python -u trainDistill_optuna.py --dataname clintox --distill_hidden 32 --split_method random --target_task 1 --device cuda:2  > optuna_search/out_clintox_32_random.file 2>&1 &
tail -f optuna_search/out_clintox_32_random.file

nohup python -u trainDistill_optuna.py --dataname clintox --distill_hidden 64 --split_method random --target_task 1 --device cuda:2  > optuna_search/out_clintox_64_random.file 2>&1 &
tail -f optuna_search/out_clintox_64_random.file





WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=1 python -m trainLM lm.train.warmup_epochs 0.6 lm.train.epochs 5 lm.train.diagram False dataset.name clintox dataset.target_task 1 dataset.split_method random







python trainGNN.py dataset.name bbbp dataset.split_method random gnn.model.hidden_dim 32 device cuda:1