

for seed in 0 1 2 3 4
do
    nohup python -u trainDistill_optuna.py \
            --dataname bace \
            --distill_hidden 32 \
            --split_method random_scaffold \
            --device cuda:0 \
            --seed $seed \
            > optuna_search/bace/random_scaffold_32_${seed}.file 2>&1

done

