
for seed in 0 1 2 3 4
do
    nohup python -u trainDistill_optuna.py \
            --dataname clintox \
            --distill_hidden 64 \
            --split_method random_scaffold \
            --device cuda:1 \
            --seed $seed \
            > optuna_search/clintox/random_scaffold_64_${seed}.file 2>&1

done

