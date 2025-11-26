# !/bin/bash

seed=${1:-0}
dataset=${2:-bike}
iteration=0
models=(gbt rf mlp)
n_models=${#models[@]}
train_background_size=(1000 500 1000)
test_background_size=(2000 1000 2000)
# models=(mlp)
# n_models=${#models[@]}
# train_background_size=(1000)
# test_background_size=(2000)
for model in "${models[@]}"
do
    for loss_fn in "PDP_vs_ICE" "PDP_vs_Mplot" "PureRisk_vs_FullRisk" "PFI_vs_CFI"
    do
        for max_depth in {0..3}
        do
            train_background_size_current=${train_background_size[iteration%n_models]}
            test_background_size_current=${test_background_size[iteration%n_models]}
            python3 2_train_granite.py --model_name=$model --dataset=$dataset --random_state=$seed \
                --loss_fn=$loss_fn --alpha=0.01 --max_depth=$max_depth\
                --train_background_size=$train_background_size_current\
                --test_background_size=$test_background_size_current
        done
    done
    iteration=$((iteration+1))
done
