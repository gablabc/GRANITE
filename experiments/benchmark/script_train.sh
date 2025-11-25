# !/bin/bash

# Models on Bikesharing
n_rep=(80 20 20)
iteration=0
models=(gbt rf mlp)
n_models=${#models[@]}
for seed in {0..4}
do
    for model in gbt rf mlp
    do
        if [ -d "models/bike/${model}_${seed}/" ]; then
            echo "Skip since already computed"
        else
            n_rep_current=${n_rep[iteration%n_models]}
            python3 1_train_model.py --model_name=$model --dataset=bike --random_state=$seed \
                --n_repetitions=$n_rep_current --n_jobs=4
        fi
        iteration=$((iteration+1))
    done
done

