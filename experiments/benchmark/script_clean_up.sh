# !/bin/bash

# Remove the cached results of running script_granite.sh
for data in "bike" "kin8nm"
do
    find "models/${data}" -type f -name 'results.csv' | xargs -p rm
    find "models/${data}" -type f -name 'decomposition*.joblib' | xargs -p rm
done
