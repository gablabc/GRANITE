# !/bin/bash

# Remove the cached results of running script_granite.sh
find models -type f -name 'results.csv' | xargs -p rm
find models -type f -name 'decomposition*.joblib' | xargs -p rm
