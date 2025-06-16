#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

# echo "Running with treeple_fast..."
# conda run -n treeple_fast python might_trainon_cohort1.py fast

echo "Running with treeple_standard..."
conda run -n treeple_standard python might_trainon_cohort1.py standard
