#!/bin/bash

# Setup environment activation
source ~/miniconda3/etc/profile.d/conda.sh

# Define Python script path
PY_SCRIPT="might_trainon_cohort1.py"
SCRIPT_DIR=$(dirname "$PY_SCRIPT")
SCRIPT_BASE=$(basename "$PY_SCRIPT" .py)

# Run in treeple_fast
echo "=== Running in treeple_fast ==="
conda activate treeple_fast
python "$PY_SCRIPT" > "${SCRIPT_DIR}/${SCRIPT_BASE}_fast.txt" 2>&1
conda deactivate

# Run in treeple_standard
echo "=== Running in treeple_standard ==="
conda activate treeple_standard
python "$PY_SCRIPT" > "${SCRIPT_DIR}/${SCRIPT_BASE}_standard.txt" 2>&1
conda deactivate
