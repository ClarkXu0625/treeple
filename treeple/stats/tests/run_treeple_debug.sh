#!/bin/bash

source /home/clark/anaconda3/etc/profile.d/conda.sh
conda activate treeple

echo "=== Starting treeple profiling ==="
which python

# Force a sleep BEFORE the real script to let VTune attach
echo "Sleeping 5s before starting workload..."
sleep 5

# Run the actual script
/home/clark/anaconda3/envs/treeple/bin/python \
  /home/clark/Documents/GitHub/treeple/treeple/stats/tests/test_sporf.py

# Force a sleep AFTER the real script to extend collection window
echo "Sleeping 5s after workload to keep process alive for VTune..."
sleep 5
