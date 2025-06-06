#!/bin/bash

# Ensure execution from the project root
#cd "$(clark/experiment "$0")"

echo "Running with neofit_standard environment..."
conda run -n neofit_standard python shuffle_benchmark.py standard

echo "Running with treeple_neofit environment..."
conda run -n treeple_neofit python shuffle_benchmark.py floyd

echo "Running with floyd_inline environment..."
conda run -n floyd_inline python shuffle_benchmark.py inline
