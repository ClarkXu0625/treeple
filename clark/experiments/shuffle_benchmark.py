# run_simulation.py
import sys
import os
import numpy as np
import pandas as pd
from treeple.datasets import make_trunk_classification
from utils.simulations import constant_nNonzeros_simulation_treeple
from utils.plot import plot_traintime_heatmap, plot_single_heatmap

# Accept suffix from CLI
suffix = sys.argv[1] if len(sys.argv) > 1 else "default"

MAX_DEPTH = 10
N_ESTIMATORS = 100
RANDOM_SEED = 42
N_JOBS = 8
BOOTSTRAP = True
MAX_FEATURE = 3000
FEATURE_COMBINATIONS = 1000.0

params_treeple = {
    "n_estimators": int(N_ESTIMATORS),
    "criterion": "entropy",
    "max_depth": MAX_DEPTH,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": MAX_FEATURE,
    "max_leaf_nodes": 30,
    "min_impurity_decrease": 0.0,
    "bootstrap": BOOTSTRAP,
    "oob_score": False,
    "n_jobs": N_JOBS,
    "random_state": RANDOM_SEED,
    "verbose": 0,
    "warm_start": False,
    "class_weight": None,
    "max_samples": None,
    "feature_combinations": FEATURE_COMBINATIONS,
}

n_rows = np.array([64, 128, 256, 512, 1024])
n_columns = np.array([64, 128, 256, 512, 1024])
target_non_zeros_per_row = float(2.0)

accs_treeple, times_treeple = constant_nNonzeros_simulation_treeple(
    100, params_treeple, target_non_zeros_per_row, n_rows, n_columns,
    n_samples=2000, n_rep=5, plot=False
)

times_treeple = np.transpose(times_treeple)
accs_treeple = np.transpose(accs_treeple)
os.makedirs("result/shuffle5", exist_ok=True)
pd.DataFrame(times_treeple, index=n_rows, columns=n_columns).to_csv(
    f"result/shuffle5/times_treeple_{suffix}.txt", sep="\t"
)
pd.DataFrame(accs_treeple, index=n_rows, columns=n_columns).to_csv(
    f"result/shuffle5/accs_treeple_{suffix}.txt", sep="\t"
)
