import treeple
from treeple import ObliqueRandomForestClassifier
from sklearn.model_selection import train_test_split
from treeple.datasets import make_trunk_classification
import cProfile
import pstats
import io
###
# Shared hyperparameters that used for both models
MAX_DEPTH = 10
N_ESTIMATORS = 10
RANDOM_SEED = 42
N_JOBS=-1
BOOTSTRAP = True
MAX_FEATURE = 3000
FEATURE_COMBINATIONS = 1000.0

params_treeple = {}
params_treeple["n_estimators"] = int(N_ESTIMATORS)
params_treeple["criterion"] = "entropy"
params_treeple["max_depth"] = MAX_DEPTH
params_treeple["min_samples_split"] = 2
params_treeple["min_samples_leaf"] = 1
params_treeple["min_weight_fraction_leaf"] = 0.0
params_treeple["max_features"] = MAX_FEATURE
params_treeple["max_leaf_nodes"] = 30
params_treeple["min_impurity_decrease"] = 0.0
params_treeple["bootstrap"] = BOOTSTRAP
params_treeple["oob_score"] = False
params_treeple["n_jobs"] = N_JOBS
params_treeple["random_state"] = RANDOM_SEED
params_treeple["verbose"] = 0
params_treeple["warm_start"] = False
params_treeple["class_weight"] = None
params_treeple["max_samples"] = None
params_treeple["feature_combinations"] = FEATURE_COMBINATIONS


def profiling_fit(n_estimators, n_dim, n_samples, max_features, feature_combinations, max_depth, n_jobs, max_leaf_nodes, treeple_params=None):
    X, y = make_trunk_classification(n_samples=n_samples, n_dim=n_dim)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if treeple_params is None:
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
            "feature_combinations": feature_combinations,
            "n_jobs": n_jobs,
            "max_leaf_nodes": max_leaf_nodes
        }
    else:
        params = treeple_params.copy()
        params.update({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
            "feature_combinations": feature_combinations,
            "n_jobs": n_jobs,
            "max_leaf_nodes": max_leaf_nodes,
        })

    model = ObliqueRandomForestClassifier(**params)

    
    profiler = cProfile.Profile()
    profiler.enable()

    model.fit(X_train, y_train)

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    ps.print_stats()

    return s.getvalue()

import sys
import time
# print("=== STARTING TREEPLE TEST ===", file=sys.stderr)
# print("Python executable:", sys.executable, file=sys.stderr)

# # Optional: force a long run
# print("Sleeping 1s...", file=sys.stderr)
# time.sleep(5)
# start_time = time.time()


if __name__=="__main__":
    print("=== RUNNING __main__ ===", file=sys.stderr)
    #time.sleep(1)
    report_njobs1 = profiling_fit(
        n_estimators=100,
        n_dim=2048,
        n_samples=1600,
        max_features=2048,
        feature_combinations=2.0,
        max_depth=10,
        n_jobs=-1,
        max_leaf_nodes=30,
        treeple_params=params_treeple
    )

    print(report_njobs1[0:2000])

    time_taken = time.time()-start_time
    print(f"time taken to train: {time_taken}")