import os
import time
import numpy as np
import pandas as pd
import gc
import cProfile
import pstats

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from treeple.datasets import make_trunk_classification
from neofit import NeuroExplainableOptimalFIT
from sklearn.inspection import permutation_importance
import shap
import lime

def test_profit(X_train, y_train, sample, dim):

    profiler = cProfile.Profile()
    profiler.enable()

    start = time.time()
    
    profit = NeuroExplainableOptimalFIT(
        n_estimators=5000,
        n_permutations=100000,
        clf_type="SPORF",
        alpha=0.05,
        verbose=False
    )
    p_values, imp_features, _ = profit.get_significant_features(X_train, y_train)

    profiler.disable()

    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    stats.print_stats(30)


    os.makedirs("./sex_classification3/results", exist_ok=True)
    np.save(f"./sex_classification3/results/p_values_{sample}_{dim}.npy", p_values)
    duration = time.time() - start
    del profit, p_values, imp_features
    gc.collect()
    return duration

def test_shap(X_train, X_val, y_train):
    start = time.time()
    rf = RandomForestClassifier(n_estimators=5000, random_state=0)
    rf.fit(X_train, y_train)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_val)
    duration = time.time() - start
    del rf, explainer, shap_values
    gc.collect()
    return duration

def test_lime(X_train, X_val, y_train, sample, dim):
    start = time.time()
    rf = RandomForestClassifier(n_estimators=5000, random_state=0)
    rf.fit(X_train, y_train)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        mode='classification',
        feature_names=list(range(X_train.shape[1])),
        verbose=False
    )
    lime_exp = explainer.explain_instance(
        X_val[0].reshape(1, -1).flatten(),
        rf.predict_proba,
        num_features=X_train.shape[1]
    )
    lime_values = lime_exp.as_list()
    importance_scores = np.array([value for _, value in lime_values])
    os.makedirs("./sex_classification3/results_LIME_testing", exist_ok=True)
    np.save(f"./sex_classification3/results_LIME_testing/lime_values_{sample}_{dim}.npy", importance_scores)
    duration = time.time() - start
    del rf, explainer, lime_exp, lime_values, importance_scores
    gc.collect()
    return duration

def time_test(dim_range, sample_range):
    time_list_neofit = []
    time_list_shap = []
    time_list_lime = []

    for dim in dim_range:
        for sample in sample_range:
            print(f"\n--- Testing dim={dim}, sample={sample} ---")

            # Data generation
            X, y = make_trunk_classification(
                n_samples=sample,
                n_dim=dim,
                n_informative=min(dim, 600),
                seed=0
            )
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

            # PROFIT
            print("Running PROFIT...")
            t_profit = test_profit(X_train, y_train, sample, dim)
            print(f"Time PROFIT: {t_profit:.2f}s")
            time_list_neofit.append(t_profit)

            # SHAP
            print("Running SHAP...")
            t_shap = test_shap(X_train, X_val, y_train)
            print(f"Time SHAP: {t_shap:.2f}s")
            time_list_shap.append(t_shap)

            # LIME
            print("Running LIME...")
            t_lime = test_lime(X_train, X_val, y_train, sample, dim)
            print(f"Time LIME: {t_lime:.2f}s")
            time_list_lime.append(t_lime)

            # Free all main variables
            del X, y, X_train, X_temp, y_train, y_temp, X_val, X_test, y_val, y_test
            gc.collect()

            with open("time_log2.txt", "a") as log_file:
                log_file.write(f"dim={dim}, sample={sample}\n")
                log_file.write(f"PROFIT: {t_profit:.2f} sec\n")
                log_file.write(f"SHAP: {t_shap:.2f} sec\n")
                log_file.write(f"LIME: {t_lime:.2f} sec\n\n")

    return time_list_neofit, time_list_shap, time_list_lime


# === Run Testing ===
dim_vals = np.array([256])
sample_vals = dim_vals  # Assuming square test grid
index = pd.MultiIndex.from_product([dim_vals, sample_vals], names=["dim", "sample"])

time_list_neofit, time_list_shap, time_list_lime = time_test(dim_vals, sample_vals)

# === Save Results ===
df_profit = pd.DataFrame({"time": time_list_neofit}, index=index).unstack(level=-1)
df_shap = pd.DataFrame({"time": time_list_shap}, index=index).unstack(level=-1)
df_lime = pd.DataFrame({"time": time_list_lime}, index=index).unstack(level=-1)

os.makedirs("./sex_classification3/results_permutation_testing", exist_ok=True)
df_profit.to_csv("./sex_classification3/results_permutation_testing/time_list_neofit.csv")
df_shap.to_csv("./sex_classification3/results_permutation_testing/time_list_shap.csv")
df_lime.to_csv("./sex_classification3/results_permutation_testing/time_list_lime.csv")
