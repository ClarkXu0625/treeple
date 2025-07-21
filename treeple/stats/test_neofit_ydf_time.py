from treeple.datasets import make_trunk_classification
from sklearn.model_selection import train_test_split
from neofit import NeuroExplainableOptimalFIT as TreepleNEOFIT
from neofit_ydf import NeuroExplainableOptimalFIT_ydf as YDFNEOFIT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os


def benchmark_neofit_versions(dim_range, sample_range):
    records = []

    for dim in dim_range:
        for sample in sample_range:
            X, y = make_trunk_classification(n_samples=sample, n_dim=dim, n_informative=min(dim, 600), seed=0)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

            # Time Treeple NEOFIT
            treeple_model = TreepleNEOFIT(n_estimators=5000, n_permutations=100000, clf_type="SPORF", alpha=0.05, verbose=False)
            start_time = time.time()
            p_values_treeple, imp_features_treeple, _ = treeple_model.get_significant_features(X_train, y_train)
            time_treeple = time.time() - start_time

            # Time YDF NEOFIT
            ydf_model = YDFNEOFIT(n_estimators=5000, n_permutations=100000, clf_type="SPORF", alpha=0.05, verbose=False)
            start_time = time.time()
            p_values_ydf, imp_features_ydf, _ = ydf_model.get_significant_features(X_train, y_train)
            time_ydf = time.time() - start_time

            records.append({
                "dim": dim,
                "sample": sample,
                "version": "Treeple",
                "time": time_treeple
            })
            records.append({
                "dim": dim,
                "sample": sample,
                "version": "YDF",
                "time": time_ydf
            })

    return pd.DataFrame(records)


# Define input settings
dim = np.array([128, 256, 512])
sample = dim  # sample size equals dimension

# Run the benchmark
results_df = benchmark_neofit_versions(dim, sample)
print(results_df)

# Save as .txt (tab-separated)
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/neofit_runtime_comparison.txt", sep='\t', index=False)


# Plotting results
import seaborn as sns

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=results_df, x="dim", y="time", hue="version", marker="o")
plt.title("Runtime Comparison of NEOFIT (Treeple vs YDF)")
plt.xlabel("Data Dimension")
plt.ylabel("Runtime (seconds)")
plt.tight_layout()
plt.show()
