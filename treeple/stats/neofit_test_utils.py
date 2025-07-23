from treeple.datasets import make_trunk_classification
from sklearn.model_selection import train_test_split
from neofit import NeuroExplainableOptimalFIT
from neofit_ydf import NeuroExplainableOptimalFIT_ydf
#from treeple.stats import NeuroExplainableOptimalFIT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
import pdb
import seaborn as sns
import matplotlib.pyplot as plt



def test_neofit(dim_range, 
              sample_range, 
              clf_type="SPORF", 
              n_permutations=10000,
              n_estimators=5000,
              device1='cuda',
              save_result=True,
              result_path="results/neofit_device_compare.csv"):
    '''Simple test on neofit functionality'''
    
    records = []
    for dim in dim_range:
        for sample in sample_range:
            X, y = make_trunk_classification(n_samples=sample, n_dim=dim, n_informative=min(dim, 600), seed=1)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=2)

            # Run on device1
            model1 = NeuroExplainableOptimalFIT(n_estimators=n_estimators,
                                                n_permutations=n_permutations,
                                                clf_type=clf_type,
                                                alpha=0.05,
                                                verbose=False,
                                                device=device1)
            start1 = time.time()
            pval1, imp1, _ = model1.get_significant_features(X_train, y_train)
            end1 = time.time()

            records.append({
                "dim": dim,
                "sample": sample,
                "time": end1 - start1,
                "type": f"{clf_type}_{device1}"
            })

    results_df = pd.DataFrame.from_records(records)

    if save_result:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        results_df.to_csv(result_path, index=False)

    return results_df
    




def time_test(dim_range, 
              sample_range, 
              clf_type="SPORF", 
              n_permutations=10000,
              n_estimators=5000,
              device1='cuda',
              device2='cpu',
              save_result=True,
              result_path="results/neofit_device_compare.csv"):
    records = []

    for dim in dim_range:
        for sample in sample_range:
            X, y = make_trunk_classification(n_samples=sample, n_dim=dim, n_informative=min(dim, 600), seed=1)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=2)

            # Run on device1
            model1 = NeuroExplainableOptimalFIT(n_estimators=n_estimators,
                                                n_permutations=n_permutations,
                                                clf_type=clf_type,
                                                alpha=0.05,
                                                verbose=False,
                                                device=device1)
            start1 = time.time()
            pval1, imp1, _ = model1.get_significant_features(X_train, y_train)
            end1 = time.time()

            records.append({
                "dim": dim,
                "sample": sample,
                "time": end1 - start1,
                "type": f"{clf_type}_{device1}"
            })

            # Run on device2
            model2 = NeuroExplainableOptimalFIT(n_estimators=n_estimators,
                                                n_permutations=n_permutations,
                                                clf_type=clf_type,
                                                alpha=0.05,
                                                verbose=False,
                                                device=device2)
            start2 = time.time()
            pval2, imp2, _ = model2.get_significant_features(X_train, y_train)
            end2 = time.time()

            records.append({
                "dim": dim,
                "sample": sample,
                "time": end2 - start2,
                "type": f"{clf_type}_{device2}"
            })

            # Print mismatch if any
            diff = np.sum(imp2.astype(int) - imp1.astype(int))
            print(f"Feature diff (imp2 - imp1) for dim={dim}, sample={sample}: {diff}")

    results_df = pd.DataFrame.from_records(records)

    if save_result:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        results_df.to_csv(result_path, index=False)

    return results_df



def plot_runtime_heatmaps(
    df: pd.DataFrame,
    save_path: str = None,
    titles: list = None,
    cmap: str = "YlGnBu",
    figsize: tuple = None,
    font_scale: float = 1.4,
    fmt: str = ".1f"
):
    """
    Plots runtime heatmaps from a DataFrame grouped by 'type'.

    Parameters:
    - df : pd.DataFrame
        Must contain columns: 'dim', 'sample', 'time', 'type'
    - save_path : str or None
        If provided, saves the plot to this file path (e.g., 'plot.png')
    - titles : list or dict or None
        Custom subplot titles. If list, matched by order of types.
        If dict, matched by type name. If None, uses 'type' values.
    - cmap : str
        Seaborn colormap (default: 'YlGnBu')
    - figsize : tuple or None
        Optional matplotlib figsize. Default scales with number of types.
    - font_scale : float
        Font scale for seaborn (default: 1.4)
    - fmt : str
        Format for heatmap annotations (default: '.1f')
    """
    if not all(col in df.columns for col in ["dim", "sample", "time", "type"]):
        raise ValueError("DataFrame must contain 'dim', 'sample', 'time', and 'type' columns")

    unique_types = df['type'].unique()
    n_types = len(unique_types)

    # Compute global color scale
    vmin = df['time'].min()
    vmax = df['time'].max()

    # Determine titles
    if titles is None:
        titles = list(unique_types)
    elif isinstance(titles, dict):
        titles = [titles.get(t, t) for t in unique_types]

    if figsize is None:
        figsize = (6 * n_types, 6)

    # Plotting
    plt.figure(figsize=figsize)
    sns.set(font_scale=font_scale)

    for i, t in enumerate(unique_types, 1):
        data_pivot = df[df['type'] == t].pivot(index='sample', columns='dim', values='time')
        plt.subplot(1, n_types, i)
        sns.heatmap(
            data_pivot,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': 'Time (s)'}
        )
        plt.title(titles[i - 1], fontsize=18)
        plt.xlabel("Dimension", fontsize=14)
        plt.ylabel("Sample Size", fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to: {save_path}")

    plt.show()
