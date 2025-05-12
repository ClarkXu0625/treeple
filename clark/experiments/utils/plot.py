import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from treeple.datasets import make_trunk_classification
import ydf
import matplotlib.pyplot as plt
from treeple import ObliqueRandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from treeple._lib.sklearn.tree._criterion import Gini
from treeple.tree._oblique_splitter import BestObliqueSplitterTester
from treeple.datasets import make_trunk_classification
import pandas as pd
import math
from matplotlib.colors import LogNorm    

def plot_traintime_heatmap(times_ydf1, times_treeple1, n_columns, n_rows):
    """
    Plot the training time heatmap for YDF and Treeple.
    """

    # Ensure arrays
    times_ydf = np.array(times_ydf1)
    times_treeple = np.array(times_treeple1)

    # Replace zeros for log scale
    times_ydf = np.maximum(times_ydf, 1e-3)
    times_treeple = np.maximum(times_treeple, 1e-3)

    # Define shared color normalization
    log_norm = LogNorm(vmin=min(times_ydf.min(), times_treeple.min()),
                       vmax=max(times_ydf.max(), times_treeple.max()))

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # Plot YDF heatmap
    cax1 = axes[0].imshow(times_ydf, cmap='RdYlGn_r', aspect='equal', norm=log_norm)
    axes[0].set_title('Training Time Heatmap (YDF)')
    axes[0].set_xlabel('Number of Features')
    axes[0].set_ylabel('Number of Projections')
    axes[0].set_xticks(np.arange(len(n_columns)))
    axes[0].set_yticks(np.arange(len(n_rows)))
    axes[0].set_xticklabels(n_columns)
    axes[0].set_yticklabels(n_rows)

    # Plot Treeple heatmap
    cax2 = axes[1].imshow(times_treeple, cmap='RdYlGn_r', aspect='equal', norm=log_norm)
    axes[1].set_title('Training Time Heatmap (Treeple)')
    axes[1].set_xlabel('Number of Features')
    axes[1].set_ylabel('Number of Projections')
    axes[1].set_xticks(np.arange(len(n_columns)))
    axes[1].set_yticks(np.arange(len(n_rows)))
    axes[1].set_xticklabels(n_columns)
    axes[1].set_yticklabels(n_rows)

    # Create one shared colorbar
    cbar = fig.colorbar(cax2, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Training Time (seconds)')

    # Annotate YDF heatmap
    for i in range(len(n_rows)):
        for j in range(len(n_columns)):
            text_color = 'black' if log_norm(times_ydf[i, j]) > 0.3 else 'white'
            axes[0].text(j, i, f"{times_ydf[i, j]:.2f}", ha='center', va='center', color=text_color)

    # Annotate Treeple heatmap
    for i in range(len(n_rows)):
        for j in range(len(n_columns)):
            text_color = 'black' if log_norm(times_treeple[i, j]) > 0.3 else 'white'
            axes[1].text(j, i, f"{times_treeple[i, j]:.2f}", ha='center', va='center', color=text_color)

    plt.show()