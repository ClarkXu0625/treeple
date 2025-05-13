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


def plot_traintime_heatmap(times1, times2, n_columns, n_rows,
                            title1="YDF Training Time", 
                            title2="Treeple Training Time", 
                            xlabel="Features", 
                            ylabel="Projections", 
                            scale="log"):
    """
    Plot the training time heatmap for YDF and Treeple.
    """

    # Ensure arrays
    times_ydf = np.array(times1)
    times_treeple = np.array(times2)

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
    axes[0].set_title(title1)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_xticks(np.arange(len(n_columns)))
    axes[0].set_yticks(np.arange(len(n_rows)))
    axes[0].set_xticklabels(n_columns)
    axes[0].set_yticklabels(n_rows)

    # Plot Treeple heatmap
    cax2 = axes[1].imshow(times_treeple, cmap='RdYlGn_r', aspect='equal', norm=log_norm)
    axes[1].set_title(title2)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
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




def plot_single_heatmap(
    time_data, 
    n_columns, 
    n_rows, 
    title="Training Time Heatmap", 
    xlabel="Number of Features", 
    ylabel="Number of Projections", 
    scale="log"
):
    """
    Plot a single heatmap for training time.

    Parameters:
        time_data (2D array): Training time data to plot.
        n_columns (list): Column labels (x-axis).
        n_rows (list): Row labels (y-axis).
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        scale (str): 'log' for log scale, 'linear' for linear scale.
    """
    time_data = np.array(time_data)
    if scale == "log":
        time_data = np.maximum(time_data, 1e-3)
        norm = LogNorm(vmin=time_data.min(), vmax=time_data.max())
    else:
        norm = None

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    cax = ax.imshow(time_data, cmap='RdYlGn_r', aspect='equal', norm=norm)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(n_columns)))
    ax.set_yticks(np.arange(len(n_rows)))
    ax.set_xticklabels(n_columns)
    ax.set_yticklabels(n_rows)

    # Annotate heatmap
    for i in range(len(n_rows)):
        for j in range(len(n_columns)):
            val = time_data[i, j]
            color_val = norm(val) if norm else val / np.max(time_data)
            text_color = 'black' if color_val > 0.3 else 'white'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_color)

    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.045, pad=0.04)
    cbar.set_label('Training Time (seconds)')

    plt.show()