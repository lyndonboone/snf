from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
import seaborn as sns


def contingency(var, group, plot=True):
    """
    Compute a chi-squared statistic and p-value for var against group.

    See https://medium.com/swlh/how-to-run-chi-square-test-in-python-4e9f5d10249d

    Args:
        var: First variable
        group: Second variable (will be placed on x-axis)
        plot: If True, show a plot of the contingency table
    """
    table = pd.crosstab(var, group)
    x, p, dof, e = chi2_contingency(table)
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(3.5,) * 2, dpi=100, facecolor="white")
        sns.heatmap(table, annot=True, cmap=sns.color_palette("Blues"), ax=ax)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        ax.text(
            0.05,
            0.05,
            f"$\chi_2$: {x:.2f}; $p$: {p:.3f}; DOF: {dof}",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
            bbox=bbox_props,
        )
        # plt.show()
    return x, p, dof, e


def dominateset(xx, KK=20):
    """
    Find the K nearest neighbours for each row in a similarity matrix.

    This function is meant to be a direct translation of the .dominateset
    function in the SNFtool package from R to Python.

    Args:
        xx: Input similarity matrix.
        KK: Number of K-nearest-neighbours (default 20).
    """

    def zero(x):
        s = np.argsort(x)  # get indices that would sort the row
        x[s[: (len(x) - KK)]] = 0  # zero the array at non-NN locations
        return x

    def normalize(X):
        return X / np.sum(X, axis=1, keepdims=True)

    A = np.zeros(xx.shape)
    for i in range(len(xx)):  # iterate over rows
        A[i] = zero(deepcopy(xx[i]))
    return normalize(A)


def make_pies(group_array, split_array, colors=None):
    for group in np.unique(group_array):
        subset = split_array[group_array == group]
        unique = np.unique(subset, return_counts=True)
        fig, ax = plt.subplots(facecolor="white")
        ax.pie(
            unique[1],
            labels=unique[0],
            autopct="%1.1f%%",
            colors=[colors[i] for i in unique[0]],
            wedgeprops={"linewidth": 1, "edgecolor": "black"},
            textprops={"fontsize": 14},
        )
        ax.set_title(group, fontsize=14)
        plt.show()


def oneway(
    group_list,
    df,
    col,
    plot=True,
    palette=None,
    xlabel="Group or cluster",
    ylabel=None,
    size=4,
    dotcolor="dodgerblue",
):
    """
    Compute a oneway ANOVA for col of df across the groups defined by group_list.

    Args:
        group_list: List or numpy array of group/cluster assignments corresponding to each row of df.
        df: Dataframe where each row is a patient/case.
        col: Column of df to treat as the dependent variable.
        plot: If True, show a plot of the relationship.
        palette: Color palette for the different groups.
        xlabel: Label for the x-axis if plot is True.
        ylabel: Label for the y-axis if plot is True.
        size: Size of swarmplot points.
        dotcolor: Color of swarmplot points.
    """
    # get list of numpy arrays for each unique value in group_list
    _ = [df[col].values[group_list == x] for x in np.unique(group_list)]
    # remove NaNs
    _ = [array[~np.isnan(array)] for array in _]
    F, p = f_oneway(*_)
    n = sum([len(array) for array in _])
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(3.5,) * 2, dpi=100, facecolor="white")
        sns.violinplot(
            x=group_list,
            y=df[col].values,
            palette=palette,
            ax=ax,
            inner="box",
        )
        sns.swarmplot(
            x=group_list,
            y=df[col].values,
            size=size,
            c=dotcolor,
        )
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(col)
        ax.set_xlabel(xlabel)
        ax.set_ylim(top=1.3 * np.nanmax(df[col].values))
        subscript = f"{len(_) - 1}, {n - 1}"
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        ax.text(
            0.05,
            0.95,
            f"$F_{{{subscript}}}$: {F:.3f}\n$p$: {p:.3f}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox=bbox_props,
        )
        # plt.show()
    return F, p, n


def plot_connectivity(array, cmap="Blues", vmax=None, sort_labels=None):
    fig, ax = plt.subplots(1, 1, figsize=(3.5,) * 2, dpi=100, facecolor="white")
    if sort_labels is not None:
        sort_indices = sort_labels.argsort()
        sorted_array = array[:, sort_indices]
        sorted_array = sorted_array[sort_indices]
        ax.imshow(sorted_array, cmap=cmap, vmax=vmax)
    else:
        ax.imshow(array, cmap=cmap, vmax=vmax)
    # Major ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # Minor ticks
    ax.set_xticks(np.arange(-0.5, array.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, array.shape[0], 1), minor=True)
    # Gridlines based on minor ticks
    ax.grid(which="minor", color="k", linestyle="-", linewidth=1)
    plt.show()


def silhouette_plot(
    silhouette_values,
    cluster_assignments,
    step=1,
    figsize=3.5,
    dpi=200,
    facecolor="white",
    # color_palette=None,
    color_palette=sns.color_palette(),
    edgecolor="same",
    alpha=0.7,
    fontsize=7,
    textx=-0.05,
    title="Silhouette plot",
    xlabel="Silhouette coefficient",
    ylabel="Cluster label",
    xlim=[-0.2, 1],
    xticks=[-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    vline_color="k",
    vline_style="--",
    vline_width=1,
):
    """
    Adapted from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    """
    y_lower = step
    fig, ax = plt.subplots(1, 1, figsize=(figsize,) * 2, dpi=dpi, facecolor=facecolor)
    for i, c in enumerate(np.unique(cluster_assignments)):
        ith_cluster_silhouette_values = silhouette_values[cluster_assignments == c]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        plot_face_color = color_palette[i]
        if edgecolor == "same":
            ecolor = color_palette[i]
        else:
            ecolor = edgecolor
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=plot_face_color,
            edgecolor=ecolor,
            alpha=alpha,
        )
        ax.text(textx, y_lower + 0.5 * size_cluster_i, str(c), fontsize=fontsize)
        y_lower = y_upper + step
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlim(*xlim)
        ax.set_yticks([])
        ax.set_xticks(xticks)
        ax.tick_params(labelsize=fontsize)
        silhouette_avg = silhouette_values.mean()
        ax.axvline(
            x=silhouette_avg, color=vline_color, linestyle=vline_style, lw=vline_width
        )
        ax.axvline(x=0, color="k", linestyle="-", lw=0.4)
    plt.show()


def sim2dist(similarity_matrix, method=1):
    """
    Converts a similarity matrix to distance matrix, preserving distribution properties.

    See: https://stats.stackexchange.com/questions/12922/will-the-silhouette-formula-change-depending-on-the-distance-metric

    Args:
        similarity_matrix: Similarity matrix.
        method: If 1, uses the formula d = max(s) - s, where d is distance,
            s is similarity, and max(s) is the max similarity in the matrix.
            Note that the diagonal elements of s are zeroed before this computation.
            If 2, the matrix s is normalized to [0, 1] after zeroing the diagonal
            and before doing the subtraction. Diagonal elements of the returned
            array are set to zero.
    """
    _similarity_matrix = deepcopy(similarity_matrix)
    np.fill_diagonal(_similarity_matrix, 0)
    if method == 1:
        dist = _similarity_matrix.max() - _similarity_matrix
        np.fill_diagonal(dist, 0)
        return dist
    else:
        _similarity_matrix /= _similarity_matrix.max()
        dist = 1 - _similarity_matrix
        np.fill_diagonal(dist, 0)
        return dist


def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(
        G,
        pos=nx.spring_layout(G, seed=42),
        with_labels=False,
        node_color=color,
        cmap="Set2",
    )
    plt.show()
