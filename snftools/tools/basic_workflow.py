import networkx as nx
import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn.metrics import silhouette_score
from snf import make_affinity

from snftools.snf import robust_core_clustering_matrix, SNF
from snftools.utils import plot_connectivity, sim2dist, visualize_graph


def snf_analysis(
    data,
    C,
    K=None,
    mu=0.5,
    t=20,
    metric="sqeuclidean",
    normalize=True,
    seed=3333,
):
    if K is None:
        K = int(data[0].shape[0] / C)
    aff_matrices = make_affinity(data, metric=metric, K=K, mu=mu, normalize=normalize)
    if len(data) == 0:
        t = 0
    snf = SNF(aff_matrices, K=K, t=t)
    dense, sparse = robust_core_clustering_matrix(aff_matrices, C, K=K, t=t)
    np.fill_diagonal(dense, 1)
    np.fill_diagonal(sparse, 1)
    labels = [
        spectral_clustering(net, n_clusters=C, random_state=C)
        for net in [snf, dense, sparse]
    ]
    sils = [
        silhouette_score(sim2dist(x), label_set)
        for x, label_set in zip([snf, dense, sparse], labels)
    ]
    matrix_labels = ["snf", "dense", "sparse"]
    for i, sil in enumerate(sils):
        print(f"Silhouette score for {matrix_labels[i]}: {sil:.5f}")
    for i, matrix in enumerate([snf, dense, sparse]):
        print(f"{matrix_labels[i]}:")
        plot_connectivity(matrix, sort_labels=labels[i])
        G = nx.from_numpy_array(matrix)
        visualize_graph(G, color=labels[i])
    return [snf, dense, sparse], labels, sils


# from scipy.spatial.distance import cdist
# from scipy.stats import zscore
# import snf

# from snftools.snf import SNF


# data
# metric
# mu
# K
# t

# affinity_networks = snf.make_affinity(data, metric=metric, K=K, mu=mu)
# fused_network = SNF(affinity_networks, K=K, t=t)


# data = load_data()  # TODO create load_data function based on config

# def compute_snf(tables, K=20, t=20, alpha=0.5, metric="sqeuclidean"):
#     aff_matrices = []
#     for table in tables:
#         table = zscore(table, ddof=1)
#         dist = cdist(table, table, metric=metric)


# for table in data:
#     table = zscore(table, ddof=1)
#     table =
