# Copyright 2023 AICONSLab
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
    """
    Runs a basic SNF workflow from end to end.

    Args:
        All args correspond to those for make_affinity and SNF functions.

    Returns:
        matrices: A list containing three connectivity matrices: the SNF matrix,
            the "dense" network output from the robust_core_clustering function,
            and a "sparse" version of this matrix only keeping those values
            > (1 / C) **2
        labels: A list of assigned labels based on the three matrices above.
        sils: A list of average silhouette values corresponding to the three
            matrices above.
    """
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
