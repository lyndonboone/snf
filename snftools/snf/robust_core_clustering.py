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


import numpy as np
from sklearn.cluster import spectral_clustering

from snftools.snf import SNF


def add_cooccurrences_to_matrix(matrix, cooccurrences):
    """
    Add one to each element (i,j) of a square matrix if cooccurrences contains both
    i and j.

    Args:
        matrix: A square 2D numpy array.
        cooccurrences: A 1D numpy array containing unique, increasing integers.
            The largest integer must not be greater than len(matrix).
    """

    def is_sorted(l):
        return all(l[i] <= l[i + 1] for i in range(len(l) - 1))

    assert matrix.ndim == 2, "matrix must be a 2D array"
    assert matrix.shape[0] == matrix.shape[1], "matrix must be square"
    assert cooccurrences.ndim == 1, "cooccurrences must be a 1D array"
    assert is_sorted(cooccurrences), "cooccurrences list must be sorted"
    assert cooccurrences[-1] <= len(
        matrix
    ), "the last element of cooccurrences must be less than len(matrix)"
    for i in range(len(cooccurrences)):
        for j in range(i + 1, len(cooccurrences)):
            a = cooccurrences[i]
            b = cooccurrences[j]
            matrix[a, b] += 1
            matrix[b, a] += 1


def robust_core_clustering_matrix(
    affinity_mat_list,
    C,
    K=20,
    t=20,
    num_samples=1000,
    percent_sample=0.8,
    seed=3333,
    verbose=True,
    **kwargs,
):
    """
    This function is a direct Python adaptation of the RobustCoreClusteringMatrix
    function implemented in R by Jacobs et al. (2020), Neuropharmacology.

    Args:
        affinity_mat_list: List of affinity matrices (same as input to SNF).
        C: Number of clusters.
        K: Number of K-nearest-neighbours for SNF (default 20).
        t: Number of fusion iterations for SNF (default 20).
        num_samples: Expected number of times you would like to sample each
            individual. Default is 1000.
        percent_sample: The percentage of individuals you'd like to sample each
            time. Default is 0.8.
        clust_type: Spectral clustering type. Default is 2.
        seed: RNG seed. Default is 3333.
        **kwargs: Keyword arguments to sklearn.cluster.spectral_clustering.
    """
    # determine number of samples to collect
    # start sampling
    n = len(affinity_mat_list[0])
    # init matrices to hold sample and cluster cooccurrences
    sampled_together = np.zeros((n,) * 2)
    clustered_together = np.zeros((n,) * 2)
    indices = list(range(n))
    np.random.seed(seed)
    for i in range(num_samples):
        if i % round(0.1 * num_samples) == 0 and verbose:
            print(f"{i} samples complete")
        # take a sample of the dataset, subset rows and columns of adj matrices
        # this could look like an N-size list of indices
        # sample = sorted(random.sample(indices, round(percent_sample * n)))
        sample = np.sort(np.random.choice(n, round(percent_sample * n), replace=False))
        # tally which patients clustered together
        add_cooccurrences_to_matrix(sampled_together, sample)
        # run SNF on subset
        subset = [
            a.take(sample, axis=0).take(sample, axis=1) for a in affinity_mat_list
        ]
        fused_network = SNF(subset, K=K, t=t)
        # perform spectral clustering on SNF similarity matrix
        clusters = spectral_clustering(fused_network, n_clusters=C, **kwargs)
        for j in np.unique(clusters):
            _ = np.where(clusters == j)[0]  # for some reason returns tuple hence [0]
            coclustered = sample[_]  # get patients where cluster == i
            add_cooccurrences_to_matrix(clustered_together, coclustered)
    if verbose:
        print(f"Sampling complete")
    # create new array with percentage of times patients cluster together if sampled together
    cluster_frequency = clustered_together / sampled_together
    # set diagonal elements to mean of the array
    np.fill_diagonal(cluster_frequency, cluster_frequency.mean())
    sparse_array = (cluster_frequency > (1 / C) ** 2) * cluster_frequency
    return cluster_frequency, sparse_array
