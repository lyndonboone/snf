import numpy as np

from snftools.utils import dominateset


def SNF(Wall, K=20, t=20):
    """
    Perform similarity network fusion on a set of affinity matrices.

    This function is meant to be a direct translation of the SNF function from
    SNFtool from R to Python. Outputs have been tested and confirmed to be the same
    as the R version in the test_snf unit test included with this package.

    Args:
        Wall: List of affinity matrices.
        K: Number of K-nearest-neighbours (default 20).
        t: Number of fusion iterations (default 20).
    """

    def normalize(X):
        row_sums_less_diag = np.sum(X, axis=1, keepdims=True) - np.expand_dims(
            np.diagonal(X), axis=1
        )
        # if row sum = 0, set to 1 to avoid div by zero
        row_sums_less_diag[row_sums_less_diag == 0] = 1.0
        X = X / (2 * row_sums_less_diag)
        np.fill_diagonal(X, 0.5)
        return X

    LW = len(Wall)
    # newW = [None] * LW
    # nextW = [None] * LW
    # newW = np.empty(LW, dtype=np.float64)
    # nextW = np.empty(LW, dtype=np.float64)
    newW = np.empty((LW,) + Wall[0].shape, dtype=np.float64)
    nextW = np.empty((LW,) + Wall[0].shape, dtype=np.float64)

    # Convert arrays in Wall to np.float64 data type
    Wall = [arr.astype(np.float64) for arr in Wall]

    # normalize affinity matrices
    for i in range(LW):
        Wall[i] = normalize(Wall[i])
        Wall[i] = (Wall[i] + Wall[i].T) / 2.0
    # calculate the local affinity array using KNNs
    for i in range(LW):
        newW[i] = dominateset(Wall[i], K)
    # perform diffusion for t iterations
    for i in range(t):
        for j in range(LW):
            sumWJ = np.zeros(Wall[j].shape, dtype=np.float64)
            for k in range(LW):
                if k != j:
                    sumWJ += Wall[k]
            # nextW[j] = newW[j] @ (sumWJ / (LW - 1)) @ newW[j].T
            nextW[j] = newW[j].dot(sumWJ / (LW - 1)).dot(newW[j].T)
        # normalize each new network
        for j in range(LW):
            Wall[j] = normalize(nextW[j])
            Wall[j] = (Wall[j] + Wall[j].T) / 2.0
    # construct combined affinity matrix by summing diffused matrices
    W = np.zeros(Wall[0].shape, dtype=np.float64)
    for i in range(LW):
        W += Wall[i]
    W = W / float(LW)
    W = normalize(W)
    W = (W + W.T) / 2.0
    return W
