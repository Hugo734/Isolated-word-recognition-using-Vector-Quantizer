"""Linde-Buzo-Gray (LBG) vector quantization."""
import numpy as np


def assign_clusters(vectors, codebook):
    """Vectorized nearest-centroid assignment. Returns (N,) index array."""
    diff = vectors[:, np.newaxis, :] - codebook[np.newaxis, :, :]  # (N, K, D)
    sq_dists = np.sum(diff ** 2, axis=2)                           # (N, K)
    return np.argmin(sq_dists, axis=1)                             # (N,)


def _update_centroids(vectors, assignments, k):
    d = vectors.shape[1]
    centroids = np.empty((k, d))
    for j in range(k):
        members = vectors[assignments == j]
        centroids[j] = members.mean(axis=0) if len(members) else vectors[np.random.randint(len(vectors))]
    return centroids


def lbg(vectors, codebook_size, max_iter=150, tol=1e-4):
    """
    LBG algorithm on LSF vectors (Euclidean distance).
    vectors:       (N, D) training vectors
    codebook_size: target number of codewords (ideally a power of 2)
    Returns:       codebook (codebook_size, D)
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    noise = 0.01 * np.std(vectors, axis=0) + 1e-8

    codebook = vectors.mean(axis=0, keepdims=True)
    current_k = 1

    while current_k < codebook_size:
        codebook = np.vstack([codebook + noise, codebook - noise])
        current_k = len(codebook)

        prev_d = np.inf
        for _ in range(max_iter):
            assignments = assign_clusters(vectors, codebook)
            codebook = _update_centroids(vectors, assignments, current_k)
            d = np.mean(np.sum((vectors - codebook[assignments]) ** 2, axis=1))
            if prev_d > 0 and abs(prev_d - d) / prev_d < tol:
                break
            prev_d = d

    return codebook
