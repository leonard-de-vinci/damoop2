"""Minimal Latin Hypercube sampler (numpy only)."""
from __future__ import annotations

import numpy as np


def lhs(n_samples: int, n_dims: int, seed: int = 0) -> np.ndarray:
    """Return an (n_samples, n_dims) array of samples in [0, 1).

    Each dimension is stratified into n_samples bins; each bin is used once
    per dimension, with bin order independently permuted per dimension.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, n_dims), dtype=float)
    jitter = rng.random((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        out[:, d] = (perm + jitter[:, d]) / n_samples
    return out
