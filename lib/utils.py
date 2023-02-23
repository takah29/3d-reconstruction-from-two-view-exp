import numpy as np
from numpy.typing import NDArray


def unit_vec(x: NDArray) -> NDArray:
    """単位ベクトルを求める"""
    assert x.shape == (3,)

    return x / np.linalg.norm(x)


def sample_normal_dist(scale: float, n: int):
    return np.random.normal(0, scale, (n, 3))


def add_noise(X, scale: float) -> NDArray:
    return X + np.random.normal(0, scale, X.shape)
