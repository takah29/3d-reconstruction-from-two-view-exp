import numpy as np


def unit_vec(x):
    """単位ベクトルを求める"""
    assert x.shape == (3,)
    return x / np.linalg.norm(x)


def sample_normal_dist(scale, n):
    return np.random.normal(0, scale, (n, 3))


def add_noise(X, scale):
    return X + np.random.normal(0, scale, X.shape)
