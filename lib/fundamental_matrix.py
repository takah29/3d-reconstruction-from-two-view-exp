import numpy as np

from .utils import unit_vec


def calc_normalize_mat(x):
    """データ点を原点中心に移動して、平均ノルムがsqrt(2)となる正規化行列を求める"""
    m = x.mean(axis=0)
    s = np.sqrt(2) / np.linalg.norm(x - m, axis=1).mean()
    S = np.array([[s, 0], [0, s]])
    W = np.block([[S, -s * m[:, np.newaxis]], [np.zeros(2), 1]])

    return W


def correct_rank(F):
    """基礎行列Fのランクを補正する

    rank(F) = 2, norm(F) = 1 にする
    """

    U, S, Vt = np.linalg.svd(F)
    s12_norm = np.linalg.norm(S[:2])
    S[:2] /= s12_norm
    S[2] = 0.0
    F_ = U @ np.diag(S) @ Vt

    return F_


def correct_rank_to_optimal(F, x1, x2, f0):
    def calc_P_theta(theta):
        return np.eye(theta.shape[0]) - theta[:, np.newaxis] @ theta[np.newaxis]

    def cofactor(A, i, j):
        X = np.delete(A, i, axis=0)
        X = np.delete(X, j, axis=1)
        a_ij = (-1) ** (i + j) * np.linalg.det(X)

        return a_ij

    def adjugate_mat(A):
        max_i, max_j = A.shape
        res = np.zeros(A.shape)
        for i in range(max_i):
            for j in range(max_j):
                res[i, j] = cofactor(A, i, j)

        return res.T

    def calc_V0_xi(x1, x2, f0):
        V0_xi = np.zeros((9, 9, x1.shape[0]))

        x1_0 = x1[:, 0]  # x
        x1_1 = x1[:, 1]  # y
        x2_0 = x2[:, 0]  # x'
        x2_1 = x2[:, 1]  # y'

        a = x1_0**2 + x2_0**2
        b = x1_0**2 + x2_1**2
        c = x1_1**2 + x2_0**2
        d = x1_1**2 + x2_1**2
        e = x1_0 * x1_1
        f = x2_0 * x2_1
        g = f0 * x1_0
        h = f0 * x1_1
        i = f0 * x2_0
        j = f0 * x2_1
        k = f0**2

        V0_xi[0, 0] = a
        V0_xi[1, 1] = b
        V0_xi[3, 3] = c
        V0_xi[4, 4] = d
        V0_xi[(0, 3, 1, 4), (3, 0, 4, 1)] = e
        V0_xi[(0, 1, 3, 4), (1, 0, 4, 3)] = f
        V0_xi[(0, 1, 6, 7), (6, 7, 0, 1)] = g
        V0_xi[(3, 4, 6, 7), (6, 7, 3, 4)] = h
        V0_xi[(0, 2, 3, 5), (2, 0, 5, 3)] = i
        V0_xi[(1, 2, 4, 5), (2, 1, 5, 4)] = j
        V0_xi[(2, 5, 6, 7), (2, 5, 6, 7)] = k

        V0_xi = V0_xi.transpose(2, 0, 1)
        return V0_xi

    def calc_V0_theta(M):
        S, U = np.linalg.eig(M)
        desc_idx = np.argsort(S)[::-1]
        S = S[desc_idx]
        U = U[:, desc_idx].T

        # (9, 9)
        V0_theta = (U[..., np.newaxis] @ U[:, np.newaxis, :] / S).mean(axis=0)

        return V0_theta

    x1_ext = np.hstack((x1, np.ones((x1.shape[0], 1))))
    x2_ext = np.hstack((x2, np.ones((x2.shape[0], 1))))
    x1_ext = np.repeat(x1_ext, 3, axis=1)
    x2_ext = np.tile(x2_ext, 3)
    xi_array = x1_ext * x2_ext

    # (3, 3) -> (9,)
    theta = F.ravel()

    # (9, 9)
    P_theta = calc_P_theta(theta)

    # (n, 9)
    P_theta_xi = xi_array @ P_theta.T

    # (n, 9, 1) @ (n, 1, 9) -> (n, 9, 9)
    num = P_theta_xi[..., np.newaxis] @ P_theta_xi[:, np.newaxis, :]

    # (n, 9, 9)
    V0_xi = calc_V0_xi(x1, x2, f0)

    # (9, ) @ (n, 9, 9) @ (9, 1) -> (n, 1) ?
    denom = theta @ (V0_xi @ theta[:, np.newaxis])

    # (9, 9)
    M = (num / denom[..., np.newaxis]).mean(axis=0)

    # (9, 9)
    V0_theta = calc_V0_theta(M)

    while True:
        # Fの余因子行列を転置して1次元化する
        theta_dagger = adjugate_mat(theta.reshape(3, 3)).T.ravel()

        V0_theta_theta_dagger = V0_theta @ theta_dagger
        alpha = (theta_dagger @ theta * V0_theta_theta_dagger) / (
            3 * theta_dagger @ V0_theta_theta_dagger
        )
        theta = unit_vec(theta - alpha)
        P_theta = calc_P_theta(theta)
        V0_theta = P_theta @ V0_theta @ P_theta

        if np.linalg.norm(theta_dagger.dot(theta)) < 1e-8:
            break

    return theta.reshape(3, 3)


def calc_fundamental_matrix_8points_method(x1, x2, f0, normalize=True, optimal=True):
    """8点法で基礎行列Fを求める

    (x1_ext, Fx2_ext) = 0 となるFを求める
    f = ravel(F) として Mf = 0 を解く
    データ数は8点以上あってもいい
    """
    assert x1.ndim == 2 and x1.shape[1] == 2
    assert x2.ndim == 2 and x2.shape[1] == 2
    assert x1.shape == x2.shape

    x1_ext = np.hstack((x1 / f0, np.ones((x1.shape[0], 1))))
    x2_ext = np.hstack((x2 / f0, np.ones((x2.shape[0], 1))))

    if normalize:
        W1 = calc_normalize_mat(x1)
        W2 = calc_normalize_mat(x2)

        x1_ext = x1_ext @ W1.T
        x2_ext = x2_ext @ W2.T

    x1_ext = np.repeat(x1_ext, 3, axis=1)
    x2_ext = np.tile(x2_ext, 3)

    M = x1_ext * x2_ext
    _, S, Vt = np.linalg.svd(M)
    F = Vt[-1].reshape(3, 3)

    # rank(F) = 2に補正する
    if optimal:
        F_ = correct_rank_to_optimal(F, x1, x2, f0)
    else:
        F_ = correct_rank(F)

    if normalize:
        F_ = W1.T @ F_ @ W2

    return F_