import numpy as np

from scipy.linalg import eig
from .utils import unit_vec

EPS = 1e-8


def _cofactor(A, i, j):
    """余因子行列の行列式を計算する"""
    X = np.delete(A, i, axis=0)
    X = np.delete(X, j, axis=1)
    a_ij = (-1) ** (i + j) * np.linalg.det(X)

    return a_ij


def _adjugate_mat(A):
    """余因子行列を計算する"""
    max_i, max_j = A.shape
    res = np.zeros(A.shape)
    for i in range(max_i):
        for j in range(max_j):
            res[i, j] = _cofactor(A, i, j)

    return res.T


def _calc_xi(x1, x2, f0):
    """
    x, x' = x1, x2, (x)i1 = xi, (x)i2 = yiとしたとき、以下のデータを返す
    xi = [
        [x1*x'1, x1*y'1, f0*x1, y1*x'1, y1*y'1, f0*y1, f0*x'1, f0*y'1, f0**2]
        [x2*x'2, x2*y'2, f0*x2, y2*x'2, y2*y'2, f0*y2, f0*x'2, f0*y'2, f0**2]
        ...
        [xn*x'n, xn*y'n, f0*xn, yn*x'n, yn*y'n, f0*yn, f0*x'n, f0*y'n, f0**2]
    ]
    xi.shape = (n, 9)
    """
    x1_ext = np.hstack((x1, f0 * np.ones((x1.shape[0], 1))))
    x2_ext = np.hstack((x2, f0 * np.ones((x2.shape[0], 1))))
    x1_ext = np.repeat(x1_ext, 3, axis=1)
    x2_ext = np.tile(x2_ext, 3)
    xi = x1_ext * x2_ext

    return xi


def _calc_V0_xi(x1, x2, f0):
    """各データの正規化共分散行列を求める

    V0_xi.shape = (n, 9, 9)
    """
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


def _calc_normalize_mat(x):
    """データ点を原点中心に移動して、平均ノルムがsqrt(2)となる正規化行列を求める"""

    x_ = x[:, :2]
    m = x_.mean(axis=0)
    s = np.sqrt(2) / np.linalg.norm(x_ - m, axis=1).mean()
    S = np.array([[s, 0], [0, s]])
    W = np.block([[S, -s * m[:, np.newaxis]], [np.zeros(2), 1]])

    return W


def _correct_rank(F):
    """基礎行列Fのランクを補正する

    rank(F) = 2, norm(F) = 1 にする
    """

    U, S, Vt = np.linalg.svd(F)
    s12_norm = np.linalg.norm(S[:2])
    S[:2] /= s12_norm
    S[2] = 0.0
    F_ = U @ np.diag(S) @ Vt

    return F_


def _correct_rank_to_optimal(F, x1, x2, f0):
    def calc_P_theta(theta):
        return np.eye(theta.shape[0]) - theta[:, np.newaxis] @ theta[np.newaxis]

    def calc_V0_theta(M):
        S, U = np.linalg.eig(M)
        desc_idx = np.argsort(S)[::-1]
        S = S[desc_idx]
        U = U[:, desc_idx].T

        # (9, 9)
        V0_theta = (U[..., np.newaxis] @ U[:, np.newaxis, :] / S).mean(axis=0)

        return V0_theta

    xi = _calc_xi(x1, x2, f0)

    # (3, 3) -> (9,)
    theta = F.ravel()

    # (9, 9)
    P_theta = calc_P_theta(theta)

    # (n, 9)
    P_theta_xi = xi @ P_theta.T

    # (n, 9, 1) @ (n, 1, 9) -> (n, 9, 9)
    num = P_theta_xi[..., np.newaxis] @ P_theta_xi[:, np.newaxis, :]

    # (n, 9, 9)
    V0_xi = _calc_V0_xi(x1, x2, f0)

    # (9, ) @ (n, 9, 9) @ (9, 1) -> (n, 1) ?
    denom = theta @ (V0_xi @ theta[:, np.newaxis])

    # (9, 9)
    M = (num / denom[..., np.newaxis]).mean(axis=0)

    # (9, 9)
    V0_theta = calc_V0_theta(M)

    while True:
        # Fの余因子行列を転置して1次元化する
        theta_dagger = _adjugate_mat(theta.reshape(3, 3)).T.ravel()

        V0_theta_theta_dagger = V0_theta @ theta_dagger
        alpha = (theta_dagger @ theta * V0_theta_theta_dagger) / (
            3 * theta_dagger @ V0_theta_theta_dagger
        )
        theta = unit_vec(theta - alpha)
        P_theta = calc_P_theta(theta)
        V0_theta = P_theta @ V0_theta @ P_theta

        if np.linalg.norm(theta_dagger.dot(theta)) < EPS:
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
        W1 = _calc_normalize_mat(x1_ext)
        W2 = _calc_normalize_mat(x2_ext)

        x1_ext = x1_ext @ W1.T
        x2_ext = x2_ext @ W2.T

    x1_ext = np.repeat(x1_ext, 3, axis=1)
    x2_ext = np.tile(x2_ext, 3)

    M = x1_ext * x2_ext
    _, S, Vt = np.linalg.svd(M)
    F = Vt[-1].reshape(3, 3)

    # rank(F) = 2に補正する
    if optimal:
        F_ = _correct_rank_to_optimal(F, x1, x2, f0)
    else:
        F_ = _correct_rank(F)

    if normalize:
        F_ = W1.T @ F_ @ W2

    return F_


def calc_fundamental_matrix_taubin_method(x1, x2, f0):
    xi = _calc_xi(x1, x2, f0)

    # (n, 9, 1) @ (n, 1, 9) -> (n, 9, 9) -> (9, 9)
    M = (xi[..., np.newaxis] @ xi[:, np.newaxis, :]).mean(axis=0)
    # (n, 9, 9) -> (9, 9)
    N = _calc_V0_xi(x1, x2, f0).mean(axis=0)

    S, U = eig(M, N)
    F = U[:, np.argmin(S)].reshape(3, 3)

    return F


def calc_fundamental_matrix_extended_fns_method(x1, x2, f0):
    """拡張FNS法で基礎行列Fを求める"""
    # Initial value
    theta = calc_fundamental_matrix_taubin_method(x1, x2, f0).ravel()

    # (n, 9)
    xi = _calc_xi(x1, x2, f0)
    # (n, 9, 1) @ (n, 1, 9) -> (n, 9, 9)
    M_num = xi[..., np.newaxis] @ xi[:, np.newaxis, :]
    # (n, 9, 9)
    V0_xi = _calc_V0_xi(x1, x2, f0)

    while True:
        # (n, 9, 9) @ (9, 1) -> (n, 9, 1) -> (n, 9) -> (9, n)
        V0_xi_theta_t = (V0_xi @ theta[:, np.newaxis]).squeeze(2).T
        # (9, ) @ (9, n) -> -> (n, ) -> (n, 1, 1)
        denom = (theta @ V0_xi_theta_t)[:, np.newaxis, np.newaxis]
        # (n, 9, 9) / (n, 1, 1) -> (n, 9, 9) -> (9, 9)
        M = (M_num / denom).mean(axis=0)

        # (n, 9) @ (9, 1) -> (n, 1)
        L_num = (xi @ theta[:, np.newaxis]) ** 2
        # (n ,1, 1) / (n, 1, 1) * (n, 9, 9) -> (n, 9, 9) -> (9, 9)
        L = (L_num[..., np.newaxis] / (denom**2) * V0_xi).mean(axis=0)

        # (9, )
        unit_theta_dagger = unit_vec(_adjugate_mat(theta.reshape(3, 3)).T.ravel())

        # (9, 9) - (9, 1) @ (1, 9) -> (9, 9)
        P_theta_dagger = (
            np.eye(9) - unit_theta_dagger[:, np.newaxis] @ unit_theta_dagger[np.newaxis]
        )

        X = M - L
        # (9, 9) @ (9, 9) @ (9, 9) -> (9, 9)
        Y = P_theta_dagger @ X @ P_theta_dagger

        # ２つの小さい固有値に対応する固有ベクトルVを取得する
        S, U = np.linalg.eig(Y)
        V = U[:, np.argsort(S)[:2]]

        # (9, 2) @ (2, 9) @ (9, 1) -> (9, 1)
        theta_hat = V @ V.T @ theta[:, np.newaxis]  # Equivalent to θ^ = <θ, v1>v1 + <θ, v2>v2
        # (9, 9) @ (9, 1) -> (9, 1) -> (9, )
        theta_prime = unit_vec(P_theta_dagger @ theta_hat).ravel()

        if np.linalg.norm(theta_prime - theta) < EPS:
            break

        theta = unit_vec(theta + theta_prime)

    return theta_prime.reshape(3, 3)


def find_fundamental_matrix_cv(x1, x2):
    import cv2

    F, _ = cv2.findFundamentalMat(x2, x1, cv2.FM_LMEDS)
    return F


def optimize_corresponding_points(F, x1, x2, f0):
    S0 = float("inf")

    x1_hat = x1.copy()
    x2_hat = x2.copy()

    x1_tilda = np.zeros(x1.shape)
    x2_tilda = np.zeros(x2.shape)

    def calc_xi_star(x1_hat, x2_hat, x1_tilda, x2_tilda, f0):
        x_h = x1_hat[:, 0]  # x^
        y_h = x1_hat[:, 1]  # y^
        xp_h = x2_hat[:, 0]  # x^'
        yp_h = x2_hat[:, 1]  # y^'

        x_t = x1_tilda[:, 0]  # x~
        y_t = x1_tilda[:, 1]  # y~
        xp_t = x2_tilda[:, 0]  # x~'
        yp_t = x2_tilda[:, 1]  # y~'

        xi_star = np.zeros((9, x1_hat.shape[0]))
        xi_star[0] = x_h * xp_h + xp_h * x_t + x_h * xp_t
        xi_star[1] = x_h * yp_h + yp_h * x_t + x_h * yp_t
        xi_star[2] = f0 * (x_h + x_t)
        xi_star[3] = y_h * xp_h + xp_h * y_t + y_h * xp_t
        xi_star[4] = y_h * yp_h + yp_h * y_t + y_h * yp_t
        xi_star[5] = f0 * (y_h + y_t)
        xi_star[6] = f0 * (xp_h + xp_t)
        xi_star[7] = f0 * (yp_h + yp_t)
        xi_star[8] = f0 * f0

        return xi_star.T

    theta = F.ravel()

    while True:
        V0_xi_hat = _calc_V0_xi(x1_hat, x2_hat, f0)
        xi_star = calc_xi_star(x1_hat, x2_hat, x1_tilda, x2_tilda, f0)

        # (n, 9) @ (9, 1) -> (n, 1) -> (n, )
        num = (xi_star @ theta[:, np.newaxis]).ravel()
        # (n, 9, 9) @ (9, 1) -> (n, 9, 1) -> (n, 9) -> (9, n)
        V0_xi_hat_theta_t = (V0_xi_hat @ theta[:, np.newaxis]).squeeze(2).T
        # (9, ) @ (n, 9) -> (n, )
        denom = theta @ V0_xi_hat_theta_t
        # (n, ) / (n, ) -> (n, )
        coefs = num / denom

        T1 = theta[:6].reshape(2, 3).T
        x1_hat_ext = np.hstack((x1_hat, f0 * np.ones((x1_hat.shape[0], 1))))
        # (n, 1) * (n, 3) @ (3, 2) -> (n, 2)
        x1_tilda = coefs[:, np.newaxis] * x1_hat_ext @ T1

        T2 = np.vstack((theta[::3], theta[1::3])).T
        x2_hat_ext = np.hstack((x2_hat, f0 * np.ones((x2_hat.shape[0], 1))))
        # (n, 1) * (n, 3) @ (3, 2) -> (n, 2)
        x2_tilda = coefs[:, np.newaxis] * x2_hat_ext @ T2

        x1_hat = x1 - x1_tilda
        x2_hat = x2 - x2_tilda

        S = (x1_tilda**2 + x2_tilda**2).sum()
        if np.abs(S0 - S) < 1e-4:
            break

        S0 = S

    return x1_hat, x2_hat


def remove_outliers(x1, x2, f0, d):
    """RANSACによるアウトライア除去を行う"""
    max_n = 0
    count = 0

    while True:
        xi = _calc_xi(x1, x2, f0)
        rnd_ind = np.random.choice(np.arange(x1.shape[0]), 8, replace=False)
        sub_xi = xi[rnd_ind]

        # (8, 9, 1) @ (8, 1, 9) -> (8, 9, 9) -> (9, 9)
        M = (sub_xi[..., np.newaxis] @ sub_xi[:, np.newaxis, :]).sum(axis=0)

        S, U = np.linalg.eig(M)
        theta_ = U[:, np.argmin(S)]

        # (n, 9) @ (9, 1) -> (n, 1) -> (n, )
        num = (xi @ theta_[:, np.newaxis]).squeeze(1) ** 2

        # (n, 9, 9) @ (9, 1) -> (n, 9, 1) -> (n, 9) -> (9, n)
        V0_xi_theta_t = (_calc_V0_xi(x1, x2, f0) @ theta_[:, np.newaxis]).squeeze(2).T
        # (9, ) @ (9, n) -> (n, )
        denom = theta_ @ V0_xi_theta_t

        satisfied = (num / denom) < 2 * d**2
        n = satisfied.sum()

        if n > max_n:
            max_n = n
            max_satisfied = satisfied
            count = 0
        else:
            count += 1
            if count >= 20:
                break

    return x1[max_satisfied], x2[max_satisfied], ~max_satisfied
