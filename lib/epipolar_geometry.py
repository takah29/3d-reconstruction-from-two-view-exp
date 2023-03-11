import numpy as np
import cv2

from lib.utils import unit_vec


def get_corresponding_indices(matches):
    query_indices = [x[0].queryIdx for x in matches]
    train_indices = [x[0].trainIdx for x in matches]

    return query_indices, train_indices


def get_keypoint_matrix(key_point1, query_indices, key_point2, train_indices):
    X1 = np.vstack([key_point1[i].pt for i in query_indices])
    X2 = np.vstack([key_point2[i].pt for i in train_indices])

    return X1, X2


def detect_corresponding_points(img1, img2):
    """2つの画像から対応点を検出する"""
    detector = cv2.AKAZE_create()

    key_point1, descript1 = detector.detectAndCompute(img1, None)
    key_point2, descript2 = detector.detectAndCompute(img2, None)

    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf_matcher.knnMatch(descript1, descript2, k=2)

    ratio = 0.5
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append([m])

    good_matches = sorted(good_matches, key=lambda x: x[0].distance)

    img3 = cv2.drawMatchesKnn(
        img1,
        key_point1,
        img2,
        key_point2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imshow("image", img3)
    cv2.waitKey()

    query_indices, train_indices = get_corresponding_indices(good_matches)

    X1, X2 = get_keypoint_matrix(key_point1, query_indices, key_point2, train_indices)

    return X1, X2


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


def calc_epipole(F):
    U, _, Vt = np.linalg.svd(F)
    e1 = U.T[-1]
    e2 = Vt[-1]

    return np.vstack((e1 / e1[2], e2 / e2[2]))[:, :2]


def calc_focal_length(F, f0):
    """基礎行列Fから焦点距離f,f_primeを計算する"""
    # 最小固有値に対する固有ベクトルを取得する
    FFt = F @ F.T
    FtF = F.T @ F

    S, P = np.linalg.eig(FFt)
    e = P[:, np.argmin(S)]
    S, P = np.linalg.eig(FtF)
    e_prime = P[:, np.argmin(S)]

    k = np.array([[0.0], [0.0], [1.0]])
    Fk = F @ k
    Ftk = F.T @ k
    Fk_norm2 = np.linalg.norm(Fk) ** 2
    Ftk_norm2 = np.linalg.norm(Ftk) ** 2
    e_cross_k_norm2 = np.linalg.norm(np.cross(e, k.T)) ** 2
    e_prime_cross_k_norm2 = np.linalg.norm(np.cross(e_prime, k.T)) ** 2
    k_dot_Fk = k.T @ Fk
    k_dot_FFtFk = k.T @ (FFt @ Fk)

    xi = (Fk_norm2 - (k_dot_FFtFk * e_prime_cross_k_norm2 / k_dot_Fk)) / (
        e_prime_cross_k_norm2 * Ftk_norm2 - k_dot_Fk**2
    )
    ita = (Ftk_norm2 - (k_dot_FFtFk * e_cross_k_norm2 / k_dot_Fk)) / (
        e_cross_k_norm2 * Fk_norm2 - k_dot_Fk**2
    )

    f = f0 / np.sqrt(1 + xi[0][0])
    f_prime = f0 / np.sqrt(1 + ita[0][0])

    return f, f_prime


def calc_motion_parameters(F, x1, x2, f, f_prime, f0):
    f0_inv = 1 / f0
    E = np.diag((f0_inv, f0_inv, 1 / f)) @ F @ np.diag((f0_inv, f0_inv, 1 / f_prime))
    S, P = np.linalg.eig(E @ E.T)
    t = P[:, np.argmin(S)]

    x1_ext = np.hstack((x1 / f, np.ones((x1.shape[0], 1))))
    x2_ext = np.hstack((x2 / f_prime, np.ones((x2.shape[0], 1))))
    x2_ext_E = x2_ext @ E.T

    scalar_triple_product_sum = sum(
        [np.linalg.det(np.vstack((t, x, y))) for x, y in zip(x1_ext, x2_ext_E)]
    )
    if scalar_triple_product_sum <= 0.0:
        t = -t

    K = -np.cross(t, E.T, axisc=0)
    U, _, Vt = np.linalg.svd(K)
    R = U @ np.diag((1, 1, np.linalg.det(U @ Vt))) @ Vt
    return R, t


def calc_camera_matrix(f, f_prime, R, t, f0):
    P = np.diag((f, f, f0)) @ np.block([np.eye(3), np.zeros((3, 1))])
    P_prime = np.diag((f_prime, f_prime, f0)) @ np.block([R.T, -R.T @ t[:, np.newaxis]])

    return P, P_prime


def reconstruct_3d_points(x1, x2, P, P_prime, f0):
    # (4, 4)
    K1 = f0 * np.vstack((P[:2], P_prime[:2]))

    # (4, 4)
    K2 = np.vstack(
        (
            np.repeat(P[-1][np.newaxis, :], 2, axis=0),
            np.repeat(P_prime[-1][np.newaxis, :], 2, axis=0),
        )
    )

    # (n, 4, 4)
    K3 = np.repeat(np.hstack((x1, x2))[:, :, np.newaxis], 4, axis=2)

    K = K1 - K2 * K3
    T = K[:, :, :3]
    p = K[:, :, -1:]

    X_ = -np.linalg.pinv(T) @ p

    return X_[:, :, 0]
