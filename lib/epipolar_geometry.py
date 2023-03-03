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

    ratio = 0.7
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append([m])

    good_matches = sorted(good_matches, key=lambda x: x[0].distance)

    # img3 = cv2.drawMatchesKnn(
    #     img1,
    #     key_point1,
    #     img2,
    #     key_point2,
    #     good_matches,
    #     None,
    #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    # )

    # cv2.imshow("image", img3)
    # cv2.waitKey()

    query_indices, train_indices = get_corresponding_indices(good_matches)

    X1, X2 = get_keypoint_matrix(key_point1, query_indices, key_point2, train_indices)

    return X1, X2


def calc_fundamental_matrix_8points_method(x1, x2):
    """8点法で基礎行列Fを求める

    f = ravel(F) として Mf = 0 を解く
    データ数は8点以上あってもいい
    """
    assert x1.ndim == 2 and x1.shape[1] == 2
    assert x2.ndim == 2 and x2.shape[1] == 2
    assert x1.shape == x2.shape

    x1_ext = np.hstack((x1, np.ones((x1.shape[0], 1))))
    x2_ext = np.hstack((x2, np.ones((x2.shape[0], 1))))

    x1_ext = np.tile(x1_ext, 3)
    x2_ext = np.repeat(x2_ext, 3, axis=1)

    M = x1_ext * x2_ext
    _, S, Vt = np.linalg.svd(M)
    F = Vt[-1].reshape(3, 3)

    # Rank(F) = 2 にする
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0
    F = U @ np.diag(S) @ Vt

    return F


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

    e = np.linalg.eig(FFt)[1][:, -1]
    e_prime = np.linalg.eig(FtF)[1][:, -1]

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
    t = np.linalg.eig(E @ E.T)[1][:, -1]

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
