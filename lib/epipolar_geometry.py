import numpy as np
import cv2


EPS = 1e-8


def convert_image_coord_to_screen_coord(points):
    return np.hstack((-points[:, -1:], points[:, :1]))


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

    ratio = 0.6
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


def calc_epipole(F):
    U, _, Vt = np.linalg.svd(F)
    e1 = U.T[-1]
    e2 = Vt[-1]

    return np.vstack((e1 / e1[2], e2 / e2[2]))


def calc_free_focal_length(F, f0, verbose=False):
    """基礎行列Fから焦点距離f,f_primeを計算する

    Reference: http://iim.cs.tut.ac.jp/member/kanatani/papers/new2views.pdf
    """
    # 最小固有値に対する固有ベクトルを取得する
    FFt = F @ F.T
    FtF = F.T @ F

    S, P = np.linalg.eig(FFt)
    e = P[:, np.argmin(S)]
    S, P = np.linalg.eig(FtF)
    e_prime = P[:, np.argmin(S)]

    k = np.array([0.0, 0.0, 1.0])

    Fk = F @ k
    Ftk = F.T @ k

    Fk_norm2 = np.linalg.norm(Fk) ** 2
    Ftk_norm2 = np.linalg.norm(Ftk) ** 2

    k_dot_Fk = k @ Fk
    k_dot_FFtFk = k @ (FFt @ Fk)

    if np.abs(k_dot_Fk) < 0.1 * np.sqrt(min(Fk_norm2, Ftk_norm2)) / f0:
        raise ValueError("Optical axes are crossed.")

    e_cross_k_norm2 = np.linalg.norm(np.cross(e, k)) ** 2
    e_prime_cross_k_norm2 = np.linalg.norm(np.cross(e_prime, k)) ** 2

    xi = (Fk_norm2 - (k_dot_FFtFk * e_prime_cross_k_norm2 / k_dot_Fk)) / (
        e_prime_cross_k_norm2 * Ftk_norm2 - k_dot_Fk**2
    )
    ita = (Ftk_norm2 - (k_dot_FFtFk * e_cross_k_norm2 / k_dot_Fk)) / (
        e_cross_k_norm2 * Fk_norm2 - k_dot_Fk**2
    )

    if verbose:
        print(f"|Fk|^2={Fk_norm2}, |Ftk|^2={Ftk_norm2}, <k, Fk>={k_dot_Fk}, xi={xi}, ita={ita}")

    f = f0 / np.sqrt(1 + xi)
    f_prime = f0 / np.sqrt(1 + ita)

    return f, f_prime


def calc_fixed_focal_length(F, f0):
    """基礎行列Fから焦点距離fをf=f_primeとして計算する

    Reference: http://iim.cs.tut.ac.jp/member/kanatani/papers/new2views.pdf
    """
    FFt = F @ F.T
    FtF = F.T @ F

    k = np.array([[0.0], [0.0], [1.0]])

    Fk = F @ k
    Ftk = F.T @ k

    Fk_norm2 = np.linalg.norm(Fk) ** 2
    Ftk_norm2 = np.linalg.norm(Ftk) ** 2

    k_dot_Fk = (k.T @ Fk)[0][0]
    k_dot_FFtFk = (k.T @ (FFt @ Fk))[0][0]

    F_norm2 = np.linalg.norm(F) ** 2
    FFt_norm2 = np.linalg.norm(FFt) ** 2
    FFtk_norm2 = np.linalg.norm(FFt @ k) ** 2
    FtFk_norm2 = np.linalg.norm(FtF @ k) ** 2

    a1 = k_dot_Fk**4 / 2
    a2 = k_dot_Fk**2 * (Ftk_norm2 + Fk_norm2)
    a3 = (Ftk_norm2 - Fk_norm2) ** 2 / 2 + k_dot_Fk * (4 * (k_dot_FFtFk - k_dot_Fk * F_norm2))
    a4 = 2 * (FFtk_norm2 + FtFk_norm2) - (Ftk_norm2 + Fk_norm2) * F_norm2
    a5 = FFt_norm2 - F_norm2**2 / 2

    if np.abs(k_dot_Fk) < EPS:
        xi = -a4 / (2 * a3)
    else:
        K = np.polynomial.Polynomial([a5, a4, a3, a2, a1])
        K_diff = K.deriv()
        roots = np.sort([x.real for x in K_diff.roots() if x.imag == 0.0])[::-1]
        if len(roots) == 1:
            xi = roots[0]
        else:
            if roots[2] <= -1 or K(roots[2]) < 0 or K(roots[0]) <= K(roots[2]):
                xi = roots[0]
            elif 0 <= K(roots[2]) < K(roots[0]):
                xi = roots[2]
            else:
                xi = -1.0

    return f0 / np.sqrt(1 + xi)


def calc_motion_parameters(F, x1, x2, f, f_prime, f0):
    """運動パラメータを計算する"""
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
        t *= -1

    K = -np.cross(t, E.T, axisc=0)
    U, _, Vt = np.linalg.svd(K)
    R = U @ np.diag((1, 1, np.linalg.det(U @ Vt))) @ Vt
    return R, t


def calc_camera_matrix(f, f_prime, R, t, f0):
    """焦点距離と運動パラメータからカメラ行列を計算する"""
    P = np.diag((f, f, f0)) @ np.block([np.eye(3), np.zeros((3, 1))])
    P_prime = np.diag((f_prime, f_prime, f0)) @ np.block([R.T, -R.T @ t[:, np.newaxis]])

    return P, P_prime


def reconstruct_3d_points(x1, x2, P, P_prime, f0):
    """カメラ行列から3次元点を復元する"""
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


def detect_mirror_image(X):
    """復元後カメラの後ろに像がある（鏡像）を検知する"""
    return np.sum(np.sign(X[:, 2])) <= 0
