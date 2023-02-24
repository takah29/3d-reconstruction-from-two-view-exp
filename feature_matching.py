import numpy as np
import cv2


def calc_fundamental_matrix_8points_method(X1, X2):
    """8点法で基礎行列Fを求める

    f = ravel(F) として Mf = 0 を解く
    データ数は8点以上あってもいい
    """
    assert X1.ndim == 2 and X1.shape[1] == 2
    assert X2.ndim == 2 and X2.shape[1] == 2
    assert X1.shape == X2.shape

    X1_ext = np.hstack((X1, np.ones((X1.shape[0], 1))))
    X2_ext = np.hstack((X2, np.ones((X2.shape[0], 1))))

    X1_ext = np.tile(X1_ext, 3)
    X2_ext = np.repeat(X2_ext, 3, axis=1)

    M = X1_ext * X2_ext
    _, _, V = np.linalg.svd(M)
    F = V[-1].reshape(3, 3)

    # Rank(F) = 2 にする
    U, S, V = np.linalg.svd(F)
    S[2] = 0.0
    F = U @ np.diag(S) @ V

    return F


def get_matches_indices(matches):
    query_indices = [x[0].queryIdx for x in matches]
    train_indices = [x[0].trainIdx for x in matches]

    return query_indices, train_indices


def get_keypoint_matrix(key_point1, query_indices, key_point2, train_indices):
    X1 = np.vstack([key_point1[i].pt for i in query_indices])
    X2 = np.vstack([key_point2[i].pt for i in train_indices])

    return X1, X2


def calc_focal_length(F, f0):
    """基礎行列Fから焦点距離f,f_primeを計算する"""
    # 最小固有値に対する固有ベクトルを取得する
    FFt = F @ F.T
    FtF = F.T @ F
    e = np.linalg.eig(FFt)[1][-1]
    e_prime = np.linalg.eig(FtF)[1][-1]

    k = np.array([[0.0], [0.0], [1.0]])
    Fk = F @ k
    Ftk = F.T @ k
    Fk_norm2 = np.linalg.norm(Fk) ** 2
    Ftk_norm2 = np.linalg.norm(Ftk) ** 2
    e_cross_k_norm2 = np.linalg.norm(np.cross(e, k)) ** 2
    e_prime_cross_k_norm2 = np.linalg.norm(np.cross(e_prime, k)) ** 2
    k_dot_Fk = k.T @ Fk
    k_dot_FFtFk = k.T @ (FFt @ Fk)

    xi = (Fk_norm2 - (k_dot_FFtFk * e_prime_cross_k_norm2 / k_dot_Fk)) / (
        e_prime_cross_k_norm2 * Ftk_norm2 - k_dot_Fk**2
    )
    ita = (Ftk_norm2 - (k_dot_FFtFk * e_cross_k_norm2 / k_dot_Fk)) / (
        e_cross_k_norm2 * Fk_norm2 - k_dot_Fk**2
    )

    f = f0 / (np.sqrt(1 + xi))
    f_prime = f0 / (np.sqrt(1 + ita))

    return f, f_prime


def main():
    img1 = cv2.imread("./images/002.jpg")
    img2 = cv2.imread("./images/003.jpg")

    detector = cv2.AKAZE_create()

    key_point1, descript1 = detector.detectAndCompute(img1, None)
    key_point2, descript2 = detector.detectAndCompute(img2, None)

    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf_matcher.knnMatch(descript1, descript2, k=2)

    ratio = 0.7
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])

    good = sorted(good, key=lambda x: x[0].distance)

    img3 = cv2.drawMatchesKnn(
        img1,
        key_point1,
        img2,
        key_point2,
        good,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imshow("image", img3)
    cv2.waitKey()

    query_indices, train_indices = get_matches_indices(good)

    X1, X2 = get_keypoint_matrix(key_point1, query_indices, key_point2, train_indices)
    print(X1.shape, X2.shape)

    F = calc_fundamental_matrix_8points_method(X1, X2)


if __name__ == "__main__":
    main()
