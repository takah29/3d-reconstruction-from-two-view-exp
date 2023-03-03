import cv2
import matplotlib.pyplot as plt
import numpy as np

from lib.epipolar_geometry import (
    calc_focal_length,
    calc_fundamental_matrix_8points_method,
    calc_motion_parameters,
    detect_corresponding_points,
    calc_camera_matrix,
    reconstruct_3d_points,
)
from lib.visualization import init_3d_ax, plot_3d_basis, plot_3d_points


def main():
    # 画像ファイル読み込み
    img1 = cv2.imread("./images/002.jpg")
    img2 = cv2.imread("./images/003.jpg")

    # 対応点の検出
    x1, x2 = detect_corresponding_points(img1, img2)

    # 基礎行列Fの計算
    F = calc_fundamental_matrix_8points_method(x1, x2)

    # 焦点距離f, f_primeの計算
    f0 = 1.0
    f, f_prime = calc_focal_length(F, f0)

    # 運動パラメータの計算
    R, t = calc_motion_parameters(F, x1, x2, f, f_prime, f0)

    # 対応点の補正
    # X1_, X2_ = optimize_matches(X1, X2, F)

    # カメラ行列の取得
    P, P_prime = calc_camera_matrix(f, f_prime, R, t, f0)

    # 3次元復元
    X_ = reconstruct_3d_points(x1, x2, P, P_prime, f0)

    ax = init_3d_ax()
    plot_3d_basis(np.zeros(3), np.eye(3), ax)
    plot_3d_basis(t, R, ax)
    plot_3d_points(X_, ax, "blue")

    plt.show()


if __name__ == "__main__":
    main()
