import numpy as np
import cv2
import matplotlib.pyplot as plt

from lib.epipolar_geometry import (
    calc_camera_matrix,
    calc_free_focal_length,
    calc_motion_parameters,
    convert_image_coord_to_screen_coord,
    detect_corresponding_points,
    reconstruct_3d_points,
    detect_mirror_image,
)
from lib.fundamental_matrix import (
    calc_fundamental_matrix_extended_fns_method,
    optimize_corresponding_points,
    remove_outliers,
)
from lib.visualization import init_3d_ax, plot_3d_basis, plot_3d_points, plot_2d_points


def main():
    np.random.seed(123)

    # 画像ファイル読み込み
    img1 = cv2.imread("./images/merton_college_I/001.jpg")
    img2 = cv2.imread("./images/merton_college_I/002.jpg")

    # 対応点の検出
    x1, x2, colors = detect_corresponding_points(img1, img2, method="AKAZE", is_show=True)

    # 画像座標をスクリーン座標へ変換
    x1 = convert_image_coord_to_screen_coord(x1, img1.shape[1], img1.shape[0])
    x2 = convert_image_coord_to_screen_coord(x2, img2.shape[1], img2.shape[0])

    f0 = max(img1.shape)

    # アウトライアの除去
    print(f"remove outliers: {x1.shape[0]} -> ", end="")
    x1, x2, not_satisfied = remove_outliers(x1, x2, f0, 2)
    colors = colors[~not_satisfied]
    print(x1.shape[0])

    # 基礎行列Fの計算
    F = calc_fundamental_matrix_extended_fns_method(x1, x2, f0)

    # 対応点の補正
    x1, x2 = optimize_corresponding_points(F, x1, x2, f0)

    # 焦点距離f, f_primeの計算
    f, f_prime = calc_free_focal_length(F, f0)

    # 運動パラメータの計算
    R, t = calc_motion_parameters(F, x1, x2, f, f_prime, f0)

    # カメラ行列の取得
    P, P_prime = calc_camera_matrix(f, f_prime, R, t, f0)

    # 3次元復元
    X_ = reconstruct_3d_points(x1, x2, P, P_prime, f0)

    # 鏡像の場合符号を反転して修正する
    if detect_mirror_image(X_):
        X_ *= -1
        t *= -1

    # 2次元に射影したデータ点の表示
    # camera1で射影した2次元データ点のプロット
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_xlim(-f0 / 2, f0 / 2)
    ax1.set_ylim(-f0 / 2, f0 / 2)
    plt.grid()
    plot_2d_points(x1, ax1, color=colors)

    # camera2で射影した2次元データ点のプロット
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_xlim(-f0 / 2, f0 / 2)
    ax2.set_ylim(-f0 / 2, f0 / 2)
    plt.grid()
    plot_2d_points(x2, ax2, color=colors)

    plt.show()

    # 復元したデータ点の表示
    ax = init_3d_ax()
    plot_3d_points(X_, ax, color=colors)
    plot_3d_basis(np.eye(3), np.zeros(3), ax, label="Camera1")
    plot_3d_basis(R, t, ax, label="Camera2")

    plt.show()


if __name__ == "__main__":
    main()
