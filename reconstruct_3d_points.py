import numpy as np
import cv2
import matplotlib.pyplot as plt

from lib.epipolar_geometry import (
    calc_camera_matrix,
    calc_fixed_focal_length,
    calc_free_focal_length,
    calc_motion_parameters,
    convert_image_coord_to_screen_coord,
    detect_corresponding_points,
    reconstruct_3d_points,
    detect_mirror,
)
from lib.fundamental_matrix import (
    calc_fundamental_matrix_8points_method,
    calc_fundamental_matrix_extended_fns_method,
    find_fundamental_matrix_cv,
    optimize_corresponding_points,
    remove_outliers,
)
from lib.visualization import init_3d_ax, plot_3d_basis, plot_3d_points


def main():
    # 画像ファイル読み込み
    img1 = cv2.imread("./images/001.jpg")
    img2 = cv2.imread("./images/002.jpg")

    # 対応点の検出
    x1, x2 = detect_corresponding_points(img1, img2)

    # 画像座標をスクリーン座標へ変換
    x1 = convert_image_coord_to_screen_coord(x1)
    x2 = convert_image_coord_to_screen_coord(x2)

    f0 = max(img1.shape)

    # アウトライアの除去
    print(f"remove outliers: {x1.shape[0]} -> ", end="")
    x1, x2, _ = remove_outliers(x1, x2, f0, 2)
    print(x1.shape[0])

    # 基礎行列Fの計算
    # F = calc_fundamental_matrix_8points_method(x1, x2, f0, normalize=True, optimal=True)
    F = calc_fundamental_matrix_extended_fns_method(x1, x2, f0)
    # F = find_fundamental_matrix_cv(x1, x2)

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
    if detect_mirror(X_):
        X_ *= -1
        t *= -1

    ax = init_3d_ax()
    plot_3d_points(X_, ax, "blue")
    plot_3d_basis(np.eye(3), np.zeros(3), ax, label="Camera1")
    plot_3d_basis(R, t, ax, label="Camera2")

    plt.show()


if __name__ == "__main__":
    main()
