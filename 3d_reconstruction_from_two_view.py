import numpy as np
import cv2

from lib.camera import Camera
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
from lib.visualization import ThreeDimensionalPlotter, TwoDimensionalMatrixPlotter


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

    # 復元したシーンデータの表示
    plotter_3d = ThreeDimensionalPlotter(figsize=(10, 10))
    plotter_3d.set_lim()
    plotter_3d.plot_points(X_, color=colors)
    plotter_3d.plot_basis(np.eye(3), np.zeros(3), label="Camera1")
    plotter_3d.plot_basis(R, t, label="Camera2")
    plotter_3d.show()
    plotter_3d.close()

    # 投影データと復元後の再投影データの表示
    cameras_ = []
    for R_pred, t_pred, f_pred in [(np.eye(3), np.zeros(3), f), (R, t, f_prime)]:
        K_pred = np.diag([f_pred, f_pred, f0])
        cameras_.append(Camera(R_pred, t_pred, K_pred))

    x_list_ = []
    for camera in cameras_:
        x = camera.project_points(X_, method="perspective")
        x_list_.append(x)

    plotter_2d = TwoDimensionalMatrixPlotter(1, 2, (10, 6))
    for i, x in enumerate([x1, x2]):
        plotter_2d.select(i)
        plotter_2d.set_property(f"Camera{i + 1}", (-1, 1), (-1, 1))
        plotter_2d.plot_points(x, color="green", label="Projection")
        plotter_2d.plot_points(x_list_[i], color="red", label="Reprojection")

    plotter_2d.show()
    plotter_2d.close()


if __name__ == "__main__":
    main()
