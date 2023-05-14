import matplotlib.pyplot as plt
import numpy as np

from lib.camera3 import Camera
from lib.epipolar_geometry import (
    calc_camera_matrix,
    calc_epipole,
    calc_fixed_focal_length,
    calc_free_focal_length,
    calc_motion_parameters,
    reconstruct_3d_points,
    detect_mirror_image,
)
from lib.fundamental_matrix import (
    calc_fundamental_matrix_8points_method,
    calc_fundamental_matrix_extended_fns_method,
    calc_fundamental_matrix_taubin_method,
    find_fundamental_matrix_cv,
    optimize_corresponding_points,
    remove_outliers,
)
from lib.visualization import ThreeDimensionalPlotter, TwoDimensionalMatrixPlotter
from lib.utils import unit_vec


def set_points():
    points = []
    for x in np.linspace(-1, 1, 10):
        for theta in np.linspace(-np.pi / 2, np.pi / 2, 20):
            r = 1 / (x + 2)
            y, z = r * np.cos(theta), r * np.sin(theta)
            points.append((x, y, z + 3))

    return np.vstack(points)


def set_points_box():
    points = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ]
    )
    return points + np.array([0, 0, 3])


def calc_true_F(R1, t1, R2, t2, f, f_prime, f0):
    return (
        np.diag((f0, f0, f))
        @ np.cross(t2 - t1, (R2 @ R1.T).T, axisc=0)
        @ np.diag((f0, f0, f_prime))
    )


def main():
    np.random.seed(123)

    f0 = 1.0
    f_ = 1.0
    f_prime_ = 1.0
    camera1 = Camera.create([1, 1, 0], [0, 0, 3], f_, f0)
    camera2 = Camera.create([2, 2, 1.1], [0.5, 0, 3], f_prime_, f0)

    # データ点の設定
    X = set_points()

    # 2次元画像平面へ射影
    x1 = camera1.project_points(X, method="perspective")
    x2 = camera2.project_points(X, method="perspective")

    # ノイズの追加
    # x1 += 0.01 * np.random.randn(*x1.shape)
    # x2 += 0.01 * np.random.randn(*x2.shape)

    # アウトライアの追加
    # x1 = np.vstack((x1, 0.5 * np.random.randn(20, 2)))
    # x2 = np.vstack((x2, 0.5 * np.random.randn(20, 2)))

    R1, t1 = camera1.get_pose()
    R2, t2 = camera2.get_pose()

    # アウトライアの除去
    x1, x2, outliers = remove_outliers(x1, x2, f0, 0.05)
    print(f"number of outlier: {outliers.sum()}")

    # 基礎行列の計算
    F1 = calc_fundamental_matrix_8points_method(x1, x2, f0, normalize=True, optimal=True)
    F2 = calc_fundamental_matrix_taubin_method(x1, x2, f0)
    F3 = calc_fundamental_matrix_extended_fns_method(x1, x2, f0)
    F4 = find_fundamental_matrix_cv(x1, x2)
    F_ = calc_true_F(R1, t1, R2, t2, f_, f_prime_, f0)
    F = F3
    print("8points:", F1)
    print("taubin:", (F1[0, 0] / F2[0, 0]) * F2)
    print("ext_fns:", (F1[0, 0] / F3[0, 0]) * F3)
    print("opencv:", (F1[0, 0] / F4[0, 0]) * F4)
    print("trueF:", (F1[0, 0] / F_[0, 0]) * F_)
    # print(f"|F_ - F|=\n{np.linalg.norm(F_ - F)}")

    # 対応点の最適補正
    x1, x2 = optimize_corresponding_points(F, x1, x2, f0)

    # エピポールの計算
    epipole = calc_epipole(F)[:, :2]
    print(f"e1={epipole[0]}, e2={epipole[1]}")

    # 焦点距離f, f_primeの計算
    f, f_prime = calc_free_focal_length(F, f0, verbose=False)
    # f = f_prime = calc_fixed_focal_length(F, f0)
    # f, f_prime = f_, f_prime_
    print(f"f={f}, f_prime={f_prime}")

    # 運動パラメータの計算
    R, t = calc_motion_parameters(F, x1, x2, f, f_prime, f0)
    # R, t = R2 @ R1.T, unit_vec(t2 - t1)
    print(f"R=\n{R}, \nt=\n{t}")

    # カメラ行列の取得
    P, P_prime = calc_camera_matrix(f, f_prime, R, t, f0)
    print(f"P=\n{P}, \nP_prime=\n{P_prime}")

    # 3次元復元
    X_ = reconstruct_3d_points(x1, x2, P, P_prime, f0)

    # 鏡像の場合符号を反転して修正する
    if detect_mirror_image(X_):
        X_ *= -1
        t *= -1

    # シーンデータの表示
    plotter_3d = ThreeDimensionalPlotter(figsize=(10, 10))
    plotter_3d.set_lim()
    plotter_3d.plot_points(X)
    for i, camera in enumerate([camera1, camera2], start=1):
        plotter_3d.plot_basis(*camera.get_pose(), label=f"Camera{i}")
    plotter_3d.show()
    plotter_3d.close()

    # 復元したシーンデータの表示
    plotter_3d = ThreeDimensionalPlotter(figsize=(10, 10))
    plotter_3d.set_lim()
    plotter_3d.plot_points(X_)
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
