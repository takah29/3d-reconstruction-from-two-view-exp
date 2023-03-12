import matplotlib.pyplot as plt
import numpy as np

from lib.camera import Camera
from lib.epipolar_geometry import (
    calc_camera_matrix,
    calc_epipole,
    calc_fixed_focal_length,
    calc_free_focal_length,
    calc_motion_parameters,
    reconstruct_3d_points,
)
from lib.fundamental_matrix import (
    calc_fundamental_matrix_8points_method,
    calc_fundamental_matrix_extended_fns_method,
    calc_fundamental_matrix_taubin_method,
    optimize_corresponding_points,
    remove_outliers,
)
from lib.visualization import init_3d_ax, plot_2d_points, plot_3d_basis, plot_3d_points


def set_points():
    points = []
    for x in np.linspace(-1, 1, 10):
        for theta in np.linspace(-np.pi / 2, 0.0, 10):
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


def calc_true_F(R, t, f, f_prime, f0):
    return np.diag((f0, f0, f)) @ np.cross(t, R.T).T @ np.diag((f0, f0, f_prime))


def main():
    np.random.seed(17)

    f_ = 1.0
    f_prime_ = 1.0
    camera1 = Camera([0, 0, 0], [0, 0, 3], f_)
    camera2 = Camera([2, 2, 1.1], [0.5, 0, 3], f_prime_)

    # データ点の設定
    X = set_points()

    # 2次元画像平面へ射影
    x1 = camera1.project_points(X, 1.0)
    x2 = camera2.project_points(X, 1.0)

    # ノイズの追加
    x1 += 0.01 * np.random.randn(*x1.shape)
    x2 += 0.01 * np.random.randn(*x2.shape)

    # アウトライアの追加
    # x1 = np.vstack((x1, 0.5 * np.random.randn(20, 2)))
    # x2 = np.vstack((x2, 0.5 * np.random.randn(20, 2)))

    R1, t1 = camera1.get_pose()
    R2, t2 = camera2.get_pose()

    f0 = 1.0

    # アウトライアの除去
    x1, x2, outliers = remove_outliers(x1, x2, f0, 0.05)
    print(f"number of outlier: {outliers.sum()}")

    # 基礎行列の計算
    #F = calc_fundamental_matrix_8points_method(x1, x2, f0, normalize=True, optimal=True)
    # F = calc_fundamental_matrix_taubin_method(x1, x2, f0)
    F = calc_fundamental_matrix_extended_fns_method(x1, x2, f0)
    F_ = calc_true_F(R2, t2, f_, f_prime_, f0)
    print(f"|F_ - F|={np.linalg.norm(F_ - F)}")

    # 対応点の最適補正
    #x1, x2 = optimize_corresponding_points(F, x1, x2, f0)

    # エピポールの計算
    epipole = calc_epipole(F)
    print(f"e1={epipole[0]}, e2={epipole[1]}")

    # 焦点距離f, f_primeの計算
    f, f_prime = calc_free_focal_length(F, f0, verbose=True)
    # f = f_prime = calc_fixed_focal_length(F, f0)
    # f, f_prime = f_, f_prime_
    print(f"f={f}, f_prime={f_prime}")

    # 運動パラメータの計算
    R, t = calc_motion_parameters(F, x1, x2, f, f_prime, f0)
    # R, t = R2, t2
    print(f"R={R}, t={t}")

    # 対応点の補正
    # X1_, X2_ = optimize_matches(X1, X2, F)

    # カメラ行列の取得
    P, P_prime = calc_camera_matrix(f, f_prime, R, t, f0)
    print(f"P={P}, P_prime={P_prime}")

    # 3次元復元
    X_ = reconstruct_3d_points(x1, x2, P, P_prime, f0)

    # 3次元点の表示
    ax = init_3d_ax()
    plot_3d_points(X, ax)
    plot_3d_basis(t1, R1.T, ax)
    plot_3d_basis(t2, R2.T, ax)
    # 3次元データ点の表示
    plt.show()
    plt.clf()

    # 2次元に射影したデータ点の表示
    # camera1で射影した2次元データ点のプロット
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    plt.grid()
    plot_2d_points(x1, ax1, color="black")
    # エピポールのプロット
    plot_2d_points(epipole[:1], ax1, color="green")

    # camera2で射影した2次元データ点のプロット
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    plt.grid()
    plot_2d_points(x2, ax2, color="black")
    # エピポールのプロット
    plot_2d_points(epipole[-1:], ax2, color="green")

    plt.show()

    # 復元したデータ点の表示
    # print(X_)
    ax = init_3d_ax()
    plot_3d_points(X_, ax)
    plot_3d_basis(t, R.T, ax)
    # plot_3d_basis(t2, R2, ax)
    # plot_3d_basis(t1, R1, ax)
    plt.show()


if __name__ == "__main__":
    main()
