import cv2
import matplotlib.pyplot as plt
import numpy as np

from lib.epipolar_geometry import (
    calc_focal_length,
    calc_fundamental_matrix_8points_method,
    calc_motion_parameters,
    calc_camera_matrix,
    reconstruct_3d_points,
    calc_epipole,
)
from lib.utils import unit_vec
from lib.visualization import init_3d_ax, plot_3d_basis, plot_3d_points, plot_2d_points


class Camera:
    def __init__(self, origin=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), f=1.0):
        origin = np.asarray(origin)
        target = np.asarray(target)
        self.o = origin
        self.d = unit_vec(target - origin)
        self.f = f

    def get_camera_matrix(self, f0):
        return self._calc_camera_matrix(f0)

    def get_pose(self):
        return self._calc_pose()

    def _calc_camera_matrix(self, f0):
        R, t = self._calc_pose()
        return np.diag((self.f, self.f, f0)) @ np.hstack([R.T, -R.T @ t[:, np.newaxis]])

    def _calc_pose(self):
        world_top = np.array([1.0, 0.0, 0.0])
        camera_z = self.d
        camera_y = np.cross(camera_z, world_top)  # camera right
        camera_x = np.cross(camera_y, camera_z)  # camera up
        R = np.vstack((camera_x, camera_y, camera_z)).T
        t = self.o
        return R, t

    def project_points(self, X, f0):
        X_ext = np.hstack((X, np.ones((X.shape[0], 1))))
        # print(self.get_camera_matrix(f0))
        Xproj = X_ext @ self.get_camera_matrix(f0).T
        return Xproj[:, :2] / Xproj[:, -1:]


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
    return points + np.array([0, 3, 0])


def main():
    f_ = 1.5
    f_prime_ = 1.0
    f0 = 1.0
    camera1 = Camera([0, 1, 0], [0, 1, 3], f_)
    camera2 = Camera([2, 2, 1.1], [1, 0, 3], f_prime_)

    # データ点の設定
    X = set_points()

    # 2次元画像平面へ射影
    x1 = camera1.project_points(X, f0)
    x2 = camera2.project_points(X, f0)
    x1 += 0.02 * np.random.rand(*x1.shape)
    x2 += 0.02 * np.random.rand(*x2.shape)

    R1, t1 = camera1.get_pose()
    R2, t2 = camera2.get_pose()

    # 基礎行列の計算
    F = calc_fundamental_matrix_8points_method(x1, x2, normalize=True)
    # F = np.cross(t2, R2.T).T
    print(f"F={F}")

    # エピポールの計算
    epipole = calc_epipole(F)
    print(f"e1={epipole[0]}, e2={epipole[1]}")

    # 焦点距離f, f_primeの計算
    f, f_prime = calc_focal_length(F, f0)
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
