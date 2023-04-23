import matplotlib.pyplot as plt
import numpy as np

from lib.conversion import get_rotation_matrix
from lib.prediction import (
    correct_rotation_matrix,
    is_optimized_rotation_matrix,
    predict_motion,
)
from lib.utils import add_noise, sample_normal_dist
from lib.visualization import init_3d_ax, plot_3d_basis, plot_3d_points

np.random.seed(2)


def main():
    # データ点Xの作成
    X = sample_normal_dist(1.0, 10)
    X = X - X.mean(axis=0)

    # データ点X1の作成
    R1 = get_rotation_matrix(np.array([0, 1, 0]), 1)
    t1 = np.array([1, 0, 0])
    X1 = X @ R1.T + t1

    # データ点X2の作成
    R2 = get_rotation_matrix(np.array([0, 0, 1]), 1)
    t2 = np.array([1, 0, 0])
    X2 = X @ (R1.T @ R2.T) + (t1 + t2)
    X2 = add_noise(X2, 0.1)

    # 回転R_と並進t_を推定
    R2_, t2_ = predict_motion(X1, X2)

    # 回転R_の最適補正
    if not is_optimized_rotation_matrix(R2_):
        print("A_ is not a rotation matrix, so it is corrected")
        R2_ = correct_rotation_matrix(R2_)

    # 推定したR_とt_でX1を変換したYを作成
    X2_ = X @ (R1.T @ R2_.T) + (t1 + t2_)

    # プロット
    ax = init_3d_ax()

    plot_3d_points(X1, ax, "blue")
    plot_3d_points(X2, ax, "red")
    plot_3d_points(X2_, ax, "green")

    plot_3d_basis(R1, t1, ax, "X1-pose")
    plot_3d_basis(R2 @ R1, t1 + t2, ax, "X2-pose")
    plot_3d_basis(R2_ @ R1, t1 + t2_, ax, "X2_-pose")

    plt.show()


if __name__ == "__main__":
    main()
