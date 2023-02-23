import matplotlib.pyplot as plt
import numpy as np

from lib.conversion import get_rotation_axis_and_angle, get_rotation_matrix
from lib.prediction import (
    correct_rotation_matrix,
    is_optimized_rotation_matrix,
    predict_motion,
    predict_motion_no_error,
)
from lib.utils import add_noise, sample_normal_dist, unit_vec
from lib.visualization import init_3d_ax, plot_basis, plot_points

np.random.seed(2)


def main():
    ax = init_3d_ax()

    t = np.array([0, 0, 2])
    A = get_rotation_matrix(np.array([1, 0, 0]), 1)
    X = sample_normal_dist(1.0, 10)

    X_ = X - X.mean(axis=0)
    r1 = unit_vec(X_[0])
    r2 = unit_vec(np.cross(X_[0], X_[1]))
    r3 = unit_vec(np.cross(X_[0], np.cross(X_[0], X_[1])))
    R = np.vstack((r1, r2, r3))

    Y = X_ @ A + (X.mean(axis=0) + t)
    Y = add_noise(Y, 0.1)

    plot_points(X, ax, "blue")
    plot_points(Y, ax, "red")

    t_, A_ = predict_motion(X, Y)
    if not is_optimized_rotation_matrix(A_):
        print("A_ is not a rotation matrix, so it is corrected")
        A_ = correct_rotation_matrix(A_)

    plot_basis(
        X.mean(axis=0),
        R,
        ax,
        "before",
    )
    plot_basis(X.mean(axis=0) + t_, R @ A_, ax, "after")

    plot_points(X_ @ A_ + (X.mean(axis=0) + t_), ax, "green")

    print(t, A, sep="\n")
    print(t_, A_, sep="\n")

    plt.show()


if __name__ == "__main__":
    main()
