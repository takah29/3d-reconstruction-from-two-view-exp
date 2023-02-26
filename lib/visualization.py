import matplotlib.pyplot as plt
from numpy.typing import NDArray


def init_3d_ax():
    """Zを奥行きとした右手系座標を設定する"""
    plt.figure(figsize=(16, 16))
    ax = plt.axes(projection="3d")

    ax.set_xlim3d([-5.0, 5.0])
    ax.set_xlabel("Y")

    ax.set_ylim3d([-5.0, 5.0])
    ax.set_ylabel("Z")

    ax.set_zlim3d([-5.0, 5.0])
    ax.set_zlabel("X")

    return ax



def plot_3d_basis(pos: NDArray, basis: NDArray, ax, label=None) -> None:
    """
    基底をプロットする。回転行列のプロットにも使用できる。
    """
    assert pos.shape == (3,)
    assert basis.shape == (3, 3)


    cols = ["r", "g", "b", "r", "r", "g", "g", "b", "b"]
    _ = ax.quiver(
        [pos[1]] * 3, [pos[2]] * 3, [pos[0]] * 3, basis[:, 1], basis[:, 2], basis[:, 0], colors=cols
    )

    if label is not None:
        ax.text(pos[0], pos[1], pos[2] + 1.0, label)


def plot_3d_points(X: NDArray, ax, color="black") -> None:
    ax.scatter(X[:, 1], X[:, 2], X[:, 0], c=color, marker="o")


def plot_2d_points(x, ax, color="black") -> None:
    ax.scatter(x[:, 1], x[:, 0], c=color, marker="o")

if __name__ == "__main__":
    import numpy as np

    ax = init_3d_ax()

    # 基底のプロット
    pos = np.array([1, 0, 0])
    omega = 1.0
    basis = np.array(
        [[1, 0, 0], [0, np.cos(omega), -np.sin(omega)], [0, np.sin(omega), np.cos(omega)]]
    ).T  # 横ベクトル向けに転置する
    plot_3dbasis(pos, basis, ax, label="test")

    # データ点のプロット
    X = np.eye(3) @ basis + pos
    plot_3d_points(X, ax, "blue")
    plt.show()
