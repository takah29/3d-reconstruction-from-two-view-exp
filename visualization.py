import matplotlib.pyplot as plt


def init_3d_ax():
    plt.figure(figsize=(16, 16))
    ax = plt.axes(projection="3d")

    ax.set_xlim3d([-5.0, 5.0])
    ax.set_xlabel("X")

    ax.set_ylim3d([-5.0, 5.0])
    ax.set_ylabel("Y")

    ax.set_zlim3d([-5.0, 5.0])
    ax.set_zlabel("Z")

    return ax


def plot_basis(pos, basis, ax, label=None):
    """
    基底をプロットする。回転行列のプロットにも使用できる。
    """
    assert pos.shape == (3,)
    assert basis.shape == (3, 3)

    cols = ["r", "g", "b", "r", "r", "g", "g", "b", "b"]
    _ = ax.quiver(
        [pos[0]] * 3, [pos[1]] * 3, [pos[2]] * 3, basis[:, 0], basis[:, 1], basis[:, 2], colors=cols
    )

    if label is not None:
        ax.text(pos[0], pos[1], pos[2] + 1.0, label)


def plot_points(points, ax, color="black"):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker="o")


if __name__ == "__main__":
    import numpy as np

    ax = init_3d_ax()

    # 基底のプロット
    pos = np.array([1, 0, 0])
    omega = 1.0
    basis = np.array(
        [[1, 0, 0], [0, np.cos(omega), -np.sin(omega)], [0, np.sin(omega), np.cos(omega)]]
    ).T  # 横ベクトル向けに転置する
    plot_basis(pos, basis, ax, label="test")

    # データ点のプロット
    X = np.eye(3) @ basis + pos
    plot_points(X, ax, "blue")
    plt.show()
