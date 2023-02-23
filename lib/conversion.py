import numpy as np
from numpy.typing import NDArray
from .utils import unit_vec


def get_rotation_matrix(l_: NDArray, omega: float) -> NDArray:
    """任意軸l_に対する回転角omegaから回転行列を求める"""
    assert l_.shape == (3,)
    assert isinstance(omega, (int, float))

    R1 = (1 - np.cos(omega)) * np.ones((3, 3))
    R2 = l_[:, np.newaxis] @ l_[:, np.newaxis].T
    R3 = np.sin(omega) * np.ones((3, 3))
    R3[(0, 1, 2), (0, 1, 2)] = np.cos(omega)
    R4 = np.array([[1, -l_[2], l_[1]], [l_[2], 1, -l_[0]], [-l_[1], l_[0], 1]])
    R = R1 * R2 + R3 * R4

    return R.T  # 横ベクトル向けに転置する


def get_rotation_axis_and_angle(R: NDArray) -> tuple[NDArray, float]:
    """回転行列から回転軸l_と回転角omegaを求める"""
    assert R.shape == (3, 3)

    R = R.T  # 横ベクトル向けに転置する
    res_omega = np.arccos((np.trace(R) - 1) / 2)
    tmp = np.array([R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0]])
    res_l = -unit_vec(tmp)

    return res_l, res_omega


def wedge(x: NDArray) -> NDArray:
    """リー代数の基底行列による線形結合に変換する

    基底行列
    J0 = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
    J1 = [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]
    J2 = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]

    X = x[0] * J0 + x[1] * J1 + x[2] * J2

    ※ xは座標ベクトル（角速度ベクトルと等価）
    """
    assert x.shape == (3,)

    X = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    return X


def vee(X: NDArray) -> NDArray:
    """wadge関数の逆関数"""
    assert X.shape == (3, 3)

    x = np.array([X[2, 1], X[0, 2], X[1, 0]])

    return x


if __name__ == "__main__":
    l_ = np.array([1, 0, 0])
    omega = 1.0

    R = get_rotation_matrix(l_, omega)
    assert (
        R
        == np.array(
            [[1, 0, 0], [0, np.cos(omega), np.sin(omega)], [0, -np.sin(omega), np.cos(omega)]]
        )
    ).all()

    res_l, res_omega = get_rotation_axis_and_angle(R)
    assert (res_l == l_).all()
    assert np.abs(res_omega - omega) < 1e-8
