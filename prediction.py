import numpy as np
from utils import unit_vec


def predict_motion_no_error(X, Y):
    """誤差がないデータから、並進tと回転Rを推定する"""
    xc = np.mean(X, axis=0)
    yc = np.mean(Y, axis=0)
    t = yc - xc

    A = X - xc
    r1 = unit_vec(A[0])
    r2 = unit_vec(np.cross(A[0], A[1]))
    r3 = unit_vec(np.cross(A[0], np.cross(A[0], A[1])))
    R1 = np.vstack((r1, r2, r3))

    B = Y - yc
    r1_ = unit_vec(B[0])
    r2_ = unit_vec(np.cross(B[0], B[1]))
    r3_ = unit_vec(np.cross(B[0], np.cross(B[0], B[1])))

    R2 = np.vstack((r1_, r2_, r3_))

    R = R1.T @ R2

    return t, R


def predict_motion(X, Y):
    """誤差があるデータから、並進tと回転Rを推定する"""
    xc = np.mean(X, axis=0)
    yc = np.mean(Y, axis=0)
    t = yc - xc

    A = X - xc
    B = Y - yc

    N = B.T @ A
    U, S, Vt = np.linalg.svd(N)
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U)]) @ U.T

    return t, R


def is_optimized_rotation_matrix(R):
    """回転行列の最適性を判定する"""
    return np.linalg.det(R) - 1 < 1e-8


def correct_rotation_matrix(R):
    """回転行列の最適補正を行う"""
    R = R.T
    U, S, Vt = np.linalg.svd(R)
    R_opt = U @ np.diag(1, 1, np.linalg.det(U @ Vt.T)) @ Vt

    return R_opt
