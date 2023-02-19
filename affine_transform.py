import numpy as np


class AffineTransform:
    """3次元アフィン変換クラス（横ベクトル）"""

    def __init__(self, M=None):
        if M is not None:
            assert M.shape == (4, 4)
            self.M = M
        else:
            self.M = np.eye(4)

    def to_array(self):
        """ndarrayに変換する"""
        return self.M

    def apply(self, X):
        """データ行列Xに対して変換を適用する"""
        assert X.shape[1] == 3

        X_ = np.hstack((X, np.ones((X.shape[0], 1))))

        return (X_ @ self.M)[:, :3]

    def trans(self, t):
        """並進tを適用する"""
        t = np.asarray(t)
        assert t.shape == (3,)

        M = np.block([[np.eye(3), np.zeros((3, 1))], [t, 1]])

        return AffineTransform(self.M @ M)

    def rot_x(self, theta):
        """x軸theta回転を適用する"""
        assert isinstance(theta, (float, int))

        R = np.array(
            [[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]]
        )
        M = np.block([[R, np.zeros((1, 3))], [np.zeros(3), 1.0]])

        return AffineTransform(self.M @ M)

    def rot_y(self, theta):
        """y軸theta回転を適用する"""
        assert isinstance(theta, (float, int))

        R = np.array(
            [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
        )
        M = np.block([[R, np.zeros((1, 3))], [np.zeros(3), 1.0]])

        return AffineTransform(self.M @ M)

    def rot_z(self, theta):
        """z軸theta回転を適用する"""
        assert isinstance(theta, (float, int))

        R = np.array(
            [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )
        M = np.block([[R, np.zeros((1, 3))], [np.zeros(3), 1.0]])

        return AffineTransform(self.M @ M)

    def ref_x(self):
        """x軸の鏡映を適用する"""
        M = np.eye(3)
        M[0, 0] = -1.0

        return AffineTransform(self.M @ M)

    def ref_y(self):
        """y軸の鏡映を適用する"""
        M = np.eye(3)
        M[1, 1] = -1.0

        return AffineTransform(self.M @ M)

    def ref_z(self):
        """z軸の鏡映を適用する"""
        M = np.eye(3)
        M[2, 2] = -1.0

        return AffineTransform(self.M @ M)

    def scale_x(self, scale):
        """x軸のスケール変換を適用する"""
        M = np.eye(3)
        M[0, 0] = scale

    def scale_y(self, scale):
        """y軸のスケール変換を適用する"""
        M = np.eye(3)
        M[1, 1] = scale

        return AffineTransform(self.M @ M)

    def scale_z(self, scale):
        """z軸のスケール変換を適用する"""
        M = np.eye(3)
        M[2, 2] = scale

        return AffineTransform(self.M @ M)

    def to_inv(self):
        """逆変換行列を取得する"""
        Ainv = np.linalg.inv(self.M[:2, :2])
        t = self.M[3, :3]
        M = np.block([[Ainv, -t @ Ainv], [np.zeros((1, 3)), 1]])

        return M
