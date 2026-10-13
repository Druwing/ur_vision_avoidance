"""Forward kinematics and Jacobian for the UR5e (standard DH), no ROS needed.

All poses are expressed in the ROS ``base_link`` frame. The UR DH base frame
is rotated by pi about Z relative to ``base_link`` (see ``ur_description``),
which is handled by ``base_yaw``.
"""

from __future__ import annotations

import numpy as np

UR5E_D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
UR5E_A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
UR5E_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
UR5E_Q_MIN = np.array([-2 * np.pi, -2 * np.pi, -np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi])
UR5E_Q_MAX = -UR5E_Q_MIN


def _dh(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


class UR5eKinematics:
    def __init__(self, base_yaw: float = np.pi) -> None:
        c, s = np.cos(base_yaw), np.sin(base_yaw)
        self.base = np.array(
            [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )

    @staticmethod
    def _check(q) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != 6:
            raise ValueError("expected 6 joint values")
        return q

    def frames(self, q) -> list[np.ndarray]:
        """Frames 0..6 (0 = base_link) as 4x4 matrices."""
        q = self._check(q)
        frames = [self.base.copy()]
        for i in range(6):
            frames.append(frames[-1] @ _dh(UR5E_A[i], UR5E_ALPHA[i], UR5E_D[i], q[i]))
        return frames

    def fk(self, q) -> np.ndarray:
        return self.frames(q)[-1]

    def position(self, q) -> np.ndarray:
        return self.fk(q)[:3, 3]

    def jacobian(self, q) -> np.ndarray:
        """Geometric Jacobian (linear on top, angular below) in base_link."""
        frames = self.frames(q)
        p_end = frames[-1][:3, 3]
        jac = np.zeros((6, 6))
        for i in range(6):
            z = frames[i][:3, 2]
            o = frames[i][:3, 3]
            jac[:3, i] = np.cross(z, p_end - o)
            jac[3:, i] = z
        return jac

    def link_points(self, q) -> np.ndarray:
        """Sample points along the arm (elbow, wrist, link mid-points)."""
        f = self.frames(q)
        o = [frame[:3, 3] for frame in f]
        return np.array(
            [
                0.5 * (o[1] + o[2]),
                o[2],
                0.5 * (o[2] + o[3]),
                o[3],
                o[4],
                o[5],
            ]
        )

    def arm_points_with_jacobians(self, q) -> list[tuple[np.ndarray, np.ndarray]]:
        """Control points on the arm with their 3x6 linear Jacobians.

        Same points (and order) as :meth:`link_points`, followed by the tool
        flange. A point attached after joint ``k`` only depends on joints 1..k.
        """
        frames = self.frames(q)
        o = [frame[:3, 3] for frame in frames]
        spec = [
            (0.5 * (o[1] + o[2]), 2),
            (o[2], 2),
            (0.5 * (o[2] + o[3]), 3),
            (o[3], 3),
            (o[4], 4),
            (o[5], 5),
            (o[6], 6),
        ]
        out = []
        for point, last_joint in spec:
            jac = np.zeros((3, 6))
            for j in range(last_joint):
                jac[:, j] = np.cross(frames[j][:3, 2], point - frames[j][:3, 3])
            out.append((point, jac))
        return out

    @staticmethod
    def manipulability(jac: np.ndarray) -> float:
        return float(np.sqrt(abs(np.linalg.det(jac @ jac.T))))

    @staticmethod
    def dls_solve(jac: np.ndarray, twist, damping: float = 0.05) -> np.ndarray:
        """Damped least squares: qdot = J^T (J J^T + lambda^2 I)^-1 twist."""
        twist = np.asarray(twist, dtype=float).reshape(6)
        regularized = jac @ jac.T + (damping**2) * np.eye(6)
        return jac.T @ np.linalg.solve(regularized, twist)
