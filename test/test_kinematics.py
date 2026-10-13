import unittest

import numpy as np

from _common import Q_HOME  # noqa: F401  (also sets sys.path)
from ur_vision_avoidance.ur_kinematics import UR5eKinematics


class KinematicsTest(unittest.TestCase):
    def test_zero_pose_matches_ur5e_reference(self):
        # Published UR5e zero-pose TCP in the UR base frame.
        position = UR5eKinematics(base_yaw=0.0).position(np.zeros(6))
        np.testing.assert_allclose(position, [-0.8172, -0.2329, 0.0628], atol=1e-4)

    def test_base_yaw_rotates_about_z(self):
        q = Q_HOME
        p0 = UR5eKinematics(base_yaw=0.0).position(q)
        p_pi = UR5eKinematics(base_yaw=np.pi).position(q)
        np.testing.assert_allclose(p_pi, [-p0[0], -p0[1], p0[2]], atol=1e-9)

    def test_jacobian_matches_numeric_derivative(self):
        kin = UR5eKinematics()
        q = Q_HOME + np.array([0.1, 0.05, -0.1, 0.2, 0.1, -0.3])
        jac = kin.jacobian(q)
        eps = 1e-6
        numeric = np.zeros((3, 6))
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = eps
            numeric[:, i] = (kin.position(q + dq) - kin.position(q - dq)) / (2 * eps)
        np.testing.assert_allclose(jac[:3], numeric, atol=1e-7)

    def test_point_jacobians_match_numeric_derivative(self):
        kin = UR5eKinematics()
        q = Q_HOME + np.array([0.2, -0.1, 0.1, 0.0, 0.2, 0.1])
        data = kin.arm_points_with_jacobians(q)
        self.assertEqual(len(data), 7)
        eps = 1e-6
        for index, (point, jac) in enumerate(data):
            numeric = np.zeros((3, 6))
            for i in range(6):
                dq = np.zeros(6)
                dq[i] = eps
                plus = self._points(kin, q + dq)[index]
                minus = self._points(kin, q - dq)[index]
                numeric[:, i] = (plus - minus) / (2 * eps)
            np.testing.assert_allclose(jac, numeric, atol=1e-6, err_msg=f"point {index}")

    @staticmethod
    def _points(kin, q):
        return np.vstack((kin.link_points(q), kin.position(q)))

    def test_dls_reproduces_twist_away_from_singularity(self):
        kin = UR5eKinematics()
        jac = kin.jacobian(Q_HOME)
        twist = np.array([0.02, -0.03, 0.01, 0.0, 0.0, 0.0])
        qdot = kin.dls_solve(jac, twist, damping=1e-4)
        np.testing.assert_allclose(jac @ qdot, twist, atol=1e-6)

    def test_rejects_wrong_joint_count(self):
        with self.assertRaises(ValueError):
            UR5eKinematics().fk([0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
