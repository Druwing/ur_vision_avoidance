import unittest

import numpy as np

from _common import Q_HOME
from ur_vision_avoidance.avoidance_logic import (
    AvoidanceParams,
    ObstacleView,
    clearance_filter,
    compute_step,
    nominal_velocity,
    pick_escape_direction,
    repulsive_velocity,
    whole_body_filter,
)
from ur_vision_avoidance.ur_kinematics import UR5eKinematics


def params(**kw):
    return AvoidanceParams(**kw)


class PrimitivesTest(unittest.TestCase):
    def test_nominal_velocity_points_to_goal_and_stops_at_tolerance(self):
        v, reached = nominal_velocity(np.zeros(3), np.array([0.4, 0.0, 0.0]), 0.05, 0.03)
        self.assertFalse(reached)
        np.testing.assert_allclose(v, [0.05, 0.0, 0.0], atol=1e-9)
        v, reached = nominal_velocity(np.zeros(3), np.array([0.02, 0.0, 0.0]), 0.05, 0.03)
        self.assertTrue(reached)
        np.testing.assert_allclose(v, 0.0)

    def test_clearance_filter_removes_only_the_inward_component(self):
        p = params()
        obstacle = ObstacleView(np.array([0.2, 0.0, 0.0]), radius=0.05)
        tcp = np.array([0.0, 0.0, 0.0])  # clearance 0.15: inside slow zone, outside stop zone
        v = np.array([0.08, 0.04, 0.0])  # toward obstacle (+x) and sideways (+y)
        out = clearance_filter(v, tcp, obstacle and [obstacle], p)
        self.assertLess(out[0], v[0])  # approach speed reduced
        self.assertAlmostEqual(out[1], v[1], places=9)  # tangential motion untouched

    def test_clearance_filter_ignores_far_obstacles_and_retreats_inside_stop_zone(self):
        p = params()
        far = ObstacleView(np.array([1.0, 0.0, 0.0]), 0.05)
        v = np.array([0.05, 0.0, 0.0])
        np.testing.assert_allclose(clearance_filter(v, np.zeros(3), [far], p), v)
        near = ObstacleView(np.array([0.12, 0.0, 0.0]), 0.05)  # clearance 0.07 < stop 0.10
        out = clearance_filter(v, np.zeros(3), [near], p)
        self.assertLess(out[0], 0.0)  # pushed away

    def test_head_on_repulsion_has_a_sideways_component(self):
        p = params(escape_direction=(0.0, 0.0, 1.0))
        obstacle = ObstacleView(np.array([0.25, 0.0, 0.0]), 0.06)
        tcp = np.array([0.0, 0.0, 0.0])
        rep = repulsive_velocity(tcp, np.array([1.0, 0.0, 0.0]), [obstacle], p)
        self.assertLess(rep[0], 0.0)  # away
        self.assertGreater(abs(rep[2]), 1e-3)  # and sideways, so it does not deadlock

    def test_repulsion_fades_near_the_goal(self):
        p = params()
        obstacle = ObstacleView(np.array([0.2, 0.0, 0.0]), 0.05)
        args = (np.zeros(3), np.array([1.0, 0.0, 0.0]), [obstacle], p)
        far = np.linalg.norm(repulsive_velocity(*args, goal_distance=1.0))
        near = np.linalg.norm(repulsive_velocity(*args, goal_distance=0.03))
        self.assertGreater(far, 0.0)
        self.assertLess(near, 0.25 * far)

    def test_whole_body_filter_limits_approach_speed_of_every_point(self):
        kin = UR5eKinematics()
        p = params()
        tcp = kin.position(Q_HOME)
        obstacle = ObstacleView(tcp + np.array([0.0, -0.17, 0.0]), 0.05)  # clearance 0.12
        qdot = kin.dls_solve(kin.jacobian(Q_HOME), [0.0, -0.1, 0.0, 0, 0, 0], 0.05)  # straight at it
        data = kin.arm_points_with_jacobians(Q_HOME)
        filtered = whole_body_filter(qdot, data, [obstacle], p)
        for point, jac in data:
            distance = np.linalg.norm(point - obstacle.center) - obstacle.radius
            if distance >= p.slow_clearance_m:
                continue
            normal = (point - obstacle.center) / np.linalg.norm(point - obstacle.center)
            inward = -float(normal @ (jac @ filtered))
            if distance >= p.stop_clearance_m:
                allowed = p.max_speed * (distance - p.stop_clearance_m) / (
                    p.slow_clearance_m - p.stop_clearance_m
                )
            else:
                allowed = -p.offset_max * min(1.0, (p.stop_clearance_m - distance) / p.stop_clearance_m)
            self.assertLessEqual(inward, allowed + 1e-6)

    def test_pick_escape_direction_is_a_unit_vector_perpendicular_to_the_path(self):
        kin = UR5eKinematics()
        path = np.array([0.0, -1.0, 0.0])
        direction, scores = pick_escape_direction(kin, Q_HOME, path)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=9)
        self.assertAlmostEqual(float(direction @ path), 0.0, places=9)
        self.assertEqual(set(scores), {"up", "down", "left", "right"})


class ComputeStepTest(unittest.TestCase):
    def setUp(self):
        self.kin = UR5eKinematics()
        self.p = params()
        transform = self.kin.fk(Q_HOME)
        self.start = transform[:3, 3]
        self.r_ref = transform[:3, :3]
        self.goal = self.start + np.array([0.0, -0.3, 0.0])

    def test_free_space_moves_toward_goal(self):
        result = compute_step(self.kin, Q_HOME, self.goal, [], self.p, "nominal", None, self.r_ref)
        self.assertEqual(result.status, "ok")
        np.testing.assert_allclose(result.v_cmd, [0.0, -0.05, 0.0], atol=2e-3)

    def test_goal_reached_holds(self):
        result = compute_step(self.kin, Q_HOME, self.start, [], self.p, "nominal", None, self.r_ref)
        self.assertTrue(result.reached)
        np.testing.assert_allclose(result.qdot, 0.0)

    def test_rl_mode_without_action_falls_back_to_potential_field(self):
        result = compute_step(self.kin, Q_HOME, self.goal, [], self.p, "rl", None, self.r_ref)
        self.assertEqual(result.mode_used, "potential_field")

    def test_rl_offset_is_added_and_bounded(self):
        base = compute_step(self.kin, Q_HOME, self.goal, [], self.p, "rl", np.zeros(3), self.r_ref)
        offset = np.array([0.0, 0.0, 10.0])  # absurdly large; must be clipped to offset_max
        out = compute_step(self.kin, Q_HOME, self.goal, [], self.p, "rl", offset, self.r_ref)
        self.assertEqual(out.mode_used, "rl")
        self.assertGreater(out.v_cmd[2], base.v_cmd[2])
        self.assertLessEqual(float(np.linalg.norm(out.v_cmd)), self.p.max_speed + 1e-6)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            compute_step(self.kin, Q_HOME, self.goal, [], self.p, "teleport", None, self.r_ref)

    def test_joint_speed_is_always_bounded(self):
        p = params(max_joint_vel=0.05)
        result = compute_step(self.kin, Q_HOME, self.goal, [], p, "nominal", None, self.r_ref)
        self.assertLessEqual(float(np.max(np.abs(result.qdot))), 0.05 + 1e-9)

    def test_holds_when_an_arm_point_is_inside_the_backstop(self):
        # Obstacle right on the forearm: the backstop must stop the arm.
        elbow = self.kin.link_points(Q_HOME)[1]
        obstacle = ObstacleView(elbow + np.array([0.0, 0.0, 0.02]), 0.03)
        result = compute_step(
            self.kin, Q_HOME, self.goal, [obstacle], self.p, "potential_field", None, self.r_ref
        )
        self.assertEqual(result.status, "link_clearance")
        np.testing.assert_allclose(result.qdot, 0.0)


if __name__ == "__main__":
    unittest.main()
