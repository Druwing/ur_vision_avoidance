"""Kinematic closed-loop check of the avoidance logic on the UR5e model.

The "plant" is an ideal joint integrator (q += qdot * dt): no controller
dynamics, no latency, no camera. This validates the avoidance logic and the
safety filter, NOT the Gazebo/ROS integration. Clearance is measured by this
file independently of the controller, on the TCP and arm sample points only.
"""

import unittest

import numpy as np

from _common import Q_HOME, true_clearance
from ur_vision_avoidance.avoidance_logic import (
    AvoidanceParams,
    ObstacleView,
    compute_step,
    pick_escape_direction,
)
from ur_vision_avoidance.ur_kinematics import UR5eKinematics

DIRECTION = np.array([0.0, -1.0, 0.0])  # reachable from Q_HOME (+y would leave the workspace)
DISTANCE = 0.45
DT = 0.1


def simulate(mode, scenario, controller_knows_obstacle=True, rl_offset=None, steps=500):
    kin = UR5eKinematics()
    p = AvoidanceParams()
    q = Q_HOME.copy()
    transform = kin.fk(q)
    start, r_ref = transform[:3, 3].copy(), transform[:3, :3].copy()
    goal = start + DISTANCE * DIRECTION
    along, offset, radius = scenario
    center = start + along * DIRECTION + np.asarray(offset, dtype=float)
    spheres = [(center, radius)]
    known = [ObstacleView(center, radius)] if controller_knows_obstacle else []

    escape, _ = pick_escape_direction(kin, q, DIRECTION)
    p.escape_direction = tuple(escape)

    out = {"min_clearance": np.inf, "max_qdot": 0.0, "max_speed": 0.0, "statuses": set()}
    for step in range(steps):
        result = compute_step(kin, q, goal, known, p, mode, rl_offset, r_ref)
        out["statuses"].add(result.status)
        out["max_qdot"] = max(out["max_qdot"], float(np.max(np.abs(result.qdot))))
        out["max_speed"] = max(out["max_speed"], float(np.linalg.norm(result.v_cmd)))
        q = q + result.qdot * DT
        out["min_clearance"] = min(out["min_clearance"], true_clearance(kin, q, spheres))
        if result.reached:
            break
    out.update(
        reached=result.reached,
        steps=step + 1,
        goal_error=float(np.linalg.norm(goal - kin.position(q))),
        orientation_error=float(np.linalg.norm(kin.fk(q)[:3, :3] - r_ref)),
    )
    return out


# (distance along the path, offset from the path in base_link, radius)
SCENARIOS = {
    "head_on": (0.25, (0.0, 0.0, 0.0), 0.06),
    "head_on_small_close": (0.20, (0.0, 0.0, 0.06), 0.04),
    "shifted_x": (0.25, (0.06, 0.0, 0.0), 0.06),
    "shifted_down": (0.25, (0.0, 0.0, -0.06), 0.06),
    "side_far": (0.20, (0.10, 0.0, 0.0), 0.06),
    "side_near": (0.20, (-0.10, 0.0, 0.0), 0.06),
}


class ClosedLoopTest(unittest.TestCase):
    def test_free_space_reaches_goal_and_holds_orientation(self):
        result = simulate("potential_field", (5.0, (0, 0, 0), 0.05))  # obstacle far away
        self.assertTrue(result["reached"])
        self.assertLess(result["orientation_error"], 0.05)

    def test_unaware_controller_collides_so_the_scenario_is_meaningful(self):
        result = simulate("nominal", SCENARIOS["head_on"], controller_knows_obstacle=False)
        self.assertLess(result["min_clearance"], 0.0)

    def test_potential_field_avoids_and_reaches_goal(self):
        for name, scenario in SCENARIOS.items():
            with self.subTest(scenario=name):
                result = simulate("potential_field", scenario)
                self.assertTrue(result["reached"], result)
                self.assertGreater(result["min_clearance"], 0.03, result)
                self.assertLess(result["orientation_error"], 0.05, result)
                self.assertLess(result["steps"], 400, result)

    def test_potential_field_is_faster_than_the_filter_alone_when_head_on(self):
        # The joint-space filter alone slides around the obstacle slowly; the
        # potential field is what makes the head-on detour efficient.
        fast = simulate("potential_field", SCENARIOS["head_on"])
        slow = simulate("nominal", SCENARIOS["head_on"])
        self.assertTrue(fast["reached"] and slow["reached"])
        self.assertLess(fast["steps"], 0.7 * slow["steps"])

    def test_limits_are_respected_in_every_step(self):
        p = AvoidanceParams()
        for name, scenario in SCENARIOS.items():
            with self.subTest(scenario=name):
                result = simulate("potential_field", scenario)
                self.assertLessEqual(result["max_qdot"], p.max_joint_vel + 1e-9)
                self.assertLessEqual(result["max_speed"], p.max_speed + 1e-9)

    def test_filter_alone_keeps_the_arm_safe_in_nominal_and_rl_zero_modes(self):
        # Without avoidance terms the task may not complete, but it must stay safe.
        for mode, offset in (("nominal", None), ("rl", np.zeros(3))):
            result = simulate(mode, SCENARIOS["head_on"], rl_offset=offset)
            with self.subTest(mode=mode):
                self.assertGreater(result["min_clearance"], 0.03, result)

    def test_rl_mode_without_policy_output_falls_back_and_completes(self):
        result = simulate("rl", SCENARIOS["head_on"], rl_offset=None)
        self.assertTrue(result["reached"], result)
        self.assertGreater(result["min_clearance"], 0.03, result)

    def test_adversarial_rl_offset_cannot_force_a_collision(self):
        # A policy that pushes straight at the obstacle must be stopped by the filter.
        toward = DIRECTION * 0.08
        result = simulate("rl", SCENARIOS["head_on"], rl_offset=toward)
        self.assertGreater(result["min_clearance"], 0.03, result)


if __name__ == "__main__":
    unittest.main()
