import unittest

import numpy as np

from _common import CONFIG, ensure_gymnasium

USING_STUB = ensure_gymnasium()

from ur_vision_avoidance.fixed_scene_env import FixedSceneAvoidanceEnv  # noqa: E402
from ur_vision_avoidance.rl_contract import (  # noqa: E402
    ACT_DIM,
    OBS_DIM,
    action_scale,
    build_observation,
    task_rotation,
)
from ur_vision_avoidance.scene_config import DEFAULT_SCENE, load_scene  # noqa: E402


class ContractTest(unittest.TestCase):
    def test_layout(self):
        vector = build_observation(
            [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14, 15], 0.5
        )
        self.assertEqual(vector.shape, (OBS_DIM,))
        self.assertEqual(vector.dtype, np.float32)
        np.testing.assert_array_equal(vector, np.arange(1, 16).tolist() + [0.5])

    def test_wrong_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            build_observation([1, 2], [0, 0, 0], [0, 0, 0], np.zeros(6), 1.0)
        with self.assertRaises(ValueError):
            build_observation([0, 0, 0], [0, 0, 0], [0, 0, 0], np.zeros(5), 1.0)

    def test_action_scale(self):
        np.testing.assert_allclose(action_scale(0.1, 0.3), [0.1] * 3 + [0.3] * 3)
        self.assertEqual(ACT_DIM, 6)

    def test_task_rotation_is_a_proper_rotation_with_x_along_the_path(self):
        for direction in ([1, 0, 0], [0, 1, 0], [0, -1, 0], [1, 1, 0.3], [0, 0, 1]):
            rotation = task_rotation(direction)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)
            unit = np.asarray(direction, float) / np.linalg.norm(direction)
            np.testing.assert_allclose(rotation[:, 0], unit, atol=1e-9)

    def test_task_rotation_is_identity_for_the_training_direction(self):
        np.testing.assert_allclose(task_rotation([1, 0, 0]), np.eye(3), atol=1e-12)

    def test_task_frame_makes_observation_independent_of_path_direction(self):
        """Same scene rotated in base_link gives the same observation in the task frame."""
        goal_rel_base = np.array([0.0, -0.4, 0.0])
        obstacle_rel_base = np.array([0.0, -0.2, 0.01])
        rotation = task_rotation([0.0, -1.0, 0.0])
        goal_task = rotation.T @ goal_rel_base
        obstacle_task = rotation.T @ obstacle_rel_base
        np.testing.assert_allclose(goal_task, [0.4, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(obstacle_task, [0.2, 0.0, 0.01], atol=1e-9)


class SceneTest(unittest.TestCase):
    def test_fixed_scene_in_yaml_reproduces_the_original_defaults(self):
        scene = load_scene(CONFIG, "fixed_scene")
        for field in (
            "goal_offset",
            "obstacle_offset",
            "obstacle_radius",
            "collision_distance",
            "goal_tolerance",
            "nominal_speed",
            "max_offset_speed",
            "max_angular",
            "dt",
            "max_episode_steps",
        ):
            np.testing.assert_allclose(
                getattr(scene, field), getattr(DEFAULT_SCENE, field), atol=1e-9, err_msg=field
            )
        np.testing.assert_allclose(scene.goal, [1.2, 0.0, 0.45])
        np.testing.assert_allclose(scene.obstacle, [0.65, 0.0, 0.45])

    def test_ur5e_scene_fits_the_robot_workspace(self):
        scene = load_scene(CONFIG, "ur5e_scene")
        self.assertLess(np.linalg.norm(scene.goal_offset), 0.85)  # UR5e reach
        self.assertGreater(
            np.linalg.norm(scene.obstacle_offset), scene.collision_distance
        )  # does not start in collision
        self.assertLess(
            np.linalg.norm(np.asarray(scene.goal_offset) - np.asarray(scene.obstacle_offset)),
            np.linalg.norm(scene.goal_offset),
        )

    def test_unknown_section_is_an_error(self):
        with self.assertRaises(KeyError):
            load_scene(CONFIG, "nope")


class EnvTest(unittest.TestCase):
    def test_observation_comes_from_the_shared_builder(self):
        env = FixedSceneAvoidanceEnv(seed=1)
        observation, info = env.reset()
        expected = build_observation(
            env.goal - env.position,
            env.obstacle - env.position,
            np.zeros(3),
            np.zeros(6),
            1.0,
        )
        np.testing.assert_array_equal(observation, expected)
        self.assertFalse(info["ard_enabled"])

    def test_default_scene_is_unchanged_by_the_refactor(self):
        env = FixedSceneAvoidanceEnv()
        np.testing.assert_allclose(env.goal, [1.2, 0.0, 0.45])
        np.testing.assert_allclose(env.obstacle, [0.65, 0.0, 0.45])
        self.assertEqual(env.max_episode_steps, 160)

    def _rollout(self, env, policy):
        observation, _ = env.reset(seed=0)  # deterministic obstacle jitter
        info = {}
        for _ in range(env.max_episode_steps):
            observation, _, terminated, truncated, info = env.step(policy(observation))
            if terminated or truncated:
                break
        return info

    def test_straight_line_collides_and_a_sidestep_policy_succeeds(self):
        for section in ("fixed_scene", "ur5e_scene"):
            scene = load_scene(CONFIG, section)
            env = FixedSceneAvoidanceEnv(scene=scene)
            straight = self._rollout(env, lambda obs: np.zeros(6, dtype=np.float32))
            self.assertTrue(straight["collision"], section)

            target = 1.6 * scene.collision_distance  # lateral offset needed to pass

            # fixed_scene allows only ~10 steps of slack over the nominal 1.2 m
            # path, so a detour needs extra forward speed to finish in time.
            nominal_time = np.linalg.norm(scene.goal_offset) / scene.nominal_speed
            speed_up = 1.0 if nominal_time > 0.8 * scene.episode_duration_s else 0.0

            # "passed" threshold: clear of the obstacle, but before the goal
            gap = float(np.linalg.norm(scene.goal_offset) - np.linalg.norm(scene.obstacle_offset))
            margin = min(1.2 * scene.collision_distance, 0.5 * gap)

            def sidestep(obs, target=target, margin=margin, speed_up=speed_up):
                action = np.zeros(6, dtype=np.float32)
                action[0] = speed_up
                if obs[3] > -margin:  # not yet clear of the obstacle: go around it
                    lateral = -obs[4]  # TCP y relative to the obstacle
                    # hold the lateral offset (nominal motion keeps pulling back)
                    action[1] = float(np.clip(10.0 * (target - lateral), -1.0, 1.0))
                else:  # passed it: return to the goal line
                    action[1] = float(np.clip(10.0 * obs[1], -1.0, 1.0))
                return action

            if section == "ur5e_scene":
                # The obstacle sits only 0.20 m before the goal and the collision
                # distance is 0.18 m (aligned with the safety filter), so a
                # sideways detour cannot return in time: climb, pass, descend.
                result = self._rollout(env, lambda obs, scene=scene: self._pass_over(obs, scene))
            else:
                result = self._rollout(env, sidestep)
            self.assertFalse(result["collision"], section)
            self.assertTrue(result["reached_goal"], section)

    @staticmethod
    def _pass_over(obs, scene, k=1.25):
        radius = k * scene.collision_distance
        goal, dx, height = obs[0:3], -float(obs[3]), -float(obs[5])
        if dx < 0:
            want = radius if dx > -radius else 0.0
        else:
            want = float(np.sqrt(max(radius * radius - dx * dx, 0.0)))
        nominal = scene.nominal_speed * goal / max(float(np.linalg.norm(goal)), 1e-6)
        velocity = np.zeros(3)
        velocity[2] = np.clip(2.0 * (want - height), -0.1, 0.1)
        climbing = -radius < dx < 0 and height < want - 0.01
        velocity[0] = 0.0 if climbing else scene.nominal_speed
        if not climbing and dx > radius:
            velocity = nominal
        action = np.zeros(6, dtype=np.float32)
        action[:3] = np.clip((velocity - nominal) / scene.max_offset_speed, -1.0, 1.0)
        return action

    def test_obstacle_jitter_is_seeded_bounded_and_off_for_the_fixed_scene(self):
        scene = load_scene(CONFIG, "ur5e_scene")
        env = FixedSceneAvoidanceEnv(scene=scene)
        nominal = scene.obstacle
        seen = []
        for seed in range(20):
            _, info = env.reset(seed=seed)
            offset = np.asarray(info["obstacle"]) - nominal
            self.assertTrue(np.all(offset >= np.asarray(scene.obstacle_jitter_min) - 1e-6))
            self.assertTrue(np.all(offset <= np.asarray(scene.obstacle_jitter_max) + 1e-6))
            seen.append(tuple(np.round(offset, 6)))
        self.assertGreater(len(set(seen)), 15)
        _, a = env.reset(seed=3)
        _, b = env.reset(seed=3)
        self.assertEqual(a["obstacle"], b["obstacle"])

        fixed = FixedSceneAvoidanceEnv(scene=load_scene(CONFIG, "fixed_scene"))
        for seed in range(3):
            _, info = fixed.reset(seed=seed)
            np.testing.assert_allclose(info["obstacle"], [0.65, 0.0, 0.45])

    def test_shaped_reward_makes_standing_still_worse_than_progress(self):
        env = FixedSceneAvoidanceEnv(scene=load_scene(CONFIG, "ur5e_scene"))
        env.reset(seed=0)
        _, still, *_ = env.step(-np.eye(6, dtype=np.float32)[0] * (env.nominal_speed / env.max_offset_speed))
        env.reset(seed=0)
        _, forward, *_ = env.step(np.zeros(6, dtype=np.float32))
        self.assertLess(still, forward)


if __name__ == "__main__":
    unittest.main()
