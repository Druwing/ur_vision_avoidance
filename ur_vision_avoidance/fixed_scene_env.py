"""Fixed-scene RL environment for the UR vision-avoidance starter stage.

This environment is deliberately deterministic apart from the optional seed. It
is a training scaffold, not a physics simulator and not a replacement for the
Gazebo/UR controller. ARD is intentionally absent from this stage.

The scene comes from ``config/rl_starter.yaml`` (``fixed_scene`` reproduces the
original 1.2 m scene; ``ur5e_scene`` is scaled to fit the UR5e workspace). The
observation is built by ``rl_contract.build_observation``, the same function the
ROS observation node uses, so training and deployment share one layout.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .rl_contract import ACT_DIM, OBS_DIM, build_observation
from .scene_config import DEFAULT_SCENE, SceneConfig


class FixedSceneAvoidanceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Learn a bounded Cartesian offset around one fixed spherical obstacle.

    Observation: relative goal (3), relative obstacle (3), current velocity (3),
    previous action (6), and remaining normalized episode time (1).
    Action: normalized Cartesian velocity offset [vx, vy, vz, wx, wy, wz],
    expressed in the task frame (x along the start->goal path).
    Angular components are included to preserve the final action contract, but
    this starter kinematic environment only applies the linear components.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_episode_steps: int | None = None,
        seed: int | None = None,
        scene: SceneConfig | None = None,
    ) -> None:
        super().__init__()
        self.scene = scene if scene is not None else DEFAULT_SCENE
        self.max_episode_steps = int(
            max_episode_steps if max_episode_steps is not None else self.scene.max_episode_steps
        )
        self.dt = float(self.scene.dt)
        self.start = np.asarray(self.scene.start, dtype=np.float32)
        self.goal = self.scene.goal.astype(np.float32)
        self.nominal_obstacle = self.scene.obstacle.astype(np.float32)
        self.obstacle = self.nominal_obstacle.copy()
        self.jitter_min = np.asarray(self.scene.obstacle_jitter_min, dtype=np.float32)
        self.jitter_max = np.asarray(self.scene.obstacle_jitter_max, dtype=np.float32)
        if self.scene.reward not in ("legacy", "shaped"):
            raise ValueError(f"unknown reward '{self.scene.reward}' (legacy or shaped)")
        self.obstacle_radius = float(self.scene.obstacle_radius)
        self.collision_distance = float(self.scene.collision_distance)
        self.goal_tolerance = float(self.scene.goal_tolerance)
        self.nominal_speed = float(self.scene.nominal_speed)
        self.max_offset_speed = float(self.scene.max_offset_speed)
        self.clearance_margin = float(self.scene.clearance_margin)
        self.max_lateral_offset = 0.32
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.position = self.start.copy()
        self.velocity = np.zeros(3, dtype=np.float32)
        self.previous_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.step_count = 0
        if seed is not None:
            self.reset(seed=seed)

    def _observation(self) -> np.ndarray:
        remaining = 1.0 - self.step_count / max(self.max_episode_steps, 1)
        return build_observation(
            self.goal - self.position,
            self.obstacle - self.position,
            self.velocity,
            self.previous_action,
            remaining,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.obstacle = self.nominal_obstacle.copy()
        if np.any(self.jitter_max > self.jitter_min):
            self.obstacle = self.obstacle + self.np_random.uniform(
                self.jitter_min, self.jitter_max
            ).astype(np.float32)
        self.position = self.start.copy()
        self.velocity.fill(0.0)
        self.previous_action.fill(0.0)
        self.step_count = 0
        return self._observation(), {"ard_enabled": False, "obstacle": self.obstacle.tolist()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(ACT_DIM)
        action = np.clip(action, -1.0, 1.0)
        previous_action = self.previous_action.copy()
        self.previous_action = action.copy()

        to_goal = self.goal - self.position
        goal_direction = to_goal / max(float(np.linalg.norm(to_goal)), 1e-6)
        nominal_velocity = self.nominal_speed * goal_direction
        offset_velocity = self.max_offset_speed * action[:3]
        self.velocity = nominal_velocity + offset_velocity
        self.position = self.position + self.velocity * self.dt
        self.step_count += 1

        distance_to_obstacle = float(np.linalg.norm(self.position - self.obstacle))
        distance_to_goal = float(np.linalg.norm(to_goal))
        collision = distance_to_obstacle <= self.collision_distance
        reached_goal = float(np.linalg.norm(self.goal - self.position)) <= self.goal_tolerance
        timeout = self.step_count >= self.max_episode_steps

        progress = distance_to_goal - float(np.linalg.norm(self.goal - self.position))
        reward = self._reward(
            progress, distance_to_obstacle, action, previous_action,
            collision, reached_goal, timeout and not reached_goal,
        )
        info = {
            "collision": collision,
            "reached_goal": reached_goal,
            "distance_to_obstacle_m": distance_to_obstacle,
            "distance_to_goal_m": float(np.linalg.norm(self.goal - self.position)),
            "ard_enabled": False,
        }
        return (
            self._observation(),
            float(reward),
            bool(collision or reached_goal),
            bool(timeout and not reached_goal),
            info,
        )

    def _reward(
        self, progress, distance_to_obstacle, action, previous_action, collision, reached, timed_out
    ) -> float:
        if self.scene.reward == "shaped":
            # The legacy reward made "stop before the obstacle" a local optimum
            # (standing still cost ~-2 per episode, trying to pass risked -100).
            # Strong progress term, time and timeout penalties, smaller terminal
            # terms, and a clearance penalty aligned with the safety filter.
            surface = distance_to_obstacle - self.obstacle_radius
            reward = 20.0 * progress - 0.05
            reward -= 2.0 * max(0.0, self.scene.penalty_clearance - surface)
            if collision:
                reward -= 30.0
            if reached:
                reward += 30.0
            if timed_out:
                reward -= 10.0
            return reward
        clearance_penalty = max(0.0, self.clearance_margin - distance_to_obstacle) * 0.35
        deviation_penalty = abs(float(self.position[1] - self.start[1])) * 0.08
        action_penalty = float(np.linalg.norm(action[:3] - previous_action[:3])) * 0.01
        reward = 2.0 * progress - clearance_penalty - deviation_penalty - action_penalty - 0.01
        if collision:
            reward -= 100.0
        if reached:
            reward += 100.0
        return reward

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
