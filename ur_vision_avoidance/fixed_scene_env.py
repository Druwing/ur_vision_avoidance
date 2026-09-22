"""Fixed-scene RL environment for the UR vision-avoidance starter stage.

This environment is deliberately deterministic apart from the optional seed. It
is a training scaffold, not a physics simulator and not a replacement for the
Gazebo/UR controller. ARD is intentionally absent from this stage.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FixedSceneAvoidanceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Learn a bounded Cartesian offset around one fixed spherical obstacle.

    Observation: relative goal (3), relative obstacle (3), current velocity (3),
    previous action (6), and remaining normalized episode time (1).
    Action: normalized Cartesian velocity offset [vx, vy, vz, wx, wy, wz].
    Angular components are included to preserve the final action contract, but
    this starter kinematic environment only applies the linear components.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 160, seed: int | None = None) -> None:
        super().__init__()
        self.max_episode_steps = int(max_episode_steps)
        self.dt = 0.1
        self.goal = np.array([1.20, 0.0, 0.45], dtype=np.float32)
        self.obstacle = np.array([0.65, 0.0, 0.45], dtype=np.float32)
        self.obstacle_radius = 0.16
        self.collision_distance = 0.22
        self.goal_tolerance = 0.08
        self.nominal_speed = 0.08
        self.max_offset_speed = 0.12
        self.max_lateral_offset = 0.32
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.previous_action = np.zeros(6, dtype=np.float32)
        self.step_count = 0
        if seed is not None:
            self.reset(seed=seed)

    def _observation(self) -> np.ndarray:
        remaining = 1.0 - self.step_count / max(self.max_episode_steps, 1)
        return np.concatenate(
            (
                self.goal - self.position,
                self.obstacle - self.position,
                self.velocity,
                self.previous_action,
                np.array([remaining], dtype=np.float32),
            )
        ).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.position = np.array([0.0, 0.0, 0.45], dtype=np.float32)
        self.velocity.fill(0.0)
        self.previous_action.fill(0.0)
        self.step_count = 0
        return self._observation(), {"ard_enabled": False}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(6)
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
        clearance_penalty = max(0.0, 0.45 - distance_to_obstacle) * 0.35
        deviation_penalty = abs(float(self.position[1])) * 0.08
        action_penalty = float(np.linalg.norm(action[:3] - previous_action[:3])) * 0.01
        reward = 2.0 * progress - clearance_penalty - deviation_penalty - action_penalty - 0.01
        if collision:
            reward -= 100.0
        if reached_goal:
            reward += 100.0
        info = {
            "collision": collision,
            "reached_goal": reached_goal,
            "distance_to_obstacle_m": distance_to_obstacle,
            "distance_to_goal_m": float(np.linalg.norm(self.goal - self.position)),
            "ard_enabled": False,
        }
        return self._observation(), float(reward), bool(collision or reached_goal), bool(timeout and not reached_goal), info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
