"""Scene configuration shared by the training environment and the ROS nodes.

``config/rl_starter.yaml`` is the single place where the scene is described.
Positions are stored as offsets from the start pose, so the same numbers work
both in the abstract training environment (start at an arbitrary point) and on
the robot (start = TCP position when the episode begins, in ``base_link``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SceneConfig:
    start: tuple = (0.0, 0.0, 0.45)
    goal_offset: tuple = (1.2, 0.0, 0.0)
    obstacle_offset: tuple = (0.65, 0.0, 0.0)
    obstacle_radius: float = 0.16
    collision_distance: float = 0.22
    goal_tolerance: float = 0.08
    nominal_speed: float = 0.08
    max_offset_speed: float = 0.12
    max_angular: float = 0.35
    dt: float = 0.1
    max_episode_steps: int = 160
    clearance_margin: float = 0.45
    # "legacy" keeps the original starter reward; "shaped" is the reward that
    # converged on the UR5e (see fixed_scene_env.FixedSceneAvoidanceEnv._reward).
    reward: str = "legacy"
    # Surface clearance under which the shaped reward penalizes (keep it above
    # the action_adapter stop_clearance_m so the policy does not learn paths
    # that the safety filter blocks).
    penalty_clearance: float = 0.16
    # Uniform obstacle-position jitter at each reset (task frame, metres).
    obstacle_jitter_min: tuple = (0.0, 0.0, 0.0)
    obstacle_jitter_max: tuple = (0.0, 0.0, 0.0)

    @property
    def goal(self) -> np.ndarray:
        return np.asarray(self.start, dtype=np.float32) + np.asarray(
            self.goal_offset, dtype=np.float32
        )

    @property
    def obstacle(self) -> np.ndarray:
        return np.asarray(self.start, dtype=np.float32) + np.asarray(
            self.obstacle_offset, dtype=np.float32
        )

    @property
    def episode_duration_s(self) -> float:
        return self.max_episode_steps * self.dt


DEFAULT_SCENE = SceneConfig()


def default_config_path() -> Path:
    """Installed config if available, otherwise the one in the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = (
            Path(get_package_share_directory("ur_vision_avoidance"))
            / "config"
            / "rl_starter.yaml"
        )
        if installed.exists():
            return installed
    except Exception:  # noqa: BLE001 - ament is optional outside ROS
        pass
    return Path(__file__).resolve().parents[1] / "config" / "rl_starter.yaml"


def load_scene(path: Optional[Path | str] = None, section: str = "fixed_scene") -> SceneConfig:
    """Read one scene section from the YAML file.

    ``fixed_scene`` reproduces the original starter scene. ``ur5e_scene`` is a
    scaled-down variant that fits inside the UR5e workspace (reach 0.85 m).
    """
    import yaml

    config_path = Path(path) if path else default_config_path()
    if not config_path.exists():
        if section == "fixed_scene":
            return DEFAULT_SCENE
        raise FileNotFoundError(config_path)

    with open(config_path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if section not in document:
        raise KeyError(f"scene section '{section}' not found in {config_path}")

    scene = document[section]
    action = document.get("action", {})
    base = DEFAULT_SCENE

    start = np.asarray(scene.get("start_position_m", base.start), dtype=float)
    goal = np.asarray(
        scene.get("goal_position_m", np.asarray(base.start) + np.asarray(base.goal_offset)),
        dtype=float,
    )
    obstacle = np.asarray(
        scene.get(
            "obstacle_position_m", np.asarray(base.start) + np.asarray(base.obstacle_offset)
        ),
        dtype=float,
    )

    def pick(key: str, fallback_key: str, default: float, source=action) -> float:
        if key in scene:
            return float(scene[key])
        return float(source.get(fallback_key, default))

    return SceneConfig(
        start=tuple(float(x) for x in start),
        goal_offset=tuple(float(x) for x in (goal - start)),
        obstacle_offset=tuple(float(x) for x in (obstacle - start)),
        obstacle_radius=float(scene.get("obstacle_radius_m", base.obstacle_radius)),
        collision_distance=float(scene.get("collision_distance_m", base.collision_distance)),
        goal_tolerance=float(scene.get("goal_tolerance_m", base.goal_tolerance)),
        nominal_speed=float(scene.get("nominal_speed_mps", base.nominal_speed)),
        max_offset_speed=pick("max_linear_mps", "max_linear_mps", base.max_offset_speed),
        max_angular=pick("max_angular_rps", "max_angular_rps", base.max_angular),
        dt=pick("control_period_s", "control_period_s", base.dt),
        max_episode_steps=int(scene.get("max_episode_steps", base.max_episode_steps)),
        clearance_margin=float(scene.get("clearance_margin_m", base.clearance_margin)),
        reward=str(scene.get("reward", base.reward)),
        penalty_clearance=float(scene.get("penalty_clearance_m", base.penalty_clearance)),
        obstacle_jitter_min=tuple(
            float(x) for x in scene.get("obstacle_jitter_min_m", base.obstacle_jitter_min)
        ),
        obstacle_jitter_max=tuple(
            float(x) for x in scene.get("obstacle_jitter_max_m", base.obstacle_jitter_max)
        ),
    )
