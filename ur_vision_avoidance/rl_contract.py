"""Single source of truth for the RL observation/action contract.

Both the training environment (``fixed_scene_env``) and the ROS observation
node build the 16-element vector through :func:`build_observation`, so the
policy always sees the same layout at training and at run time.

Layout (all quantities relative to the TCP, expressed in ``base_link``):

    [0:3]   goal - tcp
    [3:6]   obstacle_center - tcp
    [6:9]   tcp linear velocity
    [9:15]  previous normalized action (6)
    [15]    remaining normalized episode time in [0, 1]
"""

from __future__ import annotations

import numpy as np

OBS_DIM = 16
ACT_DIM = 6
OBSERVATION_SCHEMA = "ur_vision_avoidance.rl_observation.v2"


def _vec(name: str, value, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape[0] != size:
        raise ValueError(f"{name} must have {size} elements, got {array.shape[0]}")
    return array


def build_observation(goal_rel, obstacle_rel, velocity, previous_action, remaining) -> np.ndarray:
    """Return the float32 observation vector of length ``OBS_DIM``."""
    return np.concatenate(
        (
            _vec("goal_rel", goal_rel, 3),
            _vec("obstacle_rel", obstacle_rel, 3),
            _vec("velocity", velocity, 3),
            _vec("previous_action", previous_action, ACT_DIM),
            np.array([float(remaining)], dtype=np.float32),
        )
    ).astype(np.float32)


def task_rotation(direction, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotation ``R`` (task -> base_link) whose x axis follows the path.

    The training environment moves along +x with Z up, i.e. the task frame is
    the identity there. On the robot the path can point anywhere, so the
    observation and the action are expressed in this task frame; that keeps a
    policy trained along +x valid for any path direction.
    """
    x = np.asarray(direction, dtype=float).reshape(3)
    norm = np.linalg.norm(x)
    if norm < 1e-9:
        return np.eye(3)
    x = x / norm
    up = np.asarray(up, dtype=float).reshape(3)
    y = np.cross(up, x)
    if np.linalg.norm(y) < 1e-6:  # path parallel to "up": pick another helper axis
        y = np.cross(np.array([1.0, 0.0, 0.0]), x)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    return np.column_stack((x, y, z))


def action_scale(max_linear: float, max_angular: float) -> np.ndarray:
    """Per-dimension scale converting a normalized action to m/s and rad/s."""
    return np.array(
        [max_linear] * 3 + [max_angular] * 3,
        dtype=np.float32,
    )
