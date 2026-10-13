"""Pure-numpy avoidance logic used by the action adapter (no ROS imports).

Pipeline per control step, all vectors in ``base_link``:

1. nominal velocity toward the goal;
2. mode-specific term: potential-field repulsion, or the RL offset;
3. clearance filter (removes the velocity component pointing at an obstacle
   once it is closer than ``slow_clearance_m``) - applied to every mode;
4. speed / workspace limits and orientation hold;
5. damped least squares to joint velocities, with joint-speed scaling,
   singularity and joint-limit checks.

Only the TCP is steered. Other arm points (elbow, wrist) are checked and cause
a hold when too close, but they are not actively pushed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .ur_kinematics import UR5E_Q_MAX, UR5E_Q_MIN, UR5eKinematics


@dataclass
class AvoidanceParams:
    nominal_speed: float = 0.05
    max_speed: float = 0.12
    offset_max: float = 0.08
    goal_tolerance: float = 0.03
    stop_clearance_m: float = 0.10
    slow_clearance_m: float = 0.22
    influence_m: float = 0.35
    link_stop_clearance_m: float = 0.04
    escape_direction: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 1.0))
    max_joint_vel: float = 0.6
    dls_damping: float = 0.05
    # Nakamura-style adaptive damping below this manipulability; hold below
    # ``min_manipulability``.
    damping_manip_threshold: float = 0.03
    damping_max: float = 0.25
    min_manipulability: float = 0.001
    z_min_m: float = 0.05
    orientation_gain: float = 1.0
    max_angular: float = 0.3
    horizon_s: float = 0.25
    # Repulsion fades out within this distance of the goal (GNRON fix).
    goal_fade_m: float = 0.15


@dataclass
class ObstacleView:
    center: np.ndarray
    radius: float = 0.0


@dataclass
class StepResult:
    qdot: np.ndarray
    v_cmd: np.ndarray
    status: str
    mode_used: str
    min_clearance_m: Optional[float]
    reached: bool = False


def _unit(vector: np.ndarray, fallback=(1.0, 0.0, 0.0)) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.asarray(fallback, dtype=float)
    return vector / norm


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm < 1e-12:
        return vector
    return vector * (limit / norm)


def clearance(point: np.ndarray, obstacle: ObstacleView, escape=(0.0, 0.0, 1.0)):
    """Surface clearance (m) and unit vector pointing from obstacle to point."""
    offset = point - obstacle.center
    dist = float(np.linalg.norm(offset))
    normal = offset / dist if dist > 1e-6 else _unit(np.asarray(escape, dtype=float))
    return dist - obstacle.radius, normal


def nominal_velocity(tcp: np.ndarray, goal: np.ndarray, speed: float, tolerance: float):
    to_goal = goal - tcp
    dist = float(np.linalg.norm(to_goal))
    if dist <= tolerance:
        return np.zeros(3), True
    ramp = min(1.0, dist / (5.0 * tolerance))
    return speed * ramp * (to_goal / dist), False


def repulsive_velocity(
    points: np.ndarray,
    goal_dir: np.ndarray,
    obstacles: Sequence[ObstacleView],
    p: AvoidanceParams,
    goal_distance: float = float("inf"),
) -> np.ndarray:
    """Repulsion plus a tangential component, so head-on cases do not deadlock.

    ``points`` are the guard points (TCP and wrist, which move together because
    orientation is held). The strongest contribution wins instead of summing,
    so several guard points near one obstacle do not multiply the command.
    """
    best = np.zeros(3)
    escape = np.asarray(p.escape_direction, dtype=float)
    span = max(p.influence_m - p.stop_clearance_m, 1e-6)
    fade = 1.0 if p.goal_fade_m <= 0 else float(np.clip(goal_distance / p.goal_fade_m, 0.0, 1.0))
    for point in np.atleast_2d(points):
        for obstacle in obstacles:
            d, normal = clearance(point, obstacle, escape)
            if d >= p.influence_m:
                continue
            frac = float(np.clip((p.influence_m - d) / span, 0.0, 1.0))
            tangent = goal_dir - float(np.dot(goal_dir, normal)) * normal
            if np.linalg.norm(tangent) < 0.3:  # nearly head-on: use the escape side
                tangent = escape - float(np.dot(escape, normal)) * normal
            if np.linalg.norm(tangent) < 1e-6:
                tangent = np.cross(normal, np.array([0.0, 0.0, 1.0]))
            tangent = _unit(tangent)
            candidate = fade * p.offset_max * frac * (0.6 * normal + 0.8 * tangent)
            if np.linalg.norm(candidate) > np.linalg.norm(best):
                best = candidate
    return best


def clearance_filter(
    v: np.ndarray, points: np.ndarray, obstacles: Sequence[ObstacleView], p: AvoidanceParams
) -> np.ndarray:
    """Limit the approach speed toward each obstacle as clearance shrinks."""
    out = v.copy()
    escape = np.asarray(p.escape_direction, dtype=float)
    span = max(p.slow_clearance_m - p.stop_clearance_m, 1e-6)
    for point in np.atleast_2d(points):
        for obstacle in obstacles:
            d, normal = clearance(point, obstacle, escape)
            if d >= p.slow_clearance_m:
                continue
            inward = -float(np.dot(out, normal))
            allowed = p.max_speed * float(np.clip((d - p.stop_clearance_m) / span, 0.0, 1.0))
            if inward > allowed:
                out = out + (inward - allowed) * normal
            if d < p.stop_clearance_m:  # gentle retreat if already inside the stop zone
                out = out + p.offset_max * float(
                    np.clip((p.stop_clearance_m - d) / p.stop_clearance_m, 0.0, 1.0)
                ) * normal
    return out


def whole_body_filter(
    qdot: np.ndarray,
    point_data: Sequence[tuple[np.ndarray, np.ndarray]],
    obstacles: Sequence[ObstacleView],
    p: AvoidanceParams,
    passes: int = 4,
) -> np.ndarray:
    """Project ``qdot`` so no arm point approaches an obstacle too quickly.

    For every control point and obstacle closer than ``slow_clearance_m`` the
    inward speed ``a . qdot`` (``a = -n^T J_point``) is limited to a value that
    shrinks to zero at ``stop_clearance_m`` and turns into a forced retreat
    inside it. Constraints are enforced by successive projections.
    """
    out = np.asarray(qdot, dtype=float).copy()
    escape = np.asarray(p.escape_direction, dtype=float)
    span = max(p.slow_clearance_m - p.stop_clearance_m, 1e-6)
    for _ in range(passes):
        changed = False
        for point, jac in point_data:
            for obstacle in obstacles:
                d, normal = clearance(point, obstacle, escape)
                if d >= p.slow_clearance_m:
                    continue
                a = -(normal @ jac)
                norm_sq = float(a @ a)
                if norm_sq < 1e-9:
                    continue
                if d >= p.stop_clearance_m:
                    allowed = p.max_speed * float((d - p.stop_clearance_m) / span)
                else:
                    allowed = -p.offset_max * float(
                        np.clip((p.stop_clearance_m - d) / p.stop_clearance_m, 0.0, 1.0)
                    )
                inward = float(a @ out)
                if inward > allowed + 1e-9:
                    out = out - ((inward - allowed) / norm_sq) * a
                    changed = True
        if not changed:
            break
    return out


def min_clearance(points: np.ndarray, obstacles: Sequence[ObstacleView]) -> Optional[float]:
    if not obstacles:
        return None
    best = np.inf
    for point in np.atleast_2d(points):
        for obstacle in obstacles:
            best = min(best, float(np.linalg.norm(point - obstacle.center)) - obstacle.radius)
    return float(best)


def _hold(mode: str, status: str, clearance_m: Optional[float], reached: bool = False) -> StepResult:
    return StepResult(np.zeros(6), np.zeros(3), status, mode, clearance_m, reached)


def pick_escape_direction(
    kin: UR5eKinematics,
    q: np.ndarray,
    path_direction,
    step_m: float = 0.15,
    z_min_m: float = 0.05,
    up=(0.0, 0.0, 1.0),
    iterations: int = 150,
):
    """Choose the sideways direction the arm can follow with the best conditioning.

    Candidates are +/- up and +/- the horizontal direction perpendicular to the
    path. Each one is tried kinematically (position-only DLS) for ``step_m``;
    the winner reaches the displaced point with the highest minimum
    manipulability. Returns ``(direction, scores)``.
    """
    q = np.asarray(q, dtype=float)
    path = _unit(np.asarray(path_direction, dtype=float))
    up = _unit(np.asarray(up, dtype=float), fallback=(0.0, 0.0, 1.0))
    lateral = np.cross(up, path)
    if np.linalg.norm(lateral) < 1e-6:
        lateral = np.cross(np.array([1.0, 0.0, 0.0]), path)
    lateral = _unit(lateral)
    candidates = {"up": up, "down": -up, "left": lateral, "right": -lateral}

    start = kin.position(q)
    scores: dict[str, float] = {}
    for name, direction in candidates.items():
        target = start + step_m * direction
        if target[2] < z_min_m:
            scores[name] = -1.0
            continue
        qq = q.copy()
        worst = np.inf
        for _ in range(iterations):
            error = target - kin.position(qq)
            if np.linalg.norm(error) < 2e-3:
                break
            jac = kin.jacobian(qq)
            worst = min(worst, kin.manipulability(jac))
            twist = np.concatenate((np.clip(5.0 * error, -0.1, 0.1), np.zeros(3)))
            qq = qq + 0.1 * kin.dls_solve(jac, twist, 0.05)
        reached = np.linalg.norm(target - kin.position(qq)) < 5e-3
        scores[name] = float(worst) if reached and np.isfinite(worst) else -1.0
    best = max(scores, key=lambda key: scores[key])
    # Prefer a horizontal escape: the tool hangs below/above the TCP, so a
    # sideways detour clears the whole tool, while a vertical one does not.
    lateral_best = max(("left", "right"), key=lambda key: scores[key])
    if scores[lateral_best] > 0.0 and scores[lateral_best] >= 0.6 * scores[best]:
        best = lateral_best
    return candidates[best], scores


def check_goal_reachable(
    kin: UR5eKinematics,
    q: np.ndarray,
    goal: np.ndarray,
    min_manipulability: float = 0.005,
    iterations: int = 600,
):
    """Kinematic feasibility probe, run once when an episode starts.

    Drives a copy of the arm toward ``goal`` with position-only DLS. Returns
    ``(ok, final_error_m, worst_manipulability)``; ``ok`` is False when the goal
    is out of reach or the straight path crosses a (near) singularity - for
    instance when the robot starts in the stretched-out default Gazebo pose.
    """
    qq = np.asarray(q, dtype=float).copy()
    worst = np.inf
    for _ in range(iterations):
        error = goal - kin.position(qq)
        if np.linalg.norm(error) < 2e-3:
            break
        jac = kin.jacobian(qq)
        worst = min(worst, kin.manipulability(jac))
        twist = np.concatenate((np.clip(5.0 * error, -0.1, 0.1), np.zeros(3)))
        qq = qq + 0.1 * kin.dls_solve(jac, twist, 0.05)
    final_error = float(np.linalg.norm(goal - kin.position(qq)))
    worst = float(worst) if np.isfinite(worst) else float(kin.manipulability(kin.jacobian(q)))
    return (final_error < 5e-3 and worst >= min_manipulability), final_error, worst


def compute_step(
    kin: UR5eKinematics,
    q: np.ndarray,
    goal: np.ndarray,
    obstacles: Sequence[ObstacleView],
    p: AvoidanceParams,
    mode: str = "potential_field",
    rl_offset: Optional[np.ndarray] = None,
    r_ref: Optional[np.ndarray] = None,
) -> StepResult:
    """Compute one joint-velocity command.

    ``mode``: ``potential_field`` | ``rl`` | ``nominal``. In ``rl`` mode a
    missing offset (``None``: stale or absent policy output) falls back to the
    deterministic potential field.
    """
    q = np.asarray(q, dtype=float)
    transform = kin.fk(q)
    tcp = transform[:3, 3]
    rotation = transform[:3, :3]
    arm_points = kin.link_points(q)  # upper-arm mid, elbow, forearm mid, wrist 1..3
    guards = np.vstack((tcp, arm_points[4], arm_points[5]))  # TCP + wrist: move together
    tcp_clearance = min_clearance(np.vstack((tcp, arm_points)), obstacles)

    v_nom, reached = nominal_velocity(tcp, goal, p.nominal_speed, p.goal_tolerance)
    if reached:
        return _hold(mode, "goal_reached", tcp_clearance, reached=True)

    mode_used = mode
    if mode == "rl" and rl_offset is None:
        mode_used = "potential_field"

    goal_dir = _unit(goal - tcp)
    if mode_used == "rl":
        v = v_nom + _clip_norm(np.asarray(rl_offset, dtype=float).reshape(3), p.offset_max)
    elif mode_used == "potential_field":
        v = v_nom + repulsive_velocity(
            guards, goal_dir, obstacles, p, float(np.linalg.norm(goal - tcp))
        )
    elif mode_used == "nominal":
        v = v_nom
    else:
        raise ValueError(f"unknown mode '{mode}'")

    v = clearance_filter(v, guards, obstacles, p)
    v = _clip_norm(v, p.max_speed)
    if tcp[2] <= p.z_min_m and v[2] < 0.0:
        v[2] = 0.0

    # Elbow / forearm are not steered: hold if they get too close. The wrist
    # guards are handled by the filter above and by this backstop.
    link_clearance = min_clearance(arm_points, obstacles)
    if link_clearance is not None and link_clearance < p.link_stop_clearance_m:
        return _hold(mode_used, "link_clearance", tcp_clearance)

    omega = np.zeros(3)
    if r_ref is not None:
        error = 0.5 * sum(np.cross(rotation[:, i], r_ref[:, i]) for i in range(3))
        omega = _clip_norm(p.orientation_gain * error, p.max_angular)

    jac = kin.jacobian(q)
    manip = kin.manipulability(jac)
    if manip < p.min_manipulability:
        return _hold(mode_used, "singular", tcp_clearance)
    damping = p.dls_damping
    if manip < p.damping_manip_threshold:
        ratio = manip / p.damping_manip_threshold
        damping = float(np.sqrt(damping**2 + p.damping_max**2 * (1.0 - ratio**2)))
    qdot = kin.dls_solve(jac, np.concatenate((v, omega)), damping)
    if obstacles:
        qdot = whole_body_filter(qdot, kin.arm_points_with_jacobians(q), obstacles, p)
        v = jac[:3] @ qdot

    peak = float(np.max(np.abs(qdot)))
    status = "ok"
    if peak > p.max_joint_vel:
        qdot = qdot * (p.max_joint_vel / peak)
        v = v * (p.max_joint_vel / peak)
        status = "joint_speed_scaled"

    q_next = q + qdot * p.horizon_s
    violating = (q_next < UR5E_Q_MIN) | (q_next > UR5E_Q_MAX)
    if np.any(violating):
        qdot = np.where(violating, 0.0, qdot)
        status = "joint_limit"

    return StepResult(qdot, v, status, mode_used, tcp_clearance)
