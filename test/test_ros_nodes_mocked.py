"""Drive the real node classes through fake ROS plumbing (see _fake_ros.py)."""

import json
import unittest
from types import SimpleNamespace

import numpy as np

from _common import Q_HOME
import _fake_ros

FAKED = _fake_ros.install()

from std_msgs.msg import Empty, Float32MultiArray, String  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from ur_vision_avoidance.rl_contract import OBS_DIM, OBSERVATION_SCHEMA, build_observation, task_rotation  # noqa: E402
from ur_vision_avoidance.ur_kinematics import UR5eKinematics  # noqa: E402

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def jstring(payload):
    return String(data=json.dumps(payload))


def joint_state(q, qd=None):
    return JointState(name=list(JOINTS), position=[float(x) for x in q], velocity=[float(x) for x in (qd if qd is not None else np.zeros(6))])


@unittest.skipUnless(FAKED, "real rclpy present: run these nodes under ROS instead")
class ObservationNodeTest(unittest.TestCase):
    def setUp(self):
        _fake_ros.PARAM_OVERRIDES.clear()
        from ur_vision_avoidance.rl_observation_node import RLObservationNode

        self.node = RLObservationNode()
        self.node.get_clock().set_seconds(10.0)

    def task_state(self, **kw):
        rotation = task_rotation([0.0, -1.0, 0.0])
        state = {
            "episode_active": True,
            "tcp": [0.0, 0.0, 0.0],
            "goal": [0.0, -0.45, 0.0],
            "task_rotation": rotation.reshape(-1).tolist(),
            "tcp_velocity": [0.0, -0.05, 0.0],
            "elapsed_s": 5.0,
            "episode_duration_s": 20.0,
            "status": "ok",
        }
        state.update(kw)
        return state

    def tracks(self, tracks=None, perception_age=0.1):
        return {"perception_age_s": perception_age, "tracks": tracks if tracks is not None else [
            {"id": 3, "position": [0.0, -0.22, 0.0], "radius_m": 0.06, "visible": True, "confidence": 0.9}]}

    def publish(self):
        self.node.publish_observation()
        return json.loads(self.node.publishers["/rl/observation"].last.data)

    def test_vector_is_built_in_the_task_frame_like_in_training(self):
        self.node.on_task_state(jstring(self.task_state()))
        self.node.on_tracks(jstring(self.tracks()))
        payload = self.publish()
        self.assertEqual(payload["schema"], OBSERVATION_SCHEMA)
        self.assertTrue(payload["sensor_valid"])
        self.assertTrue(payload["obstacle"]["valid"])
        expected = build_observation([0.45, 0, 0], [0.22, 0, 0], [0.05, 0, 0], np.zeros(6), 0.75)
        np.testing.assert_allclose(payload["vector"], expected, atol=1e-6)
        self.assertEqual(len(payload["vector"]), OBS_DIM)

    def test_previous_action_is_normalized_into_its_slot(self):
        self.node.on_task_state(jstring(self.task_state()))
        self.node.on_tracks(jstring(self.tracks()))
        scale = self.node.scale
        self.node.on_action(Float32MultiArray(data=[0.5 * scale[0], 0.0, -scale[2], 0.0, 0.0, 0.0]))
        vector = self.publish()["vector"]
        np.testing.assert_allclose(vector[9:15], [0.5, 0.0, -1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_no_track_marks_obstacle_invalid(self):
        self.node.on_task_state(jstring(self.task_state()))
        self.node.on_tracks(jstring(self.tracks(tracks=[])))
        payload = self.publish()
        self.assertFalse(payload["obstacle"]["valid"])
        self.assertTrue(payload["sensor_valid"])  # perception is healthy, the scene is just empty

    def test_stale_perception_invalidates_the_observation(self):
        self.node.on_task_state(jstring(self.task_state()))
        self.node.on_tracks(jstring(self.tracks(perception_age=5.0)))
        self.assertFalse(self.publish()["sensor_valid"])

    def test_stale_messages_invalidate_the_observation(self):
        self.node.on_task_state(jstring(self.task_state()))
        self.node.on_tracks(jstring(self.tracks()))
        self.node.get_clock().set_seconds(11.0)  # 1 s later, timeout is 0.5 s
        self.assertFalse(self.publish()["sensor_valid"])

    def test_without_any_input_the_observation_is_invalid_zeros(self):
        payload = self.publish()
        self.assertFalse(payload["sensor_valid"])
        self.assertEqual(payload["vector"], [0.0] * OBS_DIM)


@unittest.skipUnless(FAKED, "real rclpy present")
class PolicyNodeTest(unittest.TestCase):
    class Model:
        def __init__(self, action):
            self.action = np.asarray(action, dtype=np.float32)

        def predict(self, vector, deterministic=True):
            assert vector.shape == (OBS_DIM,)
            return self.action, None

    def setUp(self):
        _fake_ros.PARAM_OVERRIDES.clear()
        from ur_vision_avoidance.rl_policy_node import RLPolicyNode

        self.node = RLPolicyNode()
        self.node.model = self.Model([2.0, -0.5, 0.0, 0.0, 0.0, 0.0])  # first value must be clipped

    def payload(self, **kw):
        base = {
            "schema": OBSERVATION_SCHEMA,
            "sensor_valid": True,
            "vector": [0.0] * OBS_DIM,
            "obstacle": {"valid": True, "distance_m": 0.2},
        }
        base.update(kw)
        return base

    def test_action_is_clipped_to_the_normalized_range(self):
        np.testing.assert_allclose(self.node.compute_action(self.payload()), [1.0, -0.5, 0, 0, 0, 0])

    def test_zero_action_without_model_or_with_wrong_schema(self):
        self.node.model = None
        self.assertEqual(float(np.abs(self.node.compute_action(self.payload())).sum()), 0.0)
        self.node.model = self.Model([1, 1, 1, 0, 0, 0])
        self.assertEqual(float(np.abs(self.node.compute_action(self.payload(schema="old.v1"))).sum()), 0.0)

    def test_zero_action_when_invalid_absent_or_out_of_training_range(self):
        for kw in (
            {"sensor_valid": False},
            {"obstacle": {"valid": False}},
            {"obstacle": {"valid": True, "distance_m": 5.0}},
            {"vector": [0.0] * 3},
        ):
            with self.subTest(kw=kw):
                self.assertEqual(float(np.abs(self.node.compute_action(self.payload(**kw))).sum()), 0.0)

    def test_published_action_is_scaled_to_si_units(self):
        self.node.observation_callback(jstring(self.payload()))
        data = self.node.publishers["/rl/action"].last.data
        self.assertAlmostEqual(data[0], float(self.node.scale[0]), places=6)
        self.assertAlmostEqual(data[1], -0.5 * float(self.node.scale[1]), places=6)


@unittest.skipUnless(FAKED, "real rclpy present")
class ActionAdapterTest(unittest.TestCase):
    def make(self, **params):
        params.setdefault("auto_start", True)
        _fake_ros.PARAM_OVERRIDES.clear()
        _fake_ros.PARAM_OVERRIDES["action_adapter"] = params
        from ur_vision_avoidance.action_adapter_node import ActionAdapter

        node = ActionAdapter()
        node.get_clock().set_seconds(1.0)
        return node

    def feed(self, node, tracks=None, perception_age=0.05, q=Q_HOME):
        node.on_joints(joint_state(q))
        node.on_tracks(jstring({"perception_age_s": perception_age, "tracks": tracks or []}))

    def state(self, node):
        return json.loads(node.publishers["/task/state"].last.data)

    def test_episode_starts_at_the_current_pose_and_publishes_task_state(self):
        node = self.make()
        self.feed(node)
        node.tick()
        state = self.state(node)
        kin = UR5eKinematics()
        np.testing.assert_allclose(state["tcp"], kin.position(Q_HOME), atol=1e-9)
        np.testing.assert_allclose(state["goal"], kin.position(Q_HOME) + [0.0, -0.45, 0.0], atol=1e-9)
        self.assertEqual(np.asarray(state["task_rotation"]).shape, (9,))
        self.assertEqual(state["status"], "ok")
        self.assertFalse(state["command_enabled"])

    def test_commands_reach_the_controller_only_when_enabled(self):
        node = self.make()
        self.feed(node)
        node.tick()
        self.assertEqual(len(node.publishers["/avoidance/adapter_command"].messages), 1)
        self.assertEqual(node.publishers["/scaled_joint_trajectory_controller/joint_trajectory"].messages, [])

        enabled = self.make(enable_command=True)
        self.feed(enabled)
        enabled.tick()
        sent = enabled.publishers["/scaled_joint_trajectory_controller/joint_trajectory"].last
        self.assertEqual(sent.joint_names, JOINTS)
        point = sent.points[0]
        self.assertEqual(len(point.positions), 6)
        self.assertGreater(float(np.max(np.abs(np.asarray(point.positions) - Q_HOME))), 1e-4)  # it moves

    def test_trajectory_point_has_no_velocity_so_the_controller_accepts_it(self):
        # The joint trajectory controller rejects a last point with non-zero velocity.
        node = self.make(enable_command=True)
        self.feed(node)
        node.tick()
        point = node.publishers["/scaled_joint_trajectory_controller/joint_trajectory"].last.points[-1]
        self.assertEqual(list(getattr(point, "velocities", [])), [])
        self.assertEqual(len(point.positions), 6)

    def test_without_auto_start_nothing_is_commanded_until_start(self):
        node = self.make(enable_command=True, auto_start=False)
        controller = "/scaled_joint_trajectory_controller/joint_trajectory"
        self.feed(node)
        node.tick()
        self.assertEqual(self.state(node)["status"], "waiting_start")
        self.assertFalse(self.state(node)["episode_active"])
        self.assertEqual(node.publishers[controller].messages, [])

        node.on_start(Empty())
        node.tick()
        self.assertTrue(self.state(node)["episode_active"])
        self.assertEqual(len(node.publishers[controller].messages), 1)

        node.on_stop(Empty())
        node.tick()
        self.assertEqual(self.state(node)["status"], "waiting_start")
        self.assertEqual(len(node.publishers[controller].messages), 1)

    def test_stale_perception_holds_position(self):
        node = self.make(enable_command=True)
        self.feed(node, perception_age=5.0)
        node.tick()
        self.assertEqual(self.state(node)["status"], "perception_stale")
        point = node.publishers["/scaled_joint_trajectory_controller/joint_trajectory"].last.points[0]
        np.testing.assert_allclose(point.positions, Q_HOME, atol=1e-12)

    def test_missing_perception_can_be_allowed_explicitly(self):
        node = self.make(require_fresh_perception=False)
        node.on_joints(joint_state(Q_HOME))
        node.tick()
        self.assertEqual(self.state(node)["status"], "ok")

    def test_stale_joint_state_publishes_status_without_a_command(self):
        node = self.make(enable_command=True)
        self.feed(node)
        node.tick()
        node.get_clock().set_seconds(5.0)
        before = len(node.publishers["/scaled_joint_trajectory_controller/joint_trajectory"].messages)
        node.tick()
        self.assertEqual(self.state(node)["status"], "stale_joint_state")
        self.assertEqual(len(node.publishers["/scaled_joint_trajectory_controller/joint_trajectory"].messages), before)

    def test_unreachable_goal_holds_and_explains(self):
        stretched = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]  # default Gazebo pose: singular
        node = self.make(enable_command=True)
        self.feed(node, q=stretched)
        node.tick()
        self.assertEqual(self.state(node)["status"], "goal_unreachable")
        self.assertTrue(any("not reachable" in text for _, text in node.get_logger().records))
        point = node.publishers["/scaled_joint_trajectory_controller/joint_trajectory"].last.points[0]
        np.testing.assert_allclose(point.positions, stretched, atol=1e-12)

    def test_reset_starts_a_new_episode_from_the_new_pose(self):
        node = self.make()
        self.feed(node)
        node.tick()
        first_goal = np.asarray(self.state(node)["goal"])
        moved = Q_HOME + np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        node.on_joints(joint_state(moved))
        node.on_reset(Empty())
        node.tick()
        self.assertGreater(float(np.linalg.norm(np.asarray(self.state(node)["goal"]) - first_goal)), 1e-3)

    def test_obstacle_track_slows_the_approach_and_stays_safe(self):
        kin = UR5eKinematics()
        tcp = kin.position(Q_HOME)
        track = {"position": (tcp + [0.0, -0.2, 0.0]).tolist(), "radius_m": 0.06}
        free, blocked = self.make(), self.make()
        self.feed(free)
        self.feed(blocked, tracks=[track])
        free.tick()
        blocked.tick()
        self.assertLess(self.state(blocked)["min_clearance_m"], 0.25)
        step_free = np.asarray(free.publishers["/avoidance/adapter_command"].last.points[0].positions) - Q_HOME
        step_blocked = np.asarray(blocked.publishers["/avoidance/adapter_command"].last.points[0].positions) - Q_HOME
        self.assertNotAlmostEqual(float(np.linalg.norm(step_free)), float(np.linalg.norm(step_blocked)), places=4)

    def test_rl_mode_rotates_the_task_frame_action_into_base_link(self):
        node = self.make(mode="rl")
        self.feed(node)
        node.tick()  # start episode, mode rl falls back (no action yet)
        self.assertEqual(self.state(node)["mode"], "potential_field")
        node.on_action(Float32MultiArray(data=[0.0, 0.05, 0.0, 0.0, 0.0, 0.0]))  # +y in the TASK frame
        node.tick()
        self.assertEqual(self.state(node)["mode"], "rl")
        stored = node.rotation @ node.rl_offset_task
        # task y axis for a -y path with up=+z is +x in base_link
        np.testing.assert_allclose(stored, [0.05, 0.0, 0.0], atol=1e-9)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make(mode="teleport")


class FakeCameraInfo:
    k = [554.25, 0.0, 320.0, 0.0, 554.25, 240.0, 0.0, 0.0, 1.0]


@unittest.skipUnless(FAKED, "real rclpy present")
@unittest.skipUnless(FAKED, "real rclpy present")
class MetricsNodeTest(unittest.TestCase):
    def test_one_episode_produces_one_row(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            _fake_ros.PARAM_OVERRIDES.clear()
            _fake_ros.PARAM_OVERRIDES["metrics_node"] = {"output_dir": tmp, "run_label": "pf"}
            from ur_vision_avoidance.metrics_node import MetricsNode

            node = MetricsNode()
            kin = UR5eKinematics()
            tcp = kin.position(Q_HOME)

            def state(t, **kw):
                payload = {"episode_active": True, "elapsed_s": t, "start": tcp.tolist(),
                           "goal": (tcp + [0, -0.45, 0]).tolist(), "mode": "potential_field",
                           "episode_duration_s": 20.0, "status": "ok", "reached": False}
                payload.update(kw)
                node.get_clock().set_seconds(100.0 + t)
                node.on_task_state(jstring(payload))

            state(0.0)
            node.on_joints(joint_state(Q_HOME))
            node.on_detections(jstring({"obstacle": False, "age_s": 0.1, "inference_ms": 30.0, "detections": []}))
            state(1.0)
            node.on_joints(joint_state(Q_HOME + np.array([0.05, 0, 0, 0, 0, 0])))
            state(2.0, reached=True, status="goal_reached")
            state(2.1, reached=True, status="goal_reached")  # lingering state: no new episode

            summary = json.loads(node.publishers["/metrics/episode"].last.data)
            self.assertEqual(len(node.publishers["/metrics/episode"].messages), 1)
            self.assertEqual(summary["outcome"], "reached")
            self.assertEqual(summary["run_label"], "pf")
            self.assertAlmostEqual(summary["time_to_goal_s"], 2.0, places=3)
            self.assertFalse(summary["collision"])
            lines = (Path(tmp) / "episodes.csv").read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)  # header + one episode
            self.assertEqual(len((Path(tmp) / "episodes.jsonl").read_text().strip().splitlines()), 1)


class LocalizerNodeTest(unittest.TestCase):
    def setUp(self):
        _fake_ros.PARAM_OVERRIDES.clear()
        import tf2_ros

        # camera looks along +x of base_link, at the origin, 0.5 m up
        tf2_ros.Buffer.calls = []
        tf2_ros.Buffer.transform = SimpleNamespace(
            transform=SimpleNamespace(
                rotation=SimpleNamespace(x=-0.5, y=0.5, z=-0.5, w=0.5),  # optical z -> base x
                translation=SimpleNamespace(x=0.0, y=0.0, z=0.5),
            )
        )
        from ur_vision_avoidance.obstacle_localizer_node import ObstacleLocalizerNode

        self.node = ObstacleLocalizerNode()
        self.node.get_clock().set_seconds(3.0)
        self.node.camera_info_callback(FakeCameraInfo())
        self.tf2_ros = tf2_ros

    def detections(self, distance=1.0, depth_fresh=True):
        return jstring({
            "stamp_ns": 2_900_000_000,
            "depth_fresh": depth_fresh,
            "detections": [{
                "class": "chair", "confidence": 0.8,
                "bbox_xyxy": [270.0, 190.0, 370.0, 290.0],  # centred on the principal point
                "distance_m": distance,
            }],
        })

    def tracks(self):
        self.node.publish_tracks()
        return json.loads(self.node.publishers["/obstacle/tracks"].last.data)

    def test_surfaces_on_the_floor_are_not_obstacles(self):
        # Box at the bottom of the image: 1.4 m ahead the ray is below z = 0 (the floor).
        floor = jstring({
            "stamp_ns": 2_900_000_000, "depth_fresh": True,
            "detections": [{"class": "unknown", "confidence": 1.0, "source": "depth",
                            "bbox_xyxy": [270.0, 440.0, 370.0, 479.0], "distance_m": 1.4}],
        })
        for k in range(3):
            self.node.get_clock().set_seconds(3.0 + 0.1 * k)
            self.node.detections_callback(floor)
        payload = self.tracks()
        self.assertEqual(payload["tracks"], [])
        self.assertEqual(payload["dropped_below_floor"], 3)

    def test_depth_anchor_replaces_bbox_centre_and_percentile(self):
        # Floor strip: low-percentile distance at the bbox centre would sit above
        # the floor; the anchor (pixel + depth measured there) is on the floor.
        strip = jstring({
            "stamp_ns": 2_900_000_000, "depth_fresh": True,
            "detections": [{"class": "unknown", "confidence": 1.0, "source": "depth",
                            "bbox_xyxy": [0.0, 420.0, 640.0, 480.0], "distance_m": 0.9,
                            "anchor_uv": [320.0, 450.0], "anchor_depth_m": 1.32}],
        })
        for k in range(3):
            self.node.get_clock().set_seconds(3.0 + 0.1 * k)
            self.node.detections_callback(strip)
        self.assertEqual(self.tracks()["tracks"], [])
        self.assertEqual(self.node.dropped_below_floor, 3)

    def test_detection_becomes_a_3d_track_in_base_link(self):
        for k in range(2):  # two hits to confirm
            self.node.get_clock().set_seconds(3.0 + 0.1 * k)
            self.node.detections_callback(self.detections(distance=1.0))
        track = self.tracks()["tracks"][0]
        # surface 1 m ahead of the camera along +x, 0.5 m up; centre pushed back by the radius
        np.testing.assert_allclose(track["surface"], [1.0, 0.0, 0.5], atol=1e-3)
        self.assertGreater(track["position"][0], 1.0)
        self.assertEqual(track["label"], "chair")
        self.assertEqual(self.tf2_ros.Buffer.calls[0][2], 2_900_000_000)  # TF looked up at image time

    def test_track_is_remembered_after_detections_stop_and_flags_perception_age(self):
        for k in range(2):
            self.node.get_clock().set_seconds(3.0 + 0.1 * k)
            self.node.detections_callback(self.detections())
        self.node.get_clock().set_seconds(6.0)
        payload = self.tracks()
        self.assertEqual(len(payload["tracks"]), 1)
        self.assertFalse(payload["tracks"][0]["visible"])
        self.assertAlmostEqual(payload["perception_age_s"], 2.9, places=2)

    def test_missing_transform_drops_detections_instead_of_guessing(self):
        self.tf2_ros.Buffer.transform = None
        for k in range(3):
            self.node.get_clock().set_seconds(3.0 + 0.1 * k)
            self.node.detections_callback(self.detections())
        self.assertEqual(self.tracks()["tracks"], [])
        self.assertGreaterEqual(self.tracks()["dropped_for_tf"], 3)

    def test_stale_depth_is_ignored(self):
        for k in range(3):
            self.node.get_clock().set_seconds(3.0 + 0.1 * k)
            self.node.detections_callback(self.detections(depth_fresh=False))
        self.assertEqual(self.tracks()["tracks"], [])

    def test_markers_are_published(self):
        for k in range(2):
            self.node.get_clock().set_seconds(3.0 + 0.1 * k)
            self.node.detections_callback(self.detections())
        self.node.publish_tracks()
        markers = self.node.publishers["/obstacle/markers"].last.markers
        self.assertEqual(len(markers), 2)  # DELETEALL + one sphere


if __name__ == "__main__":
    unittest.main()
