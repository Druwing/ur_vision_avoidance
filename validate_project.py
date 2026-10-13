from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(__file__).parent

for relative in [
    "urdf/ur_gz_camera.urdf.xacro",
    "worlds/obstacle_world.sdf",
]:
    path = root / relative
    ET.parse(path)
    print(f"XML OK: {relative}")

for relative in [
    "package.xml",
    "setup.py",
    "launch/vision_bringup.launch.py",
    "ur_vision_avoidance/vision_node.py",
    "ur_vision_avoidance/avoidance_supervisor.py",
    "yolo26n.pt",
    "config/rl_starter.yaml",
    "requirements-rl.txt",
    "ur_vision_avoidance/fixed_scene_env.py",
    "ur_vision_avoidance/train_fixed_scene.py",
    "ur_vision_avoidance/rl_observation_node.py",
    "ur_vision_avoidance/rl_policy_node.py",
    "ur_vision_avoidance/rl_contract.py",
    "ur_vision_avoidance/scene_config.py",
    "ur_vision_avoidance/ur_kinematics.py",
    "ur_vision_avoidance/avoidance_logic.py",
    "ur_vision_avoidance/obstacle_tracker.py",
    "ur_vision_avoidance/camera_geometry.py",
    "ur_vision_avoidance/obstacle_localizer_node.py",
    "ur_vision_avoidance/action_adapter_node.py",
    "test/test_closed_loop.py",
    "ur_vision_avoidance/depth_fallback.py",
    "ur_vision_avoidance/metrics.py",
    "ur_vision_avoidance/metrics_node.py",
    "ur_vision_avoidance/dataset_labels.py",
    "ur_vision_avoidance/dataset_capture_node.py",
    "config/scene_ground_truth.yaml",
    "models/ppo_ur5e_scene.zip",
    "scripts/run_gazebo_episode.sh",
    "scripts/capture_dataset.sh",
    "scripts/evaluate_kinematic.py",
    "scripts/benchmark_yolo.py",
    "scripts/summarize_metrics.py",
]:
    path = root / relative
    assert path.exists(), path
    print(f"FILE OK: {relative}")

bridge_text = (root / "config/bridge.yaml").read_text()
required = [
    "sensor_msgs/msg/Image",
    "gz.msgs.Image",
    "/camera/image_raw",
    "/camera/depth_image_raw",
    "/camera/camera_info",
]
for token in required:
    assert token in bridge_text, token
print("BRIDGE CONFIG OK")

setup_text = (root / "setup.py").read_text()
assert '"yolo26n.pt"' in setup_text, "YOLO checkpoint is not installed"
xacro_text = (root / "urdf/ur_gz_camera.urdf.xacro").read_text()
assert 'default="ur5e"' in xacro_text, "Xacro default UR model is inconsistent"
launch_text = (root / "launch/vision_bringup.launch.py").read_text()
assert '"models", "yolo26n.pt"' in launch_text, "Launch does not use installed model"
print("MODEL INSTALLATION AND DEFAULTS OK")

rl_config = (root / "config/rl_starter.yaml").read_text()
assert "ard_enabled: false" in rl_config, "Starter stage must keep ARD disabled"
assert "fixed_scene" in rl_config, "Fixed-scene RL configuration is missing"
print("FIXED-SCENE RL STARTER CONFIG OK")

assert "ur5e_scene" in rl_config, "UR5e-sized scene section is missing"
for entry in ("obstacle_localizer", "action_adapter"):
    assert entry in setup_text, f"console script {entry} is not registered"
launch_text = (root / "launch/vision_bringup.launch.py").read_text()
for token in ("obstacle_localizer", "action_adapter", "adapter_enable_command"):
    assert token in launch_text, f"launch does not start/declare {token}"
print("LOCALIZER / ADAPTER WIRING OK")

for entry in ("metrics_node", "dataset_capture"):
    assert entry in setup_text, f"console script {entry} is not registered"
for token in ("enable_metrics", "adapter_auto_start", "depth_fallback_enabled"):
    assert token in launch_text, f"launch does not declare {token}"
assert 'rpy="3.14159265359 0 0"' in xacro_text, "camera mount must keep the image upright"
assert "initial_positions" in xacro_text, "robot must spawn in a well-conditioned pose"
adapter_text = (root / "ur_vision_avoidance/action_adapter_node.py").read_text()
assert "point.velocities =" not in adapter_text, "JTC rejects a last point with non-zero velocity"
docker_text = (root / "Dockerfile").read_text()
assert "PYTHONPATH=/opt/venvs/ros" in docker_text, "console scripts cannot import Ultralytics"
print("GAZEBO INTEGRATION FIXES OK")

import py_compile

for source in sorted((root / "ur_vision_avoidance").glob("*.py")):
    py_compile.compile(str(source), doraise=True)
print("ALL MODULES COMPILE")
