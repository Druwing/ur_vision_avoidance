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
