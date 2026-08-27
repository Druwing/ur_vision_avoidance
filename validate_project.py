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
