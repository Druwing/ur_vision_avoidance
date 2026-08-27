# Deep-learning obstacle avoidance for a UR cobot in ROS 2 Jazzy and Gazebo Harmonic

**Author:** Manus AI  
**Target platform:** Ubuntu 24.04 Noble, 64-bit  
**ROS distribution:** ROS 2 Jazzy Jalisco  
**Simulator:** Gazebo Harmonic  
**Robot:** Universal Robots arm, with `ur5e` used as the default example  
**Perception model:** Ultralytics YOLO26 nano, replaceable by another supported checkpoint or a custom `best.pt`

## 1. What this tutorial builds

The finished simulation contains a Universal Robots manipulator in Gazebo Harmonic, a synthetic RGB-D camera rigidly attached to the robot `tool0` frame, a Gazebo-to-ROS 2 image bridge, a YOLO inference node, and a high-level avoidance supervisor. The detector subscribes to RGB and depth images, detects configured obstacle classes, estimates the nearest depth at each detection, and publishes an obstacle request. The supervisor can optionally send one small, deliberately conservative shoulder-pan retreat to the simulated scaled joint trajectory controller.

> **Important safety boundary:** this is a research and simulation starter. The perception result is not a safety-rated protective stop, and the example supervisor is not suitable for commanding a real robot. A real deployment requires a formal risk assessment, validated robot limits, safety-rated sensing or protective measures, collision monitoring, a certified or otherwise validated control architecture, and testing under the relevant Universal Robots safety procedures.

The architecture is intentionally split into perception and motion control. The machine-learning node does not directly actuate the robot; it publishes a high-level result that a separate supervisor can filter, rate-limit, time out, and send to a planner or controller. This separation makes it easier to replace YOLO with a custom model and to replace the simple retreat with MoveIt 2, MoveIt Servo, a velocity controller, or a validated reactive planner.

| Component | Role | Main ROS 2 interface |
|---|---|---|
| `ur_simulation_gz` | Spawns and controls the simulated UR arm in GZ Sim | `ur_sim_control.launch.py` |
| Custom UR Xacro | Adds the RGB-D camera to `tool0` | `robot_description` |
| `ros_gz_bridge` | Converts Gazebo Transport images into ROS 2 messages | `sensor_msgs/msg/Image` |
| `vision_node` | Runs YOLO and reads aligned depth | `/vision/detections`, `/obstacle/stop` |
| `avoidance_supervisor` | Optional simulation-only retreat | `FollowJointTrajectory` action |
| RViz 2 and `rqt_image_view` | Visual inspection and debugging | `/camera/image_raw`, `/vision/annotated` |

## 2. Why this software combination is appropriate

ROS 2 Jazzy and Gazebo Harmonic are the recommended LTS pairing in the current Gazebo compatibility documentation. Gazebo's ROS installation guidance states that Jazzy uses the Harmonic vendor packages and that the default pairing can be installed through the ROS package repository.[1] The Universal Robots simulation package provides a Jazzy branch and documents `ur_sim_control.launch.py` for Gazebo/controllers and `ur_sim_moveit.launch.py` for Gazebo together with MoveIt 2.[2] [3]

The camera path uses `ros_gz_bridge`, not the old Gazebo Classic ROS camera plugin. The bridge supports Gazebo images and camera information as `sensor_msgs/msg/Image` and `sensor_msgs/msg/CameraInfo`, and its documented syntax maps a Gazebo topic to a ROS topic with a YAML file or a parameter-bridge command.[4] [5]

Ultralytics' current ROS 2 quickstart uses `rclpy`, `cv_bridge`, `sensor_msgs/msg/Image`, and `qos_profile_sensor_data`. It recommends loading the model once and reusing it in the image callback.[6] The current prediction API returns `Results` objects whose `boxes.xyxy`, `boxes.conf`, `boxes.cls`, and `names` fields expose the bounding boxes, confidence, class IDs, and class names used by the example node.[7]

## 3. Prerequisites and safety preparation

Use a clean Ubuntu 24.04 installation or a dedicated development machine. Enable hardware-accelerated graphics if possible, because Gazebo's camera rendering and GUI can be expensive. The tutorial works on a CPU, but YOLO inference and Gazebo rendering may run slowly without a GPU. A practical starting point is at least 16 GB of RAM, a modern multi-core CPU, and an NVIDIA GPU or integrated graphics driver that can run the Gazebo GUI smoothly.

Do not connect this tutorial to a physical UR robot during the first stages. The package contains an action client, and the same ROS action naming convention is used by real UR deployments. Keep the entire project in simulation until perception, TF, depth alignment, latency, planner behavior, and stop behavior have been measured and reviewed.

| Requirement | Check command | Expected result |
|---|---|---|
| Ubuntu version | `lsb_release -ds` | `Ubuntu 24.04...` |
| Architecture | `dpkg --print-architecture` | `amd64` or a supported target architecture |
| Graphics | `glxinfo -B` | A working OpenGL renderer; install `mesa-utils` if needed |
| Git | `git --version` | A recent Git version |
| Network | `curl -I https://github.com` | HTTP response, subject to local proxy/network policy |

## 4. Install ROS 2 Jazzy and Gazebo Harmonic

The following sequence follows the ROS Debian-package approach. ROS 2 Jazzy is intended for Ubuntu 24.04 Noble. If your machine already has another ROS distribution sourced automatically from `.bashrc`, remove or comment out that older source line before proceeding; mixing ROS distributions in one shell is a common source of package and Python errors.

First install system prerequisites and enable the Ubuntu `universe` repository.

```bash
sudo apt update
sudo apt install -y software-properties-common curl gnupg lsb-release \
  git build-essential python3-pip python3-venv \
  python3-colcon-common-extensions python3-rosdep python3-vcstool
sudo add-apt-repository universe
```

Add the ROS 2 package signing key and repository. The command deliberately uses the codename reported by Ubuntu rather than hard-coding a different distribution.

```bash
sudo curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
| sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null

sudo apt update
```

Install the ROS desktop distribution, the Jazzy-to-Gazebo pairing, the Universal Robots binary packages, and the ROS interfaces needed by this tutorial.

```bash
sudo apt install -y \
  ros-jazzy-desktop \
  ros-dev-tools \
  ros-jazzy-ros-gz \
  ros-jazzy-ur \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-ros2controlcli \
  ros-jazzy-xacro \
  ros-jazzy-cv-bridge \
  ros-jazzy-rqt-image-view
```

If your mirror does not expose the `ros-jazzy-ros-gz` metapackage, install its principal packages directly and then retry the package query. The exact package split may change as ROS package repositories are synchronized.

```bash
sudo apt install -y ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge
```

Initialize and update `rosdep`. If another developer on the machine has already run `rosdep init`, the first command may report that the file exists; that is harmless.

```bash
sudo rosdep init 2>/dev/null || true
rosdep update
```

Source ROS 2 in the current terminal and, if desired, add it to future interactive shells.

```bash
source /opt/ros/jazzy/setup.bash

grep -qxF 'source /opt/ros/jazzy/setup.bash' ~/.bashrc || \
  echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
```

Perform a basic installation check before cloning any source repositories.

```bash
ros2 --version
ros2 pkg prefix ros_gz_sim
ros2 pkg prefix ros_gz_bridge
ros2 pkg prefix ur_description
ros2 pkg prefix ur_robot_driver
gz sim --version
```

Gazebo's current command-line interface uses `gz sim`. The official getting-started page uses `gz sim shapes.sdf` for a sample world and documents `-s` for a server-only/headless launch.[8] Test the simulator independently.

```bash
gz sim shapes.sdf -v 3
```

Close the test window with `Ctrl+C` after verifying that the GUI opens. If `gz` is not found, source `/opt/ros/jazzy/setup.bash` again and check that `ros-jazzy-ros-gz` is installed.

## 5. Obtain and build the Universal Robots GZ simulation package

Universal Robots documents binary installation of the `ur` meta-package with `sudo apt-get install ros-${ROS_DISTRO}-ur` and provides a source repository for the GZ simulation examples.[2] [3] Use a dedicated workspace so that the custom perception package and the simulation package can be rebuilt independently of `/opt/ros/jazzy`.

```bash
source /opt/ros/jazzy/setup.bash

export COLCON_WS=$HOME/workspaces/ur_gz
mkdir -p "$COLCON_WS/src"
cd "$COLCON_WS"

git clone --depth 1 --branch jazzy \
  https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation.git \
  src/ur_simulation_gz

rosdep update
rosdep install --ignore-src --from-paths src -r -y
colcon build --symlink-install
source install/setup.bash
```

The official repository documents the same colcon workflow and its Jazzy branch has Ubuntu Noble buildfarm coverage.[3] Verify the package and inspect the launch arguments.

```bash
ros2 pkg prefix ur_simulation_gz
ros2 launch ur_simulation_gz ur_sim_control.launch.py --show-args
```

Start the standard UR simulation once, without the camera, to verify controllers and RViz before introducing custom files.

```bash
ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5e
```

In a second terminal, after sourcing both workspaces, inspect controllers.

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/workspaces/ur_gz/install/setup.bash"
ros2 control list_controllers
```

You should see at least `joint_state_broadcaster` and the initially active `scaled_joint_trajectory_controller`. The official UR documentation uses the action endpoint `scaled_joint_trajectory_controller/follow_joint_trajectory` and explains that the scaled controller accounts for speed scaling.[9] [10]

The standard simulation launch can also start MoveIt 2 when you want collision-aware planning rather than the small retreat in this tutorial.

```bash
ros2 launch ur_simulation_gz ur_sim_moveit.launch.py ur_type:=ur5e
```

## 6. Install the deep-learning model runtime

This tutorial uses the current Ultralytics Python package and the `yolo26n.pt` pretrained nano detector shown in the current Ultralytics documentation.[6] [7] The nano model is selected because a CPU-friendly baseline is more useful for an initial ROS 2 integration than a large, slow model. It is not automatically the best model for your obstacle class, lighting, camera placement, or latency target.

Install the Python package into the system interpreter used by the ROS 2 executable. Ubuntu 24.04 may enforce the externally managed environment policy; `--break-system-packages` is explicit and should be used only on a dedicated robotics development installation. If you prefer stronger isolation, create a virtual environment and ensure that the ROS 2 node is launched with that interpreter; do not mix an unconfigured virtual environment with the system ROS Python path.

```bash
sudo pip3 install --break-system-packages ultralytics
```

Download the checkpoint by loading it once. Ultralytics downloads official weights automatically on first use and caches them locally.[7]

```bash
python3 -c "from ultralytics import YOLO; m=YOLO('yolo26n.pt'); print(m.names)"
```

If the installed package predates the current YOLO26 release, use a checkpoint supported by that package, such as `yolo11n.pt`, and pass it at launch time:

```bash
ros2 launch ur_vision_avoidance vision_bringup.launch.py model:=yolo11n.pt
```

The code accepts a custom checkpoint without modification. For example:

```bash
ros2 launch ur_vision_avoidance vision_bringup.launch.py \
  model:=/absolute/path/to/best.pt
```

| Model input | Meaning | Recommended use |
|---|---|---|
| `yolo26n.pt` | Official pretrained nano detector | First CPU/GPU integration test |
| `yolo26n-seg.pt` | Instance-segmentation checkpoint | When a mask is more useful than a box |
| `yolo11n.pt` | Fallback for an older Ultralytics installation | Compatibility fallback |
| `/path/to/best.pt` | Your custom trained weights | Production research after dataset validation |

The pretrained detector is trained for general classes. The included Gazebo world adds an OpenRobotics Fuel Chair because `chair` is a COCO-recognizable class. A plain blue geometric box is also included for depth and camera testing, but the general pretrained detector is not guaranteed to label it as an obstacle. In a serious experiment, train on the actual obstacle categories and camera viewpoint instead of relying on a generic checkpoint.

## 7. Add the custom camera and perception package

The attached companion folder is a complete ROS 2 Python package named `ur_vision_avoidance`. Put that folder under the source directory of the workspace. If you downloaded the package archive into `~/Downloads`, the copy operation is conceptually as follows; adapt the source path to the file location on your machine.

```bash
source /opt/ros/jazzy/setup.bash
export COLCON_WS=$HOME/workspaces/ur_gz
mkdir -p "$COLCON_WS/src"
cp -a /path/to/ur_vision_avoidance "$COLCON_WS/src/ur_vision_avoidance"
```

The important files are shown below.

| File | Purpose |
|---|---|
| `urdf/ur_gz_camera.urdf.xacro` | Reuses the official UR description and adds a fixed RGB-D sensor to `tool0` |
| `worlds/obstacle_world.sdf` | Adds Gazebo systems, ground, a Fuel Chair, and a box obstacle |
| `config/bridge.yaml` | Maps Gazebo camera topics into ROS 2 image topics |
| `ur_vision_avoidance/vision_node.py` | YOLO inference, depth lookup, annotated image, and obstacle topics |
| `ur_vision_avoidance/avoidance_supervisor.py` | Optional simulation-only retreat action client |
| `launch/vision_bringup.launch.py` | Starts the simulation, bridge, detector, and optional supervisor |

The custom Xacro does not edit the installed Universal Robots package. Instead, it uses the official `ur_robot` macro, adds `camera_link` and `camera_optical_frame`, attaches `camera_link` to `tool0` with a fixed joint, and passes the custom description to the official UR simulation launcher through its `description_file` argument. This is the preferred pattern because upstream packages remain unmodified and can be updated.

Build the workspace again.

```bash
cd "$COLCON_WS"
rosdep install --ignore-src --from-paths src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 pkg prefix ur_vision_avoidance
```

## 8. Understand the camera and bridge topics

The Gazebo Harmonic sensor uses an `rgbd_camera` sensor with a topic prefix `camera`. The official Gazebo sensor examples use this structure and produce image and depth topics below that prefix.[11] The bridge configuration maps those Gazebo topics to stable ROS names, so the perception node does not need to know the full world/model/entity path used internally by Gazebo.

| Gazebo Transport topic | ROS 2 topic | Message type | Use |
|---|---|---|---|
| `/camera/image` | `/camera/image_raw` | `sensor_msgs/msg/Image` | RGB input to YOLO |
| `/camera/depth_image` | `/camera/depth_image_raw` | `sensor_msgs/msg/Image` | Range estimate at detection center |
| `/camera/camera_info` | `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | Intrinsics for later 3-D projection |
| `/clock` | `/clock` | `rosgraph_msgs/msg/Clock` | Simulation time; started by the UR launch file |
| `/vision/annotated` | `/vision/annotated` | `sensor_msgs/msg/Image` | YOLO boxes for inspection |
| `/vision/detections` | `/vision/detections` | `std_msgs/msg/String` | JSON detection summary |
| `/obstacle/stop` | `/obstacle/stop` | `std_msgs/msg/Bool` | High-level stop request |
| `/obstacle/nearest_distance` | `/obstacle/nearest_distance` | `std_msgs/msg/Float32` | Nearest configured detection in metres |
| `/obstacle/lateral_offset` | `/obstacle/lateral_offset` | `std_msgs/msg/Float32` | Normalized horizontal offset, roughly -0.5 to +0.5 |

The bridge uses a sensor-data QoS profile. This matters because camera publishers commonly use best-effort sensor QoS; a reliable subscriber can otherwise fail to match the publisher. If a topic does not appear, first inspect Gazebo's actual names instead of guessing.

```bash
gz topic -l | grep -E 'camera|image|depth'
ros2 topic list | grep -E 'camera|vision|obstacle'
ros2 topic info /camera/image_raw -v
```

The bridge can also be started manually for troubleshooting. The documented parameter-bridge form is `/TOPIC@ROS_MSG@GZ_MSG`; a left bracket makes a Gazebo-to-ROS bridge.[4]

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /camera/image@sensor_msgs/msg/Image[gz.msgs.Image
```

Do not run a second `/clock` bridge when using `ur_sim_control.launch.py`; that official launcher already starts the Gazebo-to-ROS clock bridge.

## 9. Launch the complete simulation

Source the base ROS installation and the workspace in every terminal.

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/workspaces/ur_gz/install/setup.bash"
```

Start the simulation with avoidance disabled. The detector still publishes its outputs, but no joint trajectory action is sent.

```bash
ros2 launch ur_vision_avoidance vision_bringup.launch.py \
  ur_type:=ur5e \
  model:=yolo26n.pt \
  enable_avoidance:=false
```

The first model invocation may download the checkpoint and briefly delay image processing. Gazebo should open with the UR arm, the custom world, and RViz. If the chair is not visible, adjust the chair pose in `worlds/obstacle_world.sdf`, adjust the camera mount pose in `urdf/ur_gz_camera.urdf.xacro`, or use RViz/MoveIt to place the arm so the end-effector camera points toward the obstacle.

In a second terminal, view the annotated image.

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/workspaces/ur_gz/install/setup.bash"
ros2 run rqt_image_view rqt_image_view /vision/annotated
```

In a third terminal, inspect the JSON detections and stop request.

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/workspaces/ur_gz/install/setup.bash"
ros2 topic echo /vision/detections
ros2 topic echo /obstacle/stop
ros2 topic echo /obstacle/nearest_distance
```

The detector publishes `false` until a configured class is detected inside the central image corridor and the aligned depth at its center is below `stop_distance_m`, which defaults to 0.80 m. It requires fresh depth by default. This conservative default is intentional: RGB detection gives a 2-D location, while depth or a calibrated stereo system is needed for a range estimate.

## 10. Test the simulation-only retreat

Before enabling the model-driven supervisor, test the action endpoint and the state topics. Confirm that the controller is active.

```bash
ros2 control list_controllers
ros2 action list | grep follow_joint_trajectory
ros2 topic echo /joint_states --once
```

To run the supervisor with the default conservative settings, restart the combined launch with `enable_avoidance:=true`.

```bash
ros2 launch ur_vision_avoidance vision_bringup.launch.py \
  ur_type:=ur5e \
  model:=yolo26n.pt \
  enable_avoidance:=true
```

The supervisor only reacts to a rising edge of `/obstacle/stop`, checks that joint states are fresh, limits the command frequency, checks the action server, and sends a single small shoulder-pan retreat. It publishes the outgoing `JointTrajectory` on `/avoidance/command` for inspection and sends the same trajectory to `/scaled_joint_trajectory_controller/follow_joint_trajectory`.

For a deterministic plumbing test that does not depend on YOLO seeing the chair, publish a high-level obstacle request after the supervisor is running. This is still simulation-only.

```bash
ros2 topic pub --once /obstacle/lateral_offset std_msgs/msg/Float32 "{data: 0.25}"
ros2 topic pub --once /obstacle/stop std_msgs/msg/Bool "{data: true}"
```

A positive lateral offset is interpreted as an obstacle on the right and causes a negative shoulder-pan retreat in the sample. Camera and robot sign conventions vary, so verify the sign visually in Gazebo and invert it in the supervisor if necessary. This one-joint retreat is not obstacle avoidance in the robotics research sense; it is only a small end-to-end test of perception-to-action plumbing.

## 11. What the detector code is doing

The perception node loads the YOLO model once in its constructor, subscribes to RGB and depth topics with `qos_profile_sensor_data`, and converts images through `cv_bridge`. For every RGB frame it reads the detected boxes, filters to target classes, computes the horizontal center, samples a small depth patch around the corresponding pixel, and publishes a JSON summary. The JSON contains class name, confidence, box, distance, path membership, and the close flag.

The distance conversion recognizes two common ROS depth encodings. For `16UC1`, it converts millimetres to metres by multiplying by 0.001. For floating-point depth it assumes the values are already metres. The RGB and depth images must be aligned and synchronized for this center-pixel lookup to be meaningful. A production implementation should use message filters, camera intrinsics, an ROI median or percentile, and explicit calibration checks rather than relying only on a center patch.

The `require_depth` policy is enabled by default. Set it to `false` only to debug 2-D detection, and understand that the node will then not make a reliable distance decision. A real protective behavior should fail safe on missing, stale, malformed, or implausible sensor data rather than silently treating them as clear space.

Useful runtime parameter overrides are:

```bash
ros2 run ur_vision_avoidance vision_node --ros-args \
  -p model:=/absolute/path/to/best.pt \
  -p confidence:=0.55 \
  -p stop_distance_m:=0.60 \
  -p require_depth:=true
```

## 12. Obtain a model that actually matches your obstacles

A general pretrained model is a convenient integration test, not a substitute for a dataset. If your object is a custom fixture, workpiece, cable, pallet, person wearing protective equipment, or a partially occluded obstacle, collect images from the actual camera geometry and annotate the relevant classes.

A practical development cycle is to record representative RGB-D data, export RGB frames, label bounding boxes or masks, split the dataset into training/validation/test sets by scene rather than by adjacent frames, train a small baseline, and measure false negatives at the planned robot speed. For a moving arm, record the same workspace from multiple tool poses and lighting conditions. Add empty-scene frames, glare, motion blur, partial occlusion, and objects at the edge of the camera field of view.

The standard YOLO detection dataset uses one text label file per image. A typical directory looks like this:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

A minimal `data.yaml` for one custom class is:

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
names:
  0: obstacle
```

Train a small baseline on the workstation or a GPU machine. Training on CPU is possible but usually too slow for productive iteration.

```bash
yolo detect train \
  model=yolo26n.pt \
  data=/absolute/path/to/dataset/data.yaml \
  imgsz=640 \
  epochs=100 \
  batch=16 \
  device=0
```

Validate and export after training.

```bash
yolo detect val \
  model=runs/detect/train/weights/best.pt \
  data=/absolute/path/to/dataset/data.yaml \
  device=0

yolo export \
  model=runs/detect/train/weights/best.pt \
  format=onnx \
  imgsz=640
```

Then use the custom model in ROS 2.

```bash
ros2 launch ur_vision_avoidance vision_bringup.launch.py \
  model:=/absolute/path/to/runs/detect/train/weights/best.pt
```

For obstacle avoidance, accuracy must be evaluated in terms of system risk, not only mean average precision. Track false negatives at the distances and speeds that matter, end-to-end image-to-command latency, worst-case inference time, stale-frame behavior, detection persistence, and the behavior when the camera is blocked or the bridge stops publishing.

## 13. Add 3-D obstacle points and a real planner next

The starter node samples depth at a bounding-box center. The next useful improvement is to use `CameraInfo` and the optical-frame transform to back-project a pixel and range into a 3-D point. For a pinhole camera with focal lengths `fx`, `fy` and principal point `cx`, `cy`, a depth value `Z` gives approximately `X=(u-cx)Z/fx` and `Y=(v-cy)Z/fy`. Transform that point from `camera_optical_frame` to `base_link` using TF2, then publish a `geometry_msgs/msg/PointStamped` or a `vision_msgs/msg/Detection3DArray`.

At that point the control architecture should stop being a hand-written shoulder-pan retreat. Use a motion planner or a reactive Cartesian controller that checks the complete robot body, end-effector, carried tool, and obstacle geometry. MoveIt 2 can plan around collision objects; MoveIt Servo can provide a reactive command path but still requires careful singularity, joint-limit, collision, and timeout handling. The best choice depends on whether your obstacle is static or moving, whether the robot is already executing a trajectory, and whether you need guaranteed stop behavior or merely experimental replanning.

| Development stage | Perception output | Motion response | Appropriate status |
|---|---|---|---|
| Stage 0 | Annotated RGB image | None | Camera and bridge debugging |
| Stage 1 | 2-D boxes plus depth | High-level stop request | This tutorial |
| Stage 2 | 3-D points in `base_link` | Replan with MoveIt 2 | Research prototype |
| Stage 3 | Tracked obstacles with uncertainty | Validated reactive planner | Advanced research |
| Stage 4 | Safety-rated sensing and validated control | Protective stop / separation monitoring | Real deployment only after formal validation |

## 14. Troubleshooting

### Gazebo starts but no camera topics appear

First confirm that Gazebo is running rather than paused and inspect Gazebo Transport topics.

```bash
gz topic -l | grep -E 'camera|image|depth'
```

If the topics are absent, inspect the Gazebo log for `rgbd_camera`, rendering, or sensor-system errors. Verify that the custom Xacro has been passed through `description_file` and that the world includes `gz-sim-sensors-system`. If the camera topics have a different prefix, update `config/bridge.yaml` to match the exact Gazebo names.

### ROS topics exist but `rqt_image_view` is blank

Check that the bridge is running and that the topic type is correct.

```bash
ros2 topic info /camera/image_raw -v
ros2 topic hz /camera/image_raw
```

A QoS mismatch is common. The provided bridge and node use sensor-data QoS. Check that the simulator is not paused, verify the `frame_id`, and inspect the terminal where `parameter_bridge` is running.

### YOLO sees nothing

Display `/vision/annotated` first. If the image is blank or wrongly oriented, fix the camera mount or bridge. If the image is correct but the model does not detect the chair, lower the confidence threshold temporarily, move the arm so the chair occupies more pixels, use a different pretrained checkpoint, or train a custom model. A geometric box is intentionally not guaranteed to be recognized by a general COCO detector.

### Depth is always `-1.0`

Verify `/camera/depth_image_raw`, its encoding, and its rate.

```bash
ros2 topic echo /camera/depth_image_raw --once
ros2 topic hz /camera/depth_image_raw
```

The sample requires an RGB/depth timestamp difference below `depth_timeout_s`. If RGB and depth are not synchronized, use an approximate-time message filter and verify camera alignment. If the depth message is `32FC1`, values are assumed to be metres. If it is `16UC1`, values are converted from millimetres.

### The controller action is unavailable

Check the controller list and action names.

```bash
ros2 control list_controllers
ros2 action list | grep follow_joint_trajectory
```

If `scaled_joint_trajectory_controller` is not active, start the simulation with `initial_joint_controller:=scaled_joint_trajectory_controller` or launch the controller using the package's documented mechanisms. Do not send goals until the correct simulated controller is active.

### The custom Xacro fails to process

Run Xacro directly to expose the first XML or package-resolution error.

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/workspaces/ur_gz/install/setup.bash"
ros2 run xacro xacro \
  "$HOME/workspaces/ur_gz/src/ur_vision_avoidance/urdf/ur_gz_camera.urdf.xacro" \
  ur_type:=ur5e > /tmp/ur_camera.urdf
```

Check that `ur_description` and `ur_simulation_gz` resolve with `ros2 pkg prefix`, and ensure that the `ur_type` is one supported by the installed Universal Robots package.

## 15. Recommended experiment log

For reproducible deep-learning and robotics research, log the commit hash of the simulation repositories, the ROS/Gazebo versions, the model checkpoint hash, the camera resolution and update rate, the bridge QoS, the robot initial joint configuration, the obstacle pose, the simulated clock rate, and the detector and controller parameters. Record frame timestamps and command timestamps so that latency and stale-data behavior can be measured rather than guessed.

| Item to record | Example |
|---|---|
| ROS distribution | Jazzy |
| Gazebo release | Harmonic |
| UR model | `ur5e` |
| Camera | 640×480 RGB-D at 15 Hz |
| Checkpoint | `yolo26n.pt` or SHA-256 of custom `best.pt` |
| Confidence threshold | 0.45 |
| Stop distance | 0.80 m |
| Depth timeout | 0.50 s |
| Robot controller | `scaled_joint_trajectory_controller` |
| Supervisor mode | Disabled during baseline; enabled only in simulation test |

## References

[1]: https://gazebosim.org/docs/harmonic/ros_installation/ "Gazebo: Installing Gazebo with ROS"

[2]: https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/installation/installation.html "Universal Robots: Installation of the ur_robot_driver"

[3]: https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation "Universal Robots: Universal_Robots_ROS2_GZ_Simulation"

[4]: https://gazebosim.org/docs/latest/ros2_integration/ "Gazebo: Use ROS 2 to interact with Gazebo"

[5]: https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_bridge/README.md "ros_gz_bridge README and supported message mappings"

[6]: https://docs.ultralytics.com/guides/ros-quickstart "Ultralytics: ROS quickstart guide"

[7]: https://docs.ultralytics.com/modes/predict "Ultralytics: Model Prediction with YOLO"

[8]: https://gazebosim.org/docs/harmonic/getstarted/ "Gazebo Harmonic: Getting Started"

[9]: https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html "ros2_control Jazzy: Joint Trajectory Controller"

[10]: https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_controllers/doc/index.html "Universal Robots: ur_controllers"

[11]: https://github.com/gazebosim/gz-sim/blob/main/examples/worlds/sensors_demo.sdf "Gazebo Sim: Official sensors_demo.sdf"

[12]: https://fuel.gazebosim.org/1.0/OpenRobotics/models/Chair "Gazebo Fuel: OpenRobotics Chair model"
