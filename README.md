# UR Vision Avoidance

## Project overview

UR Vision Avoidance is a ROS 2 Jazzy and Gazebo Harmonic research prototype for attaching an RGB-D camera to a Universal Robots arm, detecting obstacles with Ultralytics YOLO, estimating their distance from depth data, and experimenting with learned avoidance actions.

The package is designed for perception and integration experiments. It is **not a safety-rated collision-avoidance system**, protective-stop implementation, or production robot controller. The repository now includes a fixed-scene reinforcement-learning starter scaffold, but no trained policy is bundled. Augmented randomized domain (ARD) is intentionally disabled in this stage. The supplied legacy motion response remains a deterministic shoulder-pan retreat.

The default simulation uses a UR5e model, a camera mounted on the tool frame, an RGB-D camera stream, a chair from Gazebo Fuel, and a geometric box obstacle. The bundled `yolo26n.pt` checkpoint is a general pretrained model. It can detect COCO classes such as `person` and `chair`, but it is not guaranteed to recognize every geometric obstacle in the sample world.

## Architecture

The process is divided into simulation, transport, perception, and optional command layers:

```mermaid
flowchart LR
    A[Gazebo Harmonic\nUR5e + camera + obstacles] --> B[Gazebo camera topics\n/camera/image\n/camera/depth_image]
    B --> C[ros_gz_bridge\nconfig/bridge.yaml]
    C --> D[ROS 2 RGB-D topics\n/camera/image_raw\n/camera/depth_image_raw]
    D --> E[vision_node\nYOLO + depth sampling]
    E --> F[/vision/annotated]
    E --> G[/vision/detections]
    E --> H[/obstacle/stop]
    E --> I[/obstacle/nearest_distance]
    E --> J[/obstacle/lateral_offset]
    H --> K[avoidance_supervisor\noptional, disabled by default]
    J --> K
    L[/joint_states/] --> K
    K --> M[FollowJointTrajectory action\nscaled_joint_trajectory_controller]
    M --> A
    E --> N[rl_observation_node\nstructured JSON state]
    L --> N
    N --> O[rl_policy_node\noptional PPO inference]
    O --> P[/rl/action\nbounded velocity offset]
```

### Components

| Component | Responsibility |
|---|---|
| `urdf/ur_gz_camera.urdf.xacro` | Expands the UR5e description, attaches `camera_link` and `camera_optical_frame`, adds the Gazebo RGB-D sensor, and loads simulated ROS 2 control. |
| `worlds/obstacle_world.sdf` | Starts Gazebo physics, rendering, sensor processing, ground plane, chair, and box obstacle. |
| `config/bridge.yaml` | Bridges Gazebo RGB, depth, and camera-info topics into ROS 2. |
| `vision_node` | Runs YOLO on each RGB frame, samples a depth patch at each detection center, and publishes detection and obstacle topics. |
| `avoidance_supervisor` | Listens for an obstacle request and, only when explicitly enabled, sends one bounded shoulder-pan trajectory to the simulated controller. |
| `fixed_scene_env.py` | Provides a deterministic Gymnasium training environment for one fixed obstacle and one nominal path. ARD is disabled. |
| `train_fixed_scene.py` | Trains an optional PPO policy against the fixed-scene environment. |
| `rl_observation_node` | Combines detections and joint state into a versioned JSON observation topic. |
| `rl_policy_node` | Loads a PPO checkpoint and publishes bounded actions. It does not directly command the robot. |
| `launch/vision_bringup.launch.py` | Starts Gazebo, robot state publication, controllers, camera bridge, perception, and optional avoidance. |
| `yolo26n.pt` | Bundled Ultralytics checkpoint. A custom `best.pt` can be supplied at launch time. |

## Data flow and decisions

The camera publishes RGB and depth frames in Gazebo. `ros_gz_bridge` converts them to ROS 2 sensor messages using sensor-data QoS. `vision_node` performs inference on the RGB image and filters detections against its configured `target_classes` list.

For each retained detection, the node computes the bounding-box center and samples a median depth patch around that pixel. A detection is considered close when its estimated range is at or below `stop_distance_m` and its center lies inside the horizontal path region from 20% to 80% of the image width. When `require_depth` is true, stale or missing depth cannot produce a close-obstacle request.

The supervisor is disabled by default. When enabled, it reacts to a rising edge on `/obstacle/stop`, checks that joint states are fresh, checks that the trajectory action server is available, and sends a bounded shoulder-pan retreat. This behavior is intended for simulation experiments only.

The RL starter is also disabled by default. When enabled, the observation adapter publishes `/rl/observation` and the optional policy node publishes `/rl/action`. The action is an interface artifact for the starter stage; it is not connected to the UR controller. ARD is not used by the training environment, configuration, or launch mode.

## Prerequisites

A full runtime requires the following environment:

- Ubuntu with ROS 2 Jazzy.
- Gazebo Harmonic and the `ros_gz` integration packages.
- `ros2_control` and `ros2_controllers`.
- The Universal Robots ROS 2 Gazebo simulation package.
- Xacro, `cv_bridge`, OpenCV, NumPy, and Ultralytics.
- A graphical display for Gazebo GUI and RViz, unless headless mode is selected.

The included Dockerfile installs the ROS packages and Ultralytics. Docker and Docker Compose are optional if the dependencies are already installed on the host.

## Installation with Docker

From the project root, build the image:

```bash
docker compose build
```

Start the development container:

```bash
docker compose up
```

The Compose configuration uses host networking for ROS 2 discovery and mounts the project at `/ros2_ws/src/ur_vision_avoidance`. It also exposes the host display and devices for simulation. On systems that require explicit X11 permission, allow the container to connect before starting Gazebo:

```bash
xhost +local:root
```

After use, restore the display access policy if appropriate for your system:

```bash
xhost -local:root
```

Inside the container, the workspace is built and sourced automatically by the Compose command. To open a shell instead, run:

```bash
docker compose run --rm ros2jazzy-gazebo bash
source /opt/ros/jazzy/setup.bash
cd /ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Installation on a ROS 2 host

Create a workspace and place this package under `src`:

```bash
mkdir -p ~/ur_vision_ws/src
cd ~/ur_vision_ws/src
git clone <your-project-location> ur_vision_avoidance
```

Clone or install the Universal Robots Gazebo simulation package in the same workspace. The launch file expects the packages `ur_simulation_gz`, `ur_description`, and `ros_gz_sim` to be discoverable.

Install package dependencies and build:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ur_vision_ws
rosdep update
rosdep install --ignore-src --from-paths src -r -y
colcon build --symlink-install
source install/setup.bash
```

Run the package-level static checks:

```bash
cd ~/ur_vision_ws/src/ur_vision_avoidance
python3 validate_project.py
python3 -m compileall -q .
```

## Launch the simulation

The default launch starts Gazebo, the UR5e, the RGB-D camera bridge, YOLO perception, and the robot controllers. The avoidance supervisor remains disabled:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ur_vision_ws/install/setup.bash
ros2 launch ur_vision_avoidance vision_bringup.launch.py
```

With Docker Compose, use the same command inside the running container:

```bash
ros2 launch ur_vision_avoidance vision_bringup.launch.py
```

To enable the simulation-only retreat supervisor:

```bash
ros2 launch ur_vision_avoidance vision_bringup.launch.py enable_avoidance:=true
```

Do not enable this mode on a physical robot. The supplied supervisor is not a validated motion planner or safety layer.

### Useful launch arguments

| Argument | Default | Purpose |
|---|---:|---|
| `ur_type` | `ur5e` | Universal Robots model supported by the installed simulation package. |
| `model` | Installed `yolo26n.pt` | Ultralytics checkpoint or absolute path to a custom checkpoint. |
| `enable_avoidance` | `false` | Enables the simulation-only retreat supervisor. |
| `gazebo_gui` | `true` | Set to `false` for server-only Gazebo execution. |
| `launch_rviz` | `true` | Starts RViz after the joint-state broadcaster. |
| `activate_joint_controller` | `true` | Starts the configured initial joint controller. |
| `initial_joint_controller` | `scaled_joint_trajectory_controller` | Controller used by the supervisor action client. |
| `world_file` | Bundled obstacle world | Alternate SDF world path. |

Example with a custom checkpoint and headless Gazebo:

```bash
ros2 launch ur_vision_avoidance vision_bringup.launch.py \
  model:=/absolute/path/to/best.pt \
  gazebo_gui:=false \
  launch_rviz:=false
```

## Inspect the running system

Use these commands in a second terminal after sourcing the workspace:

```bash
ros2 node list
ros2 topic list
ros2 topic hz /camera/image_raw
ros2 topic hz /camera/depth_image_raw
ros2 topic echo /obstacle/stop --once
ros2 topic echo /obstacle/nearest_distance --once
ros2 topic echo /vision/detections --once
ros2 action list | grep follow_joint_trajectory
ros2 control list_controllers
```

Open the annotated camera stream in RViz or `rqt_image_view`:

```bash
ros2 run rqt_image_view rqt_image_view /vision/annotated
```

The main perception topics are:

| Topic | Type | Meaning |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/msg/Image` | RGB image bridged from Gazebo. |
| `/camera/depth_image_raw` | `sensor_msgs/msg/Image` | Depth image bridged from Gazebo. |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | Camera calibration information. |
| `/vision/annotated` | `sensor_msgs/msg/Image` | RGB image with YOLO annotations. |
| `/vision/detections` | `std_msgs/msg/String` | JSON-formatted detections and range estimates. |
| `/obstacle/stop` | `std_msgs/msg/Bool` | High-level close-obstacle request. |
| `/obstacle/nearest_distance` | `std_msgs/msg/Float32` | Nearest valid sampled detection range in metres, or `-1.0`. |
| `/obstacle/lateral_offset` | `std_msgs/msg/Float32` | Nearest detection center offset from the image center. |
| `/joint_states` | `sensor_msgs/msg/JointState` | Current robot joint state. |
| `/avoidance/command` | `trajectory_msgs/msg/JointTrajectory` | Debug copy of a requested retreat trajectory. |

## Detector configuration

The default detector parameters are declared in `vision_node.py`:

- Confidence threshold: `0.45`.
- Stop distance: `0.80 m`.
- RGB/depth timestamp tolerance: `0.50 s`.
- Depth is required for a close-obstacle decision.
- Target classes: `person`, `chair`, `car`, `truck`, `bus`, `bicycle`, and `motorcycle`.

The node can be run independently with ROS parameters if the camera topics are already available:

```bash
ros2 run ur_vision_avoidance vision_node --ros-args \
  -p model:=/absolute/path/to/best.pt \
  -p confidence:=0.50 \
  -p stop_distance_m:=0.75 \
  -p require_depth:=true
```

A custom detector should be trained for the actual objects and camera viewpoint used in the experiment. A generic COCO checkpoint is suitable for pipeline testing, but it does not establish reliable obstacle coverage.

## Fixed-scene reinforcement-learning starter

The current RL stage intentionally uses one deterministic kinematic scene. It is a starter interface for testing observation construction, action dimensions, reward behavior, and PPO training before augmented randomized domain is introduced. It does not model full Gazebo dynamics and it does not send actions to the physical or simulated UR controller.

Install the optional training dependencies:

```bash
python3 -m pip install -r requirements-rl.txt
```

Train a PPO policy using the fixed obstacle and nominal path:

```bash
python3 -m ur_vision_avoidance.train_fixed_scene \
  --timesteps 100000 \
  --seed 7 \
  --output /tmp/ur_vision_avoidance_ppo
```

The command produces `/tmp/ur_vision_avoidance_ppo.zip`. ARD is rejected explicitly in this starter stage:

```bash
python3 -m ur_vision_avoidance.train_fixed_scene --ard
# Expected: ARD is intentionally disabled in this fixed-scene starter stage.
```

The training environment uses a 16-element observation and a 6-element normalized action. The observation contains relative goal state, relative obstacle state, current velocity, previous action, and normalized remaining time. The action represents bounded Cartesian linear and angular offsets. The starter kinematic environment applies only the linear part; the angular dimensions preserve the planned interface for later integration.

The fixed-scene reward favors progress toward the goal, penalizes low obstacle clearance, trajectory deviation, action changes, time, and collision, and gives a goal-completion bonus. This reward is a research starting point and must be evaluated against collision rate, minimum clearance, goal success, path deviation, latency, and unnecessary avoidance.

### Run the ROS 2 RL interface

First launch perception without the old deterministic retreat:

```bash
ros2 launch ur_vision_avoidance vision_bringup.launch.py \
  enable_avoidance:=false \
  enable_rl_starter:=true \
  rl_model:=/tmp/ur_vision_avoidance_ppo.zip
```

Inspect the structured observation and bounded action:

```bash
ros2 topic echo /rl/observation --once
ros2 topic echo /rl/action --once
```

The observation adapter consumes `/vision/detections` and `/joint_states`, then publishes JSON with the schema name `ur_vision_avoidance.rl_observation.v1`. It includes detection confidence, depth validity, sensor age, joint positions, joint velocities, and an obstacle summary. The policy node publishes six bounded values on `/rl/action`: three linear velocity offsets in metres per second and three angular velocity offsets in radians per second.

The policy node publishes zeros when no model is supplied, sensor data are stale, the observation is invalid, or inference fails. This conservative behavior is intentional. There is no automatic conversion from `/rl/action` to a `FollowJointTrajectory` goal in this stage. A later controller adapter must implement kinematic limits, collision checks, trajectory smoothing, stale-data handling, and a deterministic fallback before any action is executed.

### Fixed-stage limitations

The starter policy sees one fixed obstacle arrangement during training. It can memorize that arrangement and should not be expected to generalize. Do not enable ARD by editing the configuration and then treat the result as validated training; ARD is a separate development stage. The next stage should add obstacle-position variation, then sensor noise and latency, and only later moving obstacles and sim-to-real evaluation.

## Troubleshooting

### No camera topics appear

Check both Gazebo Transport and ROS 2 topics:

```bash
gz topic -l | grep -E 'camera|image|depth'
ros2 topic list | grep camera
```

If Gazebo topics are absent, confirm that the world includes `gz-sim-sensors-system`, that the Xacro passed to the simulator is the custom file, and that Gazebo is not paused. If the topic prefix differs from `camera`, update `config/bridge.yaml` to match the actual Gazebo topic names.

### The annotated image is blank

First confirm that `/camera/image_raw` has a nonzero rate. Then inspect the topic type and QoS:

```bash
ros2 topic info /camera/image_raw -v
ros2 topic hz /camera/image_raw
```

Also confirm that the model path exists and that the Ultralytics checkpoint loads without an exception. A blank or incorrectly oriented camera image must be fixed before evaluating detection results.

### YOLO detects nothing

The bundled model is a general pretrained checkpoint. It may not recognize the blue geometric box in the sample world because `box` is not a standard COCO object class. Test first with the chair, lower the confidence threshold temporarily, and inspect `/vision/annotated`. For a real experiment, train a custom checkpoint using images from the mounted camera and the expected workspace.

### Distance is `-1.0`

Confirm that depth is publishing and inspect its encoding:

```bash
ros2 topic echo /camera/depth_image_raw --once
ros2 topic hz /camera/depth_image_raw
```

The node supports common `32FC1` metre depth and `16UC1` millimetre depth conventions. It rejects stale frames according to `depth_timeout_s` and ignores invalid or non-positive depth values.

### The controller action is unavailable

Check the active controllers and action names:

```bash
ros2 control list_controllers
ros2 action list | grep follow_joint_trajectory
```

The configured action is `/scaled_joint_trajectory_controller/follow_joint_trajectory`. If a different controller is active, set `initial_joint_controller` and the supervisor's `action_name` consistently. Do not send goals until the intended simulated controller is active.

### The Xacro fails

Run Xacro directly after sourcing the required workspaces:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ur_vision_ws/install/setup.bash
ros2 run xacro xacro \
  ~/ur_vision_ws/src/ur_vision_avoidance/urdf/ur_gz_camera.urdf.xacro \
  ur_type:=ur5e > /tmp/ur_camera.urdf
```

Check that `ur_description`, `ur_simulation_gz`, and the requested UR model parameter files are installed.

## Extending this project toward reinforcement learning

The current supervisor should not be labeled as reinforcement learning. A real RL extension needs a defined training interface. The observation vector could combine object detections, depth ranges, camera-relative lateral offsets, robot joint positions, and joint velocities. The action space could contain bounded joint velocity commands or target poses. The environment must define reset behavior, collision and workspace constraints, reward terms, episode termination, and sensor-failure handling.

After training, the policy should be exported with a versioned checkpoint and loaded by a dedicated ROS 2 inference node. That node must enforce command limits, monitor stale observations, reject invalid actions, and provide a deterministic safe fallback. Simulation evaluation should cover empty scenes, occlusion, camera dropouts, missed detections, moving obstacles, and controller delays before any physical deployment is considered.

## Safety and deployment boundary

This repository is suitable for simulation and research integration. It is not sufficient for operating a physical UR5. A physical deployment requires risk assessment, validated sensing, collision monitoring, controller-level limits, emergency-stop integration, safe-speed configuration, and a safety-rated architecture appropriate to the application.

## Project files

- `launch/vision_bringup.launch.py`: main simulation and perception launch file.
- `ur_vision_avoidance/vision_node.py`: YOLO and RGB-D perception node.
- `ur_vision_avoidance/avoidance_supervisor.py`: optional simulation-only retreat supervisor.
- `ur_vision_avoidance/fixed_scene_env.py`: deterministic PPO training environment with ARD disabled.
- `ur_vision_avoidance/train_fixed_scene.py`: fixed-scene PPO training command.
- `ur_vision_avoidance/rl_observation_node.py`: structured ROS observation adapter.
- `ur_vision_avoidance/rl_policy_node.py`: bounded PPO inference node that does not command the robot.
- `urdf/ur_gz_camera.urdf.xacro`: UR5e plus camera description.
- `worlds/obstacle_world.sdf`: Gazebo test world.
- `config/bridge.yaml`: Gazebo-to-ROS topic bridge configuration.
- `config/rl_starter.yaml`: fixed-scene RL settings; `ard_enabled` is `false`.
- `requirements-rl.txt`: optional Gymnasium and Stable-Baselines3 dependencies.
- `validate_project.py`: static project consistency checks.
- `ROS2_Jazzy_UR_Gazebo_DeepLearning_Obstacle_Avoidance_Tutorial.md`: extended background tutorial.
- `AUDIT_REPORT.md`: archive audit findings and applied corrections.

## References

[1]: https://docs.ros.org/en/jazzy/ "ROS 2 Jazzy documentation"
[2]: https://gazebosim.org/docs/harmonic/ "Gazebo Harmonic documentation"
[3]: https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation "Universal Robots ROS 2 Gazebo Simulation"
[4]: https://docs.ultralytics.com/modes/predict/ "Ultralytics prediction documentation"
[5]: https://docs.ros.org/en/jazzy/p/ros_gz_bridge/ "ROS-Gazebo bridge documentation"
