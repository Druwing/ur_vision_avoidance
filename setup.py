from setuptools import find_packages, setup

package_name = "ur_vision_avoidance"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            ["config/bridge.yaml", "config/rl_starter.yaml", "config/scene_ground_truth.yaml"],
        ),
        ("share/" + package_name + "/launch", ["launch/vision_bringup.launch.py"]),
        ("share/" + package_name + "/urdf", ["urdf/ur_gz_camera.urdf.xacro"]),
        ("share/" + package_name + "/worlds", ["worlds/obstacle_world.sdf"]),
        ("share/" + package_name + "/models", ["yolo26n.pt", "models/ppo_ur5e_scene.zip"]),
        ("share/" + package_name + "/scripts", [
            "scripts/evaluate_kinematic.py", "scripts/benchmark_yolo.py", "scripts/summarize_metrics.py",
            "scripts/run_gazebo_episode.sh", "scripts/capture_dataset.sh",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Developer",
    maintainer_email="developer@example.com",
    description="Starter RGB-D YOLO perception and simulation-only avoidance for a UR cobot.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vision_node = ur_vision_avoidance.vision_node:main",
            "avoidance_supervisor = ur_vision_avoidance.avoidance_supervisor:main",
            "motion_loop = ur_vision_avoidance.motion_loop:main",
            "rl_observation_node = ur_vision_avoidance.rl_observation_node:main",
            "rl_policy_node = ur_vision_avoidance.rl_policy_node:main",
            "obstacle_localizer = ur_vision_avoidance.obstacle_localizer_node:main",
            "action_adapter = ur_vision_avoidance.action_adapter_node:main",
            "train_fixed_scene = ur_vision_avoidance.train_fixed_scene:main",
            "metrics_node = ur_vision_avoidance.metrics_node:main",
            "dataset_capture = ur_vision_avoidance.dataset_capture_node:main",
        ],
    },
)
