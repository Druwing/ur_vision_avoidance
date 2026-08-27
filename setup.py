from setuptools import find_packages, setup

package_name = "ur_vision_avoidance"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/bridge.yaml"]),
        ("share/" + package_name + "/launch", ["launch/vision_bringup.launch.py"]),
        ("share/" + package_name + "/urdf", ["urdf/ur_gz_camera.urdf.xacro"]),
        ("share/" + package_name + "/worlds", ["worlds/obstacle_world.sdf"]),
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
        ],
    },
)
