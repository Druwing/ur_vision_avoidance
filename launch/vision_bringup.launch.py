from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    IfElseSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def launch_simulation(context, *args, **kwargs):
    ur_type = LaunchConfiguration("ur_type")
    description_file = LaunchConfiguration("description_file")
    controllers_file = LaunchConfiguration("controllers_file")
    tf_prefix = LaunchConfiguration("tf_prefix")
    safety_limits = LaunchConfiguration("safety_limits")
    safety_pos_margin = LaunchConfiguration("safety_pos_margin")
    safety_k_position = LaunchConfiguration("safety_k_position")
    launch_rviz = LaunchConfiguration("launch_rviz")
    rviz_config_file = LaunchConfiguration("rviz_config_file")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    world_file = LaunchConfiguration("world_file")
    activate_joint_controller = LaunchConfiguration("activate_joint_controller")
    initial_joint_controller = LaunchConfiguration("initial_joint_controller")

    robot_description_content = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            description_file,
            " safety_limits:=",
            safety_limits,
            " safety_pos_margin:=",
            safety_pos_margin,
            " safety_k_position:=",
            safety_k_position,
            " name:=ur ur_type:=",
            ur_type,
            " tf_prefix:=",
            tf_prefix,
            " simulation_controllers:=",
            controllers_file,
        ]
    )
    robot_description = ParameterValue(robot_description_content, value_type=str)

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        condition=IfCondition(launch_rviz),
    )

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[{"use_sim_time": True}, {"robot_description": robot_description}],
        ),
        joint_state_broadcaster,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster,
                on_exit=[rviz_node],
            ),
            condition=IfCondition(launch_rviz),
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[initial_joint_controller, "-c", "/controller_manager"],
            condition=IfCondition(activate_joint_controller),
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[initial_joint_controller, "-c", "/controller_manager", "--stopped"],
            condition=UnlessCondition(activate_joint_controller),
        ),
        Node(
            package="ros_gz_sim",
            executable="create",
            output="screen",
            arguments=["-string", robot_description_content, "-name", "ur", "-allow_renaming", "true"],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
            ),
            launch_arguments={
                "gz_args": IfElseSubstitution(
                    gazebo_gui,
                    if_value=[" -r -v 4 ", world_file],
                    else_value=[" -s -r -v 4 ", world_file],
                )
            }.items(),
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
            output="screen",
        ),
    ]


def generate_launch_description():
    package_share = FindPackageShare("ur_vision_avoidance")

    ur_type = LaunchConfiguration("ur_type")
    model = LaunchConfiguration("model")
    enable_avoidance = LaunchConfiguration("enable_avoidance")

    custom_description = PathJoinSubstitution(
        [package_share, "urdf", "ur_gz_camera.urdf.xacro"]
    )
    world_file = PathJoinSubstitution([package_share, "worlds", "obstacle_world.sdf"])
    bridge_config = PathJoinSubstitution([package_share, "config", "bridge.yaml"])
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "ur_type",
                default_value="ur5e",
                description="UR model: ur3e, ur5e, ur10e, ur20, and others supported by the package.",
            ),
            DeclareLaunchArgument(
                "model",
                default_value="yolo26n.pt",
                description="Ultralytics checkpoint or path to a custom best.pt.",
            ),
            DeclareLaunchArgument(
                "enable_avoidance",
                default_value="false",
                description="Enable the simulation-only retreat supervisor.",
            ),
            DeclareLaunchArgument("description_file", default_value=custom_description),
            DeclareLaunchArgument(
                "controllers_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("ur_simulation_gz"), "config", "ur_controllers.yaml"]
                ),
            ),
            DeclareLaunchArgument("tf_prefix", default_value='""'),
            DeclareLaunchArgument("safety_limits", default_value="true"),
            DeclareLaunchArgument("safety_pos_margin", default_value="0.15"),
            DeclareLaunchArgument("safety_k_position", default_value="20"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument(
                "rviz_config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("ur_description"), "rviz", "view_robot.rviz"]
                ),
            ),
            DeclareLaunchArgument("gazebo_gui", default_value="true"),
            DeclareLaunchArgument("world_file", default_value=world_file),
            DeclareLaunchArgument("activate_joint_controller", default_value="true"),
            DeclareLaunchArgument(
                "initial_joint_controller",
                default_value="scaled_joint_trajectory_controller",
            ),
            OpaqueFunction(function=launch_simulation),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="camera_bridge",
                output="screen",
                parameters=[{"config_file": bridge_config}],
            ),
            Node(
                package="ur_vision_avoidance",
                executable="vision_node",
                name="vision_obstacle_detector",
                output="screen",
                parameters=[
                    {"use_sim_time": True},
                    {"model": model},
                ],
            ),
            Node(
                package="ur_vision_avoidance",
                executable="avoidance_supervisor",
                name="avoidance_supervisor",
                output="screen",
                condition=IfCondition(enable_avoidance),
                parameters=[{"use_sim_time": True}],
            ),
        ]
    )
