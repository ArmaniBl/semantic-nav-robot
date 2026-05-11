#!/usr/bin/env python3
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = PROJECT_ROOT / "config" / "nav2_odom_params.yaml"


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    log_level = LaunchConfiguration("log_level")
    publish_lidar_tf = LaunchConfiguration("publish_lidar_tf")
    base_frame = LaunchConfiguration("base_frame")
    lidar_frame = LaunchConfiguration("lidar_frame")
    lidar_x = LaunchConfiguration("lidar_x")
    lidar_y = LaunchConfiguration("lidar_y")
    lidar_z = LaunchConfiguration("lidar_z")
    lidar_roll = LaunchConfiguration("lidar_roll")
    lidar_pitch = LaunchConfiguration("lidar_pitch")
    lidar_yaw = LaunchConfiguration("lidar_yaw")

    remappings = [
        ("/tf", "tf"),
        ("/tf_static", "tf_static"),
        ("cmd_vel", "/cmd_vel"),
    ]

    return LaunchDescription(
        [
            SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"),
            DeclareLaunchArgument(
                "params_file",
                default_value=str(DEFAULT_PARAMS),
                description="Path to the odom-frame Nav2 params file.",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use /clock from Isaac Sim.",
            ),
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="ROS log level.",
            ),
            DeclareLaunchArgument(
                "publish_lidar_tf",
                default_value="true",
                description="Publish an approximate base_link -> front_3d_lidar static TF.",
            ),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("lidar_frame", default_value="front_3d_lidar"),
            DeclareLaunchArgument(
                "lidar_x",
                default_value="0.40",
                description="Approximate lidar x offset in base frame.",
            ),
            DeclareLaunchArgument("lidar_y", default_value="0.00"),
            DeclareLaunchArgument(
                "lidar_z",
                default_value="0.45",
                description="Approximate lidar z offset in base frame.",
            ),
            DeclareLaunchArgument("lidar_roll", default_value="0.0"),
            DeclareLaunchArgument("lidar_pitch", default_value="0.0"),
            DeclareLaunchArgument("lidar_yaw", default_value="0.0"),
            SetParameter("use_sim_time", use_sim_time),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="front_lidar_static_tf",
                output="screen",
                condition=IfCondition(publish_lidar_tf),
                arguments=[
                    "--x",
                    lidar_x,
                    "--y",
                    lidar_y,
                    "--z",
                    lidar_z,
                    "--roll",
                    lidar_roll,
                    "--pitch",
                    lidar_pitch,
                    "--yaw",
                    lidar_yaw,
                    "--frame-id",
                    base_frame,
                    "--child-frame-id",
                    lidar_frame,
                ],
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=[params_file],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings,
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[params_file],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings,
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                output="screen",
                parameters=[params_file],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings,
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                parameters=[params_file],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings,
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                parameters=[params_file],
                arguments=["--ros-args", "--log-level", log_level],
            ),
        ]
    )
