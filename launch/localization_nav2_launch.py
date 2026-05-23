#!/usr/bin/env python3
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, RegisterEventHandler, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = PROJECT_ROOT / "config" / "nav2_localization_params.yaml"


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    map_yaml = LaunchConfiguration("map")
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

    scan_angle_min = LaunchConfiguration("scan_angle_min")
    scan_angle_max = LaunchConfiguration("scan_angle_max")
    scan_range_max = LaunchConfiguration("scan_range_max")
    initial_pose_x = LaunchConfiguration("initial_pose_x")
    initial_pose_y = LaunchConfiguration("initial_pose_y")
    initial_pose_yaw = LaunchConfiguration("initial_pose_yaw")

    remappings = [
        ("/tf", "tf"),
        ("/tf_static", "tf_static"),
        ("cmd_vel", "/cmd_vel"),
    ]

    nav2_ready_waiter = ExecuteProcess(
        cmd=[
            sys.executable,
            str(PROJECT_ROOT / "wait_for_slam_ready.py"),
            "--map-topic",
            "/map",
            "--scan-topic",
            "/scan",
            "--target-frame",
            "map",
            "--source-frame",
            "base_link",
        ],
        cwd=str(PROJECT_ROOT),
        output="screen",
    )

    nav2_nodes = [
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
    ]

    nav2_activator = ExecuteProcess(
        cmd=[
            sys.executable,
            str(PROJECT_ROOT / "activate_nav2_lifecycle.py"),
            "--start-delay-sec",
            "1.0",
        ],
        cwd=str(PROJECT_ROOT),
        output="screen",
    )

    initial_pose_publisher = ExecuteProcess(
        cmd=[
            sys.executable,
            str(PROJECT_ROOT / "publish_initial_pose.py"),
            "--x",
            initial_pose_x,
            "--y",
            initial_pose_y,
            "--yaw",
            initial_pose_yaw,
            "--use-sim-time",
        ],
        cwd=str(PROJECT_ROOT),
        output="screen",
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"),
            DeclareLaunchArgument(
                "params_file",
                default_value=str(DEFAULT_PARAMS),
                description="Path to Nav2 localization params.",
            ),
            DeclareLaunchArgument("map", description="Path to saved occupancy map YAML."),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("log_level", default_value="info"),
            DeclareLaunchArgument("publish_lidar_tf", default_value="true"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("lidar_frame", default_value="front_3d_lidar"),
            DeclareLaunchArgument("lidar_x", default_value="0.40"),
            DeclareLaunchArgument("lidar_y", default_value="0.00"),
            DeclareLaunchArgument("lidar_z", default_value="0.45"),
            DeclareLaunchArgument("lidar_roll", default_value="0.0"),
            DeclareLaunchArgument("lidar_pitch", default_value="0.0"),
            DeclareLaunchArgument("lidar_yaw", default_value="0.0"),
            DeclareLaunchArgument("scan_angle_min", default_value="-3.14159"),
            DeclareLaunchArgument("scan_angle_max", default_value="3.14159"),
            DeclareLaunchArgument("scan_range_max", default_value="10.0"),
            DeclareLaunchArgument("initial_pose_x", default_value="-6.0"),
            DeclareLaunchArgument("initial_pose_y", default_value="-1.0"),
            DeclareLaunchArgument("initial_pose_yaw", default_value="0.0"),
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
                package="pointcloud_to_laserscan",
                executable="pointcloud_to_laserscan_node",
                name="pointcloud_to_laserscan",
                output="screen",
                parameters=[
                    {
                        "target_frame": lidar_frame,
                        "transform_tolerance": 0.05,
                        "min_height": -0.35,
                        "max_height": 1.20,
                        "angle_min": scan_angle_min,
                        "angle_max": scan_angle_max,
                        "angle_increment": 0.0087,
                        "scan_time": 0.1,
                        "range_min": 0.10,
                        "range_max": scan_range_max,
                        "use_inf": True,
                        "inf_epsilon": 1.0,
                    }
                ],
                remappings=[
                    ("cloud_in", "/front_3d_lidar/lidar_points"),
                    ("scan", "/scan"),
                ],
            ),
            Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=[params_file],
                arguments=[
                    "--ros-args",
                    "-p",
                    ["yaml_filename:=", map_yaml],
                    "--log-level",
                    log_level,
                ],
            ),
            Node(
                package="nav2_amcl",
                executable="amcl",
                name="amcl",
                output="screen",
                parameters=[params_file],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings,
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_localization",
                output="screen",
                parameters=[params_file],
                arguments=["--ros-args", "--log-level", log_level],
            ),
            TimerAction(period=3.0, actions=[initial_pose_publisher]),
            nav2_ready_waiter,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=nav2_ready_waiter,
                    on_exit=[
                        LogInfo(msg="[LocalizationLaunch] saved map ready; starting Nav2"),
                        *nav2_nodes,
                        TimerAction(period=1.0, actions=[nav2_activator]),
                    ],
                )
            ),
        ]
    )
