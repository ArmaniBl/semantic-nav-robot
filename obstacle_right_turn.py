#!/usr/bin/env python3
import argparse
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


class ObstacleRightTurn(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("obstacle_right_turn")
        self.args = args
        self.nearest_distance: float | None = None
        self.nearest_angle_deg: float | None = None
        self.last_scan_time = 0.0
        self.last_log_time = 0.0
        self.state = "drive"
        self.turn_started_at = 0.0
        self.turn_duration_sec = math.radians(args.turn_angle_deg) / abs(args.angular_speed)

        if self.has_parameter("use_sim_time"):
            self.set_parameters(
                [Parameter("use_sim_time", Parameter.Type.BOOL, args.use_sim_time)]
            )
        else:
            self.declare_parameter("use_sim_time", args.use_sim_time)

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(LaserScan, args.scan_topic, self.on_scan, scan_qos)
        self.cmd_pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.create_timer(1.0 / args.rate_hz, self.on_timer)

        self.get_logger().info(
            "Simple obstacle turn started: "
            f"scan={args.scan_topic}, cmd_vel={args.cmd_vel_topic}, "
            f"stop_distance={args.stop_distance_m:.2f}m, "
            f"turn_angle={args.turn_angle_deg:.1f}deg"
        )

    def on_scan(self, msg: LaserScan) -> None:
        nearest = None
        nearest_angle = None
        for index, raw_range in enumerate(msg.ranges):
            if not math.isfinite(raw_range):
                continue
            if raw_range < msg.range_min or raw_range > msg.range_max:
                continue
            if nearest is None or raw_range < nearest:
                nearest = raw_range
                nearest_angle = math.degrees(msg.angle_min + index * msg.angle_increment)

        self.nearest_distance = nearest
        self.nearest_angle_deg = nearest_angle
        self.last_scan_time = time.monotonic()

    def on_timer(self) -> None:
        if self.last_scan_time == 0.0:
            self.publish_stop()
            return

        if time.monotonic() - self.last_scan_time > self.args.scan_timeout_sec:
            self.log_status("No fresh scan; stopping")
            self.publish_stop()
            return

        if self.state == "turn":
            self.turn_step()
            return

        if self.nearest_distance is None:
            self.start_turn("No finite obstacle ranges; turning 90 deg right")
            return

        if self.nearest_distance <= self.args.stop_distance_m:
            self.start_turn("Obstacle visible; turning 90 deg right")
            return

        self.drive_forward()
        self.log_status("Path clear; driving forward")

    def drive_forward(self) -> None:
        twist = Twist()
        twist.linear.x = self.args.linear_speed
        self.cmd_pub.publish(twist)

    def turn_right(self) -> None:
        twist = Twist()
        twist.angular.z = -abs(self.args.angular_speed)
        self.cmd_pub.publish(twist)

    def start_turn(self, message: str) -> None:
        self.state = "turn"
        self.turn_started_at = time.monotonic()
        self.publish_stop()
        self.log_status(message)

    def turn_step(self) -> None:
        elapsed = time.monotonic() - self.turn_started_at
        if elapsed >= self.turn_duration_sec:
            self.publish_stop()
            self.state = "drive"
            self.log_status("90 deg turn complete; driving forward")
            return
        self.turn_right()

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def log_status(self, message: str) -> None:
        now = time.monotonic()
        if now - self.last_log_time < self.args.status_log_period_sec:
            return
        self.last_log_time = now
        nearest = (
            f"{self.nearest_distance:.2f}m @ {self.nearest_angle_deg:.1f}deg"
            if self.nearest_distance is not None and self.nearest_angle_deg is not None
            else "none"
        )
        self.get_logger().info(f"{message}: nearest={nearest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive forward, and turn right whenever any obstacle is visible nearby."
    )
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--stop-distance-m", type=float, default=2.0)
    parser.add_argument("--linear-speed", type=float, default=0.25)
    parser.add_argument("--angular-speed", type=float, default=0.4)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--scan-timeout-sec", type=float, default=1.0)
    parser.add_argument("--status-log-period-sec", type=float, default=1.0)
    parser.add_argument("--use-sim-time", action=argparse.BooleanOptionalAction, default=True)

    # Kept so old commands do not fail. Only --turn-angle-deg is used.
    parser.add_argument("--turns", type=int, default=0)
    parser.add_argument("--turn-angle-deg", type=float, default=90.0)
    parser.add_argument("--front-center-deg", type=float, default=0.0)
    parser.add_argument("--front-angle-deg", type=float, default=0.0)
    parser.add_argument("--safety-angle-deg", type=float, default=0.0)
    parser.add_argument("--emergency-distance-m", type=float, default=0.0)
    parser.add_argument("--max-safe-linear-speed", type=float, default=0.0)
    parser.add_argument("--unsafe-allow-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stop_distance_m <= 0.0:
        raise ValueError("--stop-distance-m must be > 0")
    if args.linear_speed <= 0.0:
        raise ValueError("--linear-speed must be > 0")
    if args.angular_speed == 0.0:
        raise ValueError("--angular-speed must be non-zero")
    if args.turn_angle_deg <= 0.0:
        raise ValueError("--turn-angle-deg must be > 0")
    if args.rate_hz <= 0.0:
        raise ValueError("--rate-hz must be > 0")
    if args.scan_timeout_sec <= 0.0:
        raise ValueError("--scan-timeout-sec must be > 0")
    if args.status_log_period_sec <= 0.0:
        raise ValueError("--status-log-period-sec must be > 0")

    rclpy.init()
    node = ObstacleRightTurn(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
