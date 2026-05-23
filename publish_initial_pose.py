#!/usr/bin/env python3
import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.parameter import Parameter


class InitialPosePublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("initial_pose_publisher")
        self.args = args
        if args.use_sim_time:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.publisher = self.create_publisher(PoseWithCovarianceStamped, args.topic, 10)
        self.sent = 0
        self.started_at = time.monotonic()
        self.timer = self.create_timer(1.0 / args.rate_hz, self.on_timer)

    def on_timer(self) -> None:
        now = self.get_clock().now()
        if self.args.use_sim_time and now.nanoseconds == 0:
            return
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.args.frame
        msg.pose.pose.position.x = self.args.x
        msg.pose.pose.position.y = self.args.y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = math.sin(self.args.yaw * 0.5)
        msg.pose.pose.orientation.w = math.cos(self.args.yaw * 0.5)
        msg.pose.covariance[0] = self.args.xy_covariance
        msg.pose.covariance[7] = self.args.xy_covariance
        msg.pose.covariance[35] = self.args.yaw_covariance
        self.publisher.publish(msg)
        self.sent += 1
        if self.sent == 1:
            self.get_logger().info(
                f"Published initial pose on {self.args.topic}: "
                f"x={self.args.x:.3f}, y={self.args.y:.3f}, yaw={self.args.yaw:.3f}"
            )
        if time.monotonic() - self.started_at >= self.args.duration_sec:
            raise SystemExit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish an AMCL initial pose for a short time.")
    parser.add_argument("--topic", default="/initialpose")
    parser.add_argument("--frame", default="map")
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--xy-covariance", type=float, default=0.25)
    parser.add_argument("--yaw-covariance", type=float, default=0.25)
    parser.add_argument("--duration-sec", type=float, default=8.0)
    parser.add_argument("--rate-hz", type=float, default=2.0)
    parser.add_argument("--use-sim-time", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration_sec <= 0.0:
        raise ValueError("--duration-sec must be > 0")
    if args.rate_hz <= 0.0:
        raise ValueError("--rate-hz must be > 0")
    rclpy.init()
    node = InitialPosePublisher(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
