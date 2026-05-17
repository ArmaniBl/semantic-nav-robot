#!/usr/bin/env python3
import argparse
import signal
import sys
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


class SlamReadyWaiter(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("slam_ready_waiter")
        self.args = args
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

    def ready(self) -> tuple[bool, str]:
        missing = []
        if not self.get_publishers_info_by_topic("/clock"):
            missing.append("/clock")
        if not self.get_publishers_info_by_topic(self.args.scan_topic):
            missing.append(self.args.scan_topic)
        if not self.get_publishers_info_by_topic(self.args.map_topic):
            missing.append(self.args.map_topic)
        if not self.tf_buffer.can_transform(
            self.args.target_frame,
            self.args.source_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=0.0),
        ):
            missing.append(f"{self.args.target_frame}->{self.args.source_frame}")
        if missing:
            return False, "missing " + ", ".join(missing)
        return True, "ready"

    def close(self) -> None:
        try:
            self.tf_listener.unregister()
        except Exception:
            pass
        executor = getattr(self.tf_listener, "executor", None)
        if executor is not None:
            try:
                executor.shutdown()
            except Exception:
                pass
        thread = getattr(self.tf_listener, "dedicated_listener_thread", None)
        if thread is not None:
            thread.join(timeout=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait until SLAM has map, clock, scan, and TF.")
    parser.add_argument("--map-topic", default="/map")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--target-frame", default="map")
    parser.add_argument("--source-frame", default="base_link")
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument("--log-period-sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stop = False

    def on_signal(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    rclpy.init()
    node = SlamReadyWaiter(args)
    start = time.monotonic()
    last_log = 0.0
    try:
        while rclpy.ok() and not stop:
            rclpy.spin_once(node, timeout_sec=0.2)
            ok, summary = node.ready()
            if ok:
                node.get_logger().info("SLAM is ready; starting Nav2")
                return 0
            now = time.monotonic()
            if now - last_log >= args.log_period_sec:
                node.get_logger().info(f"Waiting for SLAM readiness: {summary}")
                last_log = now
            if args.timeout_sec > 0.0 and now - start >= args.timeout_sec:
                node.get_logger().error(f"Timed out waiting for SLAM readiness: {summary}")
                return 1
        return 130
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
