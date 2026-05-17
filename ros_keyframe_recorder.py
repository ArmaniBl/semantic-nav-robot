#!/usr/bin/env python3
import argparse
import json
import math
import re
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener


def stamp_to_sec(msg) -> float:
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def shortest_angle_delta(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def quaternion_multiply(left: tuple[float, float, float, float], right: tuple[float, float, float, float]):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def normalize_quaternion(quaternion: tuple[float, float, float, float]):
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-6:
        raise ValueError("invalid quaternion")
    return x / norm, y / norm, z / norm, w / norm


def rotate_vector(vector: tuple[float, float, float], quaternion: tuple[float, float, float, float]):
    q_conjugate = (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, (vector[0], vector[1], vector[2], 0.0)),
        q_conjugate,
    )
    return rotated[:3]


def transform_pose(position, orientation, transform):
    t = transform.transform.translation
    r = transform.transform.rotation
    transform_quat = normalize_quaternion((r.x, r.y, r.z, r.w))
    pose_quat = normalize_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))

    rotated_position = rotate_vector((position.x, position.y, position.z), transform_quat)
    out_quat = normalize_quaternion(quaternion_multiply(transform_quat, pose_quat))
    return (
        {
            "x": rotated_position[0] + t.x,
            "y": rotated_position[1] + t.y,
            "z": rotated_position[2] + t.z,
        },
        {
            "x": out_quat[0],
            "y": out_quat[1],
            "z": out_quat[2],
            "w": out_quat[3],
        },
    )


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    dtype = np.uint8
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
    }
    if msg.encoding not in channels_by_encoding:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    channels = channels_by_encoding[msg.encoding]
    data = np.frombuffer(msg.data, dtype=dtype)
    image = data.reshape((msg.height, msg.step))
    image = image[:, : msg.width * channels].reshape((msg.height, msg.width, channels))

    if msg.encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if msg.encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if msg.encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if msg.encoding == "mono8":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def find_last_keyframe_index(images_dir: Path) -> int:
    pattern = re.compile(r"^keyframe_(\d+)\.png$")
    indexes = []
    for path in images_dir.glob("keyframe_*.png"):
        match = pattern.match(path.name)
        if match:
            indexes.append(int(match.group(1)))
    return max(indexes, default=0)


class KeyframeRecorder(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("keyframe_recorder")
        self.image_topic = args.image_topic
        self.odom_topic = args.odom_topic
        self.output_dir = Path(args.output_dir)
        self.images_dir = self.output_dir / "images"
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.min_time_delta_sec = args.min_time_delta_sec
        self.min_translation_delta_m = args.min_translation_delta_m
        self.min_rotation_delta_rad = args.min_rotation_delta_rad
        self.max_pose_age_sec = args.max_pose_age_sec
        self.target_frame = args.target_frame.strip()
        self.tf_timeout_sec = args.tf_timeout_sec
        self.allow_latest_tf_fallback = args.allow_latest_tf_fallback
        self.warned_latest_tf_fallback = False

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.latest_odom: Optional[Odometry] = None
        self.last_saved_odom: Optional[Odometry] = None
        self.last_saved_time: Optional[float] = None
        self.saved_count = find_last_keyframe_index(self.images_dir)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Image, self.image_topic, self.on_image, sensor_qos)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, odom_qos)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.get_logger().info(
            f"Recording keyframes: image={self.image_topic}, odom={self.odom_topic}, "
            f"target_frame={self.target_frame or '<odom header>'}, "
            f"output={self.output_dir}, next_index={self.saved_count + 1}"
        )

    def on_odom(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def should_save(self, image_time: float, odom: Odometry) -> bool:
        if self.last_saved_odom is None or self.last_saved_time is None:
            return True

        dt = image_time - self.last_saved_time
        if dt < self.min_time_delta_sec:
            return False

        p = odom.pose.pose.position
        last_p = self.last_saved_odom.pose.pose.position
        translation = math.hypot(p.x - last_p.x, p.y - last_p.y)

        yaw = yaw_from_quaternion(odom.pose.pose.orientation)
        last_yaw = yaw_from_quaternion(self.last_saved_odom.pose.pose.orientation)
        rotation = abs(shortest_angle_delta(yaw, last_yaw))

        return (
            translation >= self.min_translation_delta_m
            or rotation >= self.min_rotation_delta_rad
        )

    def recorded_pose(self, odom: Odometry) -> tuple[str, dict, dict]:
        source_frame = odom.header.frame_id
        if not source_frame:
            raise ValueError("Odometry header.frame_id is empty")

        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        target_frame = self.target_frame or source_frame
        if target_frame == source_frame:
            qx, qy, qz, qw = normalize_quaternion((q.x, q.y, q.z, q.w))
            return (
                source_frame,
                {"x": p.x, "y": p.y, "z": p.z},
                {"x": qx, "y": qy, "z": qz, "w": qw},
            )

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time.from_msg(odom.header.stamp),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except TransformException:
            if not self.allow_latest_tf_fallback:
                raise
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
            if not self.warned_latest_tf_fallback:
                self.get_logger().warn(
                    f"Using latest TF {target_frame}->{source_frame} for keyframe pose "
                    "because timestamped TF was unavailable"
                )
                self.warned_latest_tf_fallback = True

        position, orientation = transform_pose(p, q, transform)
        return transform.header.frame_id, position, orientation

    def on_image(self, msg: Image) -> None:
        if self.latest_odom is None:
            self.get_logger().warn("Image received, but odometry is not available yet")
            return

        image_time = stamp_to_sec(msg)
        odom_time = stamp_to_sec(self.latest_odom)
        pose_age = abs(image_time - odom_time)
        if pose_age > self.max_pose_age_sec:
            self.get_logger().warn(
                f"Skipping image: pose age {pose_age:.3f}s exceeds "
                f"{self.max_pose_age_sec:.3f}s"
            )
            return

        if not self.should_save(image_time, self.latest_odom):
            return

        odom = self.latest_odom
        try:
            pose_frame, position, orientation = self.recorded_pose(odom)
        except (ValueError, TransformException) as exc:
            self.get_logger().warn(f"Skipping image: failed to resolve keyframe pose: {exc}")
            return

        try:
            image = image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().error(f"Failed to decode image: {exc}")
            return

        self.saved_count += 1
        frame_id = f"keyframe_{self.saved_count:06d}"
        image_path = self.images_dir / f"{frame_id}.png"
        while image_path.exists():
            self.saved_count += 1
            frame_id = f"keyframe_{self.saved_count:06d}"
            image_path = self.images_dir / f"{frame_id}.png"

        if not cv2.imwrite(str(image_path), image):
            self.get_logger().error(f"Failed to save image: {image_path}")
            return

        record = {
            "memory_id": frame_id,
            "timestamp": image_time,
            "image_path": str(image_path),
            "image_topic": self.image_topic,
            "image_frame": msg.header.frame_id,
            "pose_topic": self.odom_topic,
            "pose_frame": pose_frame,
            "child_frame_id": odom.child_frame_id,
            "pose_age_sec": pose_age,
            "pose": {"position": position, "orientation": orientation},
            "source_pose_frame": odom.header.frame_id,
            "status": "active",
        }
        with self.metadata_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.last_saved_odom = odom
        self.last_saved_time = image_time
        self.get_logger().info(
            f"Saved {frame_id}: pose_frame={pose_frame}, source_frame={odom.header.frame_id}, "
            f"image_frame={msg.header.frame_id}, pose_age={pose_age:.3f}s"
        )

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
    parser = argparse.ArgumentParser(description="Record RGB+odom keyframes from ROS2.")
    parser.add_argument(
        "--image-topic",
        default="/front_stereo_camera/left/image_raw",
        help="RGB image topic.",
    )
    parser.add_argument(
        "--odom-topic",
        default="/chassis/odom",
        help="Odometry topic used as observation pose.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/arman/test/diplom/data/keyframes",
        help="Directory for PNG keyframes and metadata.jsonl.",
    )
    parser.add_argument("--min-time-delta-sec", type=float, default=1.0)
    parser.add_argument("--min-translation-delta-m", type=float, default=0.25)
    parser.add_argument("--min-rotation-delta-rad", type=float, default=0.25)
    parser.add_argument("--max-pose-age-sec", type=float, default=0.25)
    parser.add_argument(
        "--target-frame",
        default="",
        help="Frame to store keyframe poses in. Empty keeps odometry header.frame_id.",
    )
    parser.add_argument("--tf-timeout-sec", type=float, default=2.0)
    parser.add_argument(
        "--allow-latest-tf-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the latest target-frame transform if timestamped TF is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    default_excepthook = threading.excepthook

    def recorder_excepthook(args) -> None:
        if args.exc_type is ExternalShutdownException:
            return
        default_excepthook(args)

    threading.excepthook = recorder_excepthook

    args = parse_args()
    rclpy.init()
    node = KeyframeRecorder(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
