#!/usr/bin/env python3
import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import ComputePathToPose, FollowPath, NavigateToPose
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener


MODEL_NAME = "ruclip-vit-base-patch32-224"
MODEL_REPO = "ai-forever/ruclip-vit-base-patch32-224"
MODEL_FILES = ("config.json", "bpe.model", "pytorch_model.bin")
MODEL_DIR = Path("/home/arman/test/diplom/.cache/ruclip") / MODEL_NAME
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTION = "semantic_visual_memory"

NAV2_ERROR_NAMES = {
    NavigateToPose.Result.NONE: "NONE",
    ComputePathToPose.Result.UNKNOWN: "COMPUTE_PATH_UNKNOWN",
    ComputePathToPose.Result.INVALID_PLANNER: "INVALID_PLANNER",
    ComputePathToPose.Result.TF_ERROR: "COMPUTE_PATH_TF_ERROR",
    ComputePathToPose.Result.START_OUTSIDE_MAP: "START_OUTSIDE_MAP",
    ComputePathToPose.Result.GOAL_OUTSIDE_MAP: "GOAL_OUTSIDE_MAP",
    ComputePathToPose.Result.START_OCCUPIED: "START_OCCUPIED",
    ComputePathToPose.Result.GOAL_OCCUPIED: "GOAL_OCCUPIED",
    ComputePathToPose.Result.TIMEOUT: "COMPUTE_PATH_TIMEOUT",
    ComputePathToPose.Result.NO_VALID_PATH: "NO_VALID_PATH",
    FollowPath.Result.UNKNOWN: "FOLLOW_PATH_UNKNOWN",
    FollowPath.Result.INVALID_CONTROLLER: "INVALID_CONTROLLER",
    FollowPath.Result.TF_ERROR: "FOLLOW_PATH_TF_ERROR",
    FollowPath.Result.INVALID_PATH: "INVALID_PATH",
    FollowPath.Result.PATIENCE_EXCEEDED: "PATIENCE_EXCEEDED",
    FollowPath.Result.FAILED_TO_MAKE_PROGRESS: "FAILED_TO_MAKE_PROGRESS",
    FollowPath.Result.NO_VALID_CONTROL: "NO_VALID_CONTROL",
    FollowPath.Result.CONTROLLER_TIMED_OUT: "CONTROLLER_TIMED_OUT",
}


def download_file(session: requests.Session, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    with session.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def ensure_model_files(model_dir: Path) -> Path:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    base_url = f"https://huggingface.co/{MODEL_REPO}/resolve/main"
    for filename in MODEL_FILES:
        download_file(session, f"{base_url}/{filename}", model_dir / filename)
    return model_dir


def normalize(vector):
    return vector / vector.norm(dim=-1, keepdim=True)


def build_search_filter(include_stale: bool) -> models.Filter | None:
    if include_stale:
        return None
    return models.Filter(
        must_not=[
            models.FieldCondition(
                key="status",
                match=models.MatchValue(value="stale"),
            )
        ]
    )


def valid_quaternion(pose: PoseStamped) -> bool:
    q = pose.pose.orientation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    return math.isfinite(norm) and norm > 1e-6


def normalize_quaternion(pose: PoseStamped) -> None:
    q = pose.pose.orientation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    q.x /= norm
    q.y /= norm
    q.z /= norm
    q.w /= norm


def quaternion_multiply(left: tuple[float, float, float, float], right: tuple[float, float, float, float]):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def rotate_vector(vector: tuple[float, float, float], quaternion: tuple[float, float, float, float]):
    q_conjugate = (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, (vector[0], vector[1], vector[2], 0.0)),
        q_conjugate,
    )
    return rotated[:3]


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def apply_transform(pose: PoseStamped, transform) -> PoseStamped:
    t = transform.transform.translation
    r = transform.transform.rotation
    transform_quat = (r.x, r.y, r.z, r.w)

    p = pose.pose.position
    rotated_position = rotate_vector((p.x, p.y, p.z), transform_quat)

    q = pose.pose.orientation
    pose_quat = (q.x, q.y, q.z, q.w)
    out_quat = quaternion_multiply(transform_quat, pose_quat)

    transformed = PoseStamped()
    transformed.header.frame_id = transform.header.frame_id
    transformed.header.stamp = pose.header.stamp
    transformed.pose.position.x = rotated_position[0] + t.x
    transformed.pose.position.y = rotated_position[1] + t.y
    transformed.pose.position.z = rotated_position[2] + t.z
    transformed.pose.orientation.x = out_quat[0]
    transformed.pose.orientation.y = out_quat[1]
    transformed.pose.orientation.z = out_quat[2]
    transformed.pose.orientation.w = out_quat[3]
    normalize_quaternion(transformed)
    return transformed


def payload_to_pose(payload: dict, stamp) -> PoseStamped:
    pose_data = payload.get("pose") or {}
    position = pose_data.get("position") or {}
    orientation = pose_data.get("orientation") or {}

    pose = PoseStamped()
    pose.header.frame_id = payload.get("pose_frame") or ""
    pose.header.stamp = stamp
    pose.pose.position.x = float(position.get("x", 0.0))
    pose.pose.position.y = float(position.get("y", 0.0))
    pose.pose.position.z = float(position.get("z", 0.0))
    pose.pose.orientation.x = float(orientation.get("x", 0.0))
    pose.pose.orientation.y = float(orientation.get("y", 0.0))
    pose.pose.orientation.z = float(orientation.get("z", 0.0))
    pose.pose.orientation.w = float(orientation.get("w", 1.0))
    return pose


def image_msg_to_rgb(msg: Image) -> np.ndarray:
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
    data = np.frombuffer(msg.data, dtype=np.uint8)
    image = data.reshape((msg.height, msg.step))
    image = image[:, : msg.width * channels].reshape((msg.height, msg.width, channels))

    if msg.encoding == "bgr8":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if msg.encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if msg.encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    if msg.encoding == "mono8":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


class SemanticNavToPose(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("semantic_nav_to_pose")
        self.args = args
        self.last_feedback_log_time = 0.0
        self.current_goal_handle = None
        self.latest_image: Image | None = None
        self.last_mission_check_time = 0.0
        self.mission_model = None
        self.mission_processor = None
        self.mission_device = None
        self.mission_text_vector = None
        self.action_client = ActionClient(self, NavigateToPose, args.action_name)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.latest_map: OccupancyGrid | None = None
        self.latest_costmap: OccupancyGrid | None = None
        if args.complete_on_visual_match:
            image_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.create_subscription(Image, args.image_topic, self.on_image, image_qos)
        if args.reject_goals_outside_map:
            map_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.create_subscription(OccupancyGrid, args.map_topic, self.on_map, map_qos)
        if args.reject_goals_outside_costmap:
            costmap_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.create_subscription(
                OccupancyGrid,
                args.global_costmap_topic,
                self.on_costmap,
                costmap_qos,
            )

    def on_map(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def on_costmap(self, msg: OccupancyGrid) -> None:
        self.latest_costmap = msg

    def on_image(self, msg: Image) -> None:
        self.latest_image = msg

    def encode_query(self, query: str) -> list[float]:
        import torch
        from ruclip import CLIP, RuCLIPProcessor

        model_dir = ensure_model_files(self.args.model_dir)
        device = self.args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Encoding query on device={device}")

        model = CLIP.from_pretrained(model_dir).eval().to(device)
        processor = RuCLIPProcessor.from_pretrained(model_dir)
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        with torch.inference_mode():
            text_vector = model.encode_text(inputs["input_ids"].to(device))
            text_vector = normalize(text_vector)

        self.mission_model = model
        self.mission_processor = processor
        self.mission_device = device
        self.mission_text_vector = text_vector
        return text_vector.squeeze(0).detach().cpu().tolist()

    def search_candidates(self, query_vector: list[float]):
        client = QdrantClient(url=self.args.qdrant_url)
        return client.query_points(
            collection_name=self.args.collection,
            query=query_vector,
            query_filter=build_search_filter(self.args.include_stale),
            limit=self.args.top_k,
            with_payload=True,
            with_vectors=False,
        ).points

    def transform_goal(self, pose: PoseStamped) -> PoseStamped:
        if not pose.header.frame_id:
            raise ValueError("Candidate payload has empty pose_frame")

        if not valid_quaternion(pose):
            raise ValueError("Candidate pose has invalid quaternion")
        normalize_quaternion(pose)

        if pose.header.frame_id == self.args.goal_frame:
            pose.header.stamp = self.get_clock().now().to_msg()
            return self.limit_goal_distance(pose)

        transform = self.tf_buffer.lookup_transform(
            self.args.goal_frame,
            pose.header.frame_id,
            rclpy.time.Time(),
            timeout=Duration(seconds=self.args.tf_timeout_sec),
        )
        goal = apply_transform(pose, transform)
        goal.header.stamp = self.get_clock().now().to_msg()
        return self.limit_goal_distance(goal)

    def limit_goal_distance(self, goal: PoseStamped) -> PoseStamped:
        if self.args.max_goal_distance is None:
            return goal
        if self.args.max_goal_distance <= 0.0:
            return goal

        transform = self.tf_buffer.lookup_transform(
            goal.header.frame_id,
            self.args.robot_base_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=self.args.tf_timeout_sec),
        )
        start = transform.transform.translation
        dx = goal.pose.position.x - start.x
        dy = goal.pose.position.y - start.y
        distance = math.hypot(dx, dy)
        if distance <= self.args.max_goal_distance:
            return goal

        scale = self.args.max_goal_distance / distance
        limited = PoseStamped()
        limited.header = goal.header
        limited.pose.position.x = start.x + dx * scale
        limited.pose.position.y = start.y + dy * scale
        limited.pose.position.z = goal.pose.position.z
        qx, qy, qz, qw = quaternion_from_yaw(math.atan2(dy, dx))
        limited.pose.orientation.x = qx
        limited.pose.orientation.y = qy
        limited.pose.orientation.z = qz
        limited.pose.orientation.w = qw
        self.get_logger().info(
            f"Limiting goal distance from {distance:.3f}m to "
            f"{self.args.max_goal_distance:.3f}m for smoke-test safety"
        )
        return limited

    def wait_for_map(self) -> OccupancyGrid | None:
        if not self.args.reject_goals_outside_map:
            return None
        if self.latest_map is not None:
            return self.latest_map

        deadline = time.monotonic() + self.args.map_timeout_sec
        while rclpy.ok() and self.latest_map is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.latest_map

    def wait_for_costmap(self) -> OccupancyGrid | None:
        if not self.args.reject_goals_outside_costmap:
            return None
        if self.latest_costmap is not None:
            return self.latest_costmap

        deadline = time.monotonic() + self.args.costmap_timeout_sec
        while rclpy.ok() and self.latest_costmap is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.latest_costmap

    def goal_inside_grid(
        self,
        goal: PoseStamped,
        grid: OccupancyGrid,
        label: str,
        margin: float,
    ) -> bool:
        if goal.header.frame_id != grid.header.frame_id:
            self.get_logger().warn(
                f"Cannot validate goal bounds against {label}: "
                f"goal_frame={goal.header.frame_id}, grid_frame={grid.header.frame_id}"
            )
            return True

        resolution = float(grid.info.resolution)
        width_m = float(grid.info.width) * resolution
        height_m = float(grid.info.height) * resolution
        origin = grid.info.origin
        yaw = yaw_from_quaternion(origin.orientation)
        dx = goal.pose.position.x - origin.position.x
        dy = goal.pose.position.y - origin.position.y
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        grid_x = dx * cos_yaw + dy * sin_yaw
        grid_y = -dx * sin_yaw + dy * cos_yaw
        inside = (
            margin <= grid_x < width_m - margin
            and margin <= grid_y < height_m - margin
        )
        if not inside:
            self.get_logger().warn(
                f"Goal outside current {label} bounds: frame={goal.header.frame_id} "
                f"x={goal.pose.position.x:.3f} y={goal.pose.position.y:.3f}; "
                f"{label}_x=[{origin.position.x:.3f}, {origin.position.x + width_m:.3f}] "
                f"{label}_y=[{origin.position.y:.3f}, {origin.position.y + height_m:.3f}]"
            )
        return inside

    def goal_inside_current_map(self, goal: PoseStamped) -> bool:
        if not self.args.reject_goals_outside_map:
            return True
        grid = self.latest_map or self.wait_for_map()
        if grid is None:
            self.get_logger().warn(
                f"No {self.args.map_topic} received; cannot validate goal bounds"
            )
            return True
        return self.goal_inside_grid(
            goal,
            grid,
            "map",
            self.args.map_bounds_margin_m,
        )

    def goal_inside_current_costmap(self, goal: PoseStamped) -> bool:
        if not self.args.reject_goals_outside_costmap:
            return True
        grid = self.latest_costmap or self.wait_for_costmap()
        if grid is None:
            self.get_logger().warn(
                f"No {self.args.global_costmap_topic} received; cannot validate costmap bounds"
            )
            return True
        return self.goal_inside_grid(
            goal,
            grid,
            "global_costmap",
            self.args.costmap_bounds_margin_m,
        )

    def wait_for_future(self, future, timeout_sec: float | None) -> bool:
        if timeout_sec is None or timeout_sec <= 0.0:
            rclpy.spin_until_future_complete(self, future)
            return future.done()

        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return future.done()

    def visual_match_score(self) -> float | None:
        if not self.args.complete_on_visual_match:
            return None
        if self.latest_image is None:
            return None
        if (
            self.mission_model is None
            or self.mission_processor is None
            or self.mission_device is None
            or self.mission_text_vector is None
        ):
            return None

        import torch
        from PIL import Image as PILImage

        rgb = image_msg_to_rgb(self.latest_image)
        image = PILImage.fromarray(rgb)
        inputs = self.mission_processor(text="", images=[image], return_tensors="pt", padding=True)
        with torch.inference_mode():
            image_vector = self.mission_model.encode_image(
                inputs["pixel_values"].to(self.mission_device)
            )
            image_vector = normalize(image_vector)
            return float((image_vector @ self.mission_text_vector.T).item())

    def check_visual_mission_complete(self, memory_id: str) -> bool:
        now = time.monotonic()
        if now - self.last_mission_check_time < self.args.mission_check_period_sec:
            return False
        self.last_mission_check_time = now

        try:
            score = self.visual_match_score()
        except Exception as exc:
            self.get_logger().warn(f"Visual mission check failed: {exc}")
            return False

        if score is None:
            self.get_logger().info("Visual mission check: waiting for camera frame")
            return False

        self.get_logger().info(
            f"Visual mission check for {memory_id}: query={self.args.query!r} "
            f"score={score:.4f}, threshold={self.args.mission_match_threshold:.4f}"
        )
        return score >= self.args.mission_match_threshold

    def wait_for_navigation_or_visual_match(self, goal_handle, result_future, memory_id: str) -> bool:
        deadline = None
        if self.args.result_timeout_sec and self.args.result_timeout_sec > 0.0:
            deadline = time.monotonic() + self.args.result_timeout_sec

        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.check_visual_mission_complete(memory_id):
                self.get_logger().info(
                    f"Mission complete: visual match for {self.args.query!r} "
                    f"detected while navigating to {memory_id}; canceling Nav2 goal"
                )
                cancel_future = goal_handle.cancel_goal_async()
                self.wait_for_future(cancel_future, self.args.action_timeout_sec)
                self.current_goal_handle = None
                return True
            if deadline is not None and time.monotonic() >= deadline:
                self.get_logger().warn(
                    f"Navigation result timed out for {memory_id}; requesting cancel"
                )
                cancel_future = goal_handle.cancel_goal_async()
                self.wait_for_future(cancel_future, self.args.action_timeout_sec)
                self.current_goal_handle = None
                return False

        return self.handle_navigation_result(result_future, memory_id)

    def handle_navigation_result(self, result_future, memory_id: str) -> bool:
        result = result_future.result()
        self.current_goal_handle = None
        if result is None:
            self.get_logger().error("Nav2 returned no result")
            return False

        status_name = GoalStatus.STATUS_SUCCEEDED
        if result.status == status_name and result.result.error_code == NavigateToPose.Result.NONE:
            self.get_logger().info(f"Navigation succeeded for {memory_id}")
            return True

        error_name = NAV2_ERROR_NAMES.get(result.result.error_code, "UNKNOWN_NAV2_ERROR")
        self.get_logger().warn(
            f"Navigation failed for {memory_id}: status={result.status}, "
            f"error_code={result.result.error_code} ({error_name}), "
            f"error_msg={result.result.error_msg!r}"
        )
        return False

    def send_goal(self, goal_pose: PoseStamped, memory_id: str) -> bool:
        if self.args.dry_run:
            self.get_logger().info(
                f"Dry run: would send {memory_id} to {goal_pose.header.frame_id} "
                f"x={goal_pose.pose.position.x:.3f} y={goal_pose.pose.position.y:.3f}"
            )
            return True

        self.get_logger().info(f"Waiting for Nav2 action server {self.args.action_name}")
        if not self.action_client.wait_for_server(timeout_sec=self.args.action_timeout_sec):
            self.get_logger().error(f"Nav2 action server is not available: {self.args.action_name}")
            return False

        goal = NavigateToPose.Goal()
        goal.pose = goal_pose
        goal.behavior_tree = self.args.behavior_tree
        self.get_logger().info(
            f"Sending {memory_id}: frame={goal_pose.header.frame_id} "
            f"x={goal_pose.pose.position.x:.3f} y={goal_pose.pose.position.y:.3f}"
        )

        goal_future = self.action_client.send_goal_async(
            goal,
            feedback_callback=self.on_feedback,
        )
        if not self.wait_for_future(goal_future, self.args.action_timeout_sec):
            self.get_logger().error(f"Timed out while sending {memory_id}")
            return False
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn(f"Nav2 rejected candidate {memory_id}")
            return False

        self.get_logger().info(f"Nav2 accepted candidate {memory_id}")
        self.current_goal_handle = goal_handle
        if self.args.no_wait_result:
            return True

        result_future = goal_handle.get_result_async()
        return self.wait_for_navigation_or_visual_match(goal_handle, result_future, memory_id)

    def cancel_current_goal(self) -> None:
        goal_handle = self.current_goal_handle
        if goal_handle is None:
            return
        self.get_logger().warn("Cancelling active Nav2 goal before shutdown")
        try:
            cancel_future = goal_handle.cancel_goal_async()
            self.wait_for_future(cancel_future, self.args.action_timeout_sec)
        except Exception as exc:
            self.get_logger().warn(f"Failed to cancel active Nav2 goal: {exc}")
        finally:
            self.current_goal_handle = None

    def on_feedback(self, feedback_msg) -> None:
        now = time.monotonic()
        if now - self.last_feedback_log_time < self.args.feedback_log_period_sec:
            return
        self.last_feedback_log_time = now

        feedback = feedback_msg.feedback
        self.get_logger().info(
            f"Nav2 feedback: distance_remaining={feedback.distance_remaining:.3f}, "
            f"recoveries={feedback.number_of_recoveries}"
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

    def run(self) -> int:
        query_vector = self.encode_query(self.args.query)
        candidates = self.search_candidates(query_vector)
        if not candidates:
            self.get_logger().error("Qdrant returned no candidates")
            return 1

        self.get_logger().info(f"Received {len(candidates)} candidates from Qdrant")
        for rank, candidate in enumerate(candidates, start=1):
            payload = candidate.payload or {}
            memory_id = payload.get("memory_id") or str(candidate.id)
            try:
                observation_pose = payload_to_pose(payload, self.get_clock().now().to_msg())
                goal_pose = self.transform_goal(observation_pose)
                if not self.goal_inside_current_map(goal_pose):
                    raise ValueError("Candidate goal is outside current /map")
                if not self.goal_inside_current_costmap(goal_pose):
                    raise ValueError("Candidate goal is outside current global costmap")
            except (ValueError, RuntimeError, TransformException) as exc:
                self.get_logger().warn(
                    f"Skipping candidate #{rank} {memory_id}: {exc}"
                )
                continue

            self.get_logger().info(
                f"Candidate #{rank}: {memory_id}, score={candidate.score:.4f}, "
                f"source_frame={payload.get('pose_frame')}, goal_frame={goal_pose.header.frame_id}"
            )
            if self.send_goal(goal_pose, memory_id):
                return 0
            if not self.args.try_next_candidates:
                self.get_logger().warn(
                    f"Stopping after failed candidate #{rank} {memory_id}; "
                    "automatic fallback is disabled"
                )
                return 2

        self.get_logger().warn("All candidates failed or were skipped")
        return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search semantic visual memory and send the best observation pose to Nav2."
    )
    parser.add_argument("query", help="Russian text query, for example: 'дорога'")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--try-next-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If the selected Nav2 goal fails, try lower-ranked semantic candidates. Disabled by default to avoid switching targets mid-mission.",
    )
    parser.add_argument("--include-stale", action="store_true")
    parser.add_argument("--goal-frame", default="map")
    parser.add_argument("--map-topic", default="/map")
    parser.add_argument("--map-timeout-sec", type=float, default=2.0)
    parser.add_argument("--map-bounds-margin-m", type=float, default=0.0)
    parser.add_argument("--global-costmap-topic", default="/global_costmap/costmap")
    parser.add_argument("--costmap-timeout-sec", type=float, default=2.0)
    parser.add_argument("--costmap-bounds-margin-m", type=float, default=0.25)
    parser.add_argument(
        "--reject-goals-outside-map",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip semantic candidates whose goal pose is outside /map. Disabled by default because SLAM /map may lag behind the rolling Nav2 costmap.",
    )
    parser.add_argument(
        "--reject-goals-outside-costmap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip semantic candidates whose goal pose is outside the current Nav2 global costmap. Disabled by default so exact stored coordinates are always sent to Nav2.",
    )
    parser.add_argument("--action-name", default="/navigate_to_pose")
    parser.add_argument("--robot-base-frame", default="base_link")
    parser.add_argument("--image-topic", default="/front_stereo_camera/left/image_raw")
    parser.add_argument(
        "--complete-on-visual-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="While driving to the exact coordinates, stop early and report success if the live camera matches the query.",
    )
    parser.add_argument("--mission-match-threshold", type=float, default=0.30)
    parser.add_argument("--mission-check-period-sec", type=float, default=1.5)
    parser.add_argument("--behavior-tree", default="")
    parser.add_argument("--tf-timeout-sec", type=float, default=2.0)
    parser.add_argument("--action-timeout-sec", type=float, default=10.0)
    parser.add_argument(
        "--result-timeout-sec",
        type=float,
        default=0.0,
        help="Cancel the Nav2 goal if no action result is received within this time. 0 disables this client-side timeout.",
    )
    parser.add_argument(
        "--max-goal-distance",
        type=float,
        default=0.0,
        help="0 disables goal distance limiting; otherwise sends a nearer pose along the same ray.",
    )
    parser.add_argument("--feedback-log-period-sec", type=float, default=1.0)
    parser.add_argument("--no-wait-result", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Default: cuda if available, otherwise cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mission_check_period_sec <= 0.0:
        raise ValueError("--mission-check-period-sec must be > 0")
    if not 0.0 <= args.mission_match_threshold <= 1.0:
        raise ValueError("--mission-match-threshold must be between 0 and 1")
    rclpy.init()
    node = SemanticNavToPose(args)
    try:
        exit_code = node.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.cancel_current_goal()
        exit_code = 130
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
