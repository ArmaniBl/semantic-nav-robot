#!/usr/bin/env python3
import argparse
import math
import time
from pathlib import Path

import requests
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener


MODEL_NAME = "ruclip-vit-base-patch32-224"
MODEL_REPO = "ai-forever/ruclip-vit-base-patch32-224"
MODEL_FILES = ("config.json", "bpe.model", "pytorch_model.bin")
MODEL_DIR = Path("/home/arman/test/diplom/.cache/ruclip") / MODEL_NAME
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTION = "semantic_visual_memory"


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


class SemanticNavToPose(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("semantic_nav_to_pose")
        self.args = args
        self.last_feedback_log_time = 0.0
        self.current_goal_handle = None
        self.action_client = ActionClient(self, NavigateToPose, args.action_name)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

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
            text_vector = normalize(text_vector).squeeze(0).detach().cpu().tolist()
        return text_vector

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
            raise ValueError("--max-goal-distance must be > 0")

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

    def wait_for_future(self, future, timeout_sec: float | None) -> bool:
        if timeout_sec is None:
            rclpy.spin_until_future_complete(self, future)
            return future.done()

        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return future.done()

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
        if not self.wait_for_future(result_future, self.args.result_timeout_sec):
            self.get_logger().warn(
                f"Navigation result timed out for {memory_id}; requesting cancel"
            )
            cancel_future = goal_handle.cancel_goal_async()
            self.wait_for_future(cancel_future, self.args.action_timeout_sec)
            self.current_goal_handle = None
            return False

        result = result_future.result()
        self.current_goal_handle = None
        if result is None:
            self.get_logger().error("Nav2 returned no result")
            return False

        status_name = GoalStatus.STATUS_SUCCEEDED
        if result.status == status_name and result.result.error_code == NavigateToPose.Result.NONE:
            self.get_logger().info(f"Navigation succeeded for {memory_id}")
            return True

        self.get_logger().warn(
            f"Navigation failed for {memory_id}: status={result.status}, "
            f"error_code={result.result.error_code}, error_msg={result.result.error_msg!r}"
        )
        return False

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

        self.get_logger().error("All candidates failed or were skipped")
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
    parser.add_argument("--include-stale", action="store_true")
    parser.add_argument("--goal-frame", default="map")
    parser.add_argument("--action-name", default="/navigate_to_pose")
    parser.add_argument("--robot-base-frame", default="base_link")
    parser.add_argument("--behavior-tree", default="")
    parser.add_argument("--tf-timeout-sec", type=float, default=2.0)
    parser.add_argument("--action-timeout-sec", type=float, default=10.0)
    parser.add_argument(
        "--result-timeout-sec",
        type=float,
        default=180.0,
        help="Cancel the Nav2 goal if no action result is received within this time.",
    )
    parser.add_argument(
        "--max-goal-distance",
        type=float,
        default=None,
        help="Optional cap for live smoke tests; sends a nearer pose along the same ray.",
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
