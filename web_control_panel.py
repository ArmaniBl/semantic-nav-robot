#!/usr/bin/env python3
import argparse
import html
import json
import mimetypes
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_TOPIC = "/front_stereo_camera/left/image_raw"
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"
MAX_SERVER_LOGS = 250
EVENT_STREAM_INITIAL_BACKLOG = 80


def stamp_to_sec(msg) -> float:
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def image_msg_to_bgr(msg: Image) -> np.ndarray:
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

    if msg.encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if msg.encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if msg.encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if msg.encoding == "mono8":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame_condition = threading.Condition(self.lock)
        self.latest_jpeg: bytes | None = None
        self.latest_frame_wall_time = 0.0
        self.latest_frame_ros_time = 0.0
        self.latest_image_frame = ""
        self.frame_count = 0
        self.logs: deque[dict] = deque(maxlen=MAX_SERVER_LOGS)
        self.next_log_id = 1
        self.nav_process: subprocess.Popen | None = None
        self.nav_started_at = 0.0
        self.nav_query = ""
        self.nav_exit_code: int | None = None
        self.slam_process: subprocess.Popen | None = None
        self.slam_started_at = 0.0
        self.slam_exit_code: int | None = None
        self.record_process: subprocess.Popen | None = None
        self.record_started_at = 0.0
        self.record_exit_code: int | None = None
        self.record_output_dir = ""
        self.record_target_collection = ""
        self.record_finalizing = False
        self.record_last_result: dict | None = None
        self.qdrant_status = {
            "ready": False,
            "summary": "Qdrant unknown",
            "url": "",
        }
        self.system_status = {
            "map_topic": False,
            "navigate_action": False,
            "goal_frame": False,
            "ready": False,
            "summary": "SLAM/Nav2 unknown",
        }

    def add_log(self, source: str, message: str) -> None:
        line = message.rstrip()
        if not line:
            return
        if self.is_noisy_log(line):
            return
        with self.lock:
            item = {
                "id": self.next_log_id,
                "time": time.strftime("%H:%M:%S"),
                "source": source,
                "message": line,
            }
            self.next_log_id += 1
            self.logs.append(item)

    def is_noisy_log(self, line: str) -> bool:
        noisy_patterns = (
            "TF_OLD_DATA ignoring data from the past",
            "Possible reasons are listed at http://wiki.ros.org/tf/Errors%20explained",
            "Message Filter dropping message",
            "Passing new path to controller.",
        )
        return any(pattern in line for pattern in noisy_patterns)

    def set_frame(self, jpeg: bytes, msg: Image) -> None:
        with self.frame_condition:
            self.latest_jpeg = jpeg
            self.latest_frame_wall_time = time.monotonic()
            self.latest_frame_ros_time = stamp_to_sec(msg)
            self.latest_image_frame = msg.header.frame_id
            self.frame_count += 1
            self.frame_condition.notify_all()

    def status(self) -> dict:
        with self.lock:
            proc = self.nav_process
            running = proc is not None and proc.poll() is None
            frame_age = None
            if self.latest_frame_wall_time:
                frame_age = round(time.monotonic() - self.latest_frame_wall_time, 3)
            nav_runtime = None
            if running:
                nav_runtime = round(time.monotonic() - self.nav_started_at, 1)
            slam_proc = self.slam_process
            slam_managed = slam_proc is not None and slam_proc.poll() is None
            system_snapshot = dict(self.system_status)
            slam_external = (
                not slam_managed
                and bool(system_snapshot.get("map_topic"))
                and bool(system_snapshot.get("navigate_action"))
            )
            slam_running = slam_managed or slam_external
            slam_runtime = None
            if slam_managed:
                slam_runtime = round(time.monotonic() - self.slam_started_at, 1)
            record_proc = self.record_process
            record_running = record_proc is not None and record_proc.poll() is None
            record_runtime = None
            if record_running:
                record_runtime = round(time.monotonic() - self.record_started_at, 1)
            return {
                "camera": {
                    "has_frame": self.latest_jpeg is not None,
                    "frame_age_sec": frame_age,
                    "frame_count": self.frame_count,
                    "image_frame": self.latest_image_frame,
                    "ros_time_sec": round(self.latest_frame_ros_time, 3),
                },
                "navigation": {
                    "running": running,
                    "query": self.nav_query if running else "",
                    "runtime_sec": nav_runtime,
                    "last_exit_code": self.nav_exit_code,
                },
                "slam": {
                    "running": slam_running,
                    "managed": slam_managed,
                    "external": slam_external,
                    "runtime_sec": slam_runtime,
                    "last_exit_code": self.slam_exit_code,
                },
                "recording": {
                    "running": record_running,
                    "finalizing": self.record_finalizing,
                    "runtime_sec": record_runtime,
                    "last_exit_code": self.record_exit_code,
                    "output_dir": self.record_output_dir,
                    "target_collection": self.record_target_collection,
                    "last_result": self.record_last_result,
                },
                "qdrant": dict(self.qdrant_status),
                "system": system_snapshot,
            }

    def set_system_status(self, status: dict) -> None:
        with self.lock:
            self.system_status = status

    def navigation_ready(self) -> tuple[bool, str]:
        with self.lock:
            status = dict(self.system_status)
            qdrant = dict(self.qdrant_status)
        ready = bool(status.get("ready")) and bool(qdrant.get("ready"))
        if ready:
            return True, "SLAM/Nav2 and Qdrant ready"
        parts = []
        if not status.get("ready"):
            parts.append(str(status.get("summary", "SLAM/Nav2 unknown")))
        if not qdrant.get("ready"):
            parts.append(str(qdrant.get("summary", "Qdrant unknown")))
        return False, "; ".join(parts)

    def navigation_request_ready(self) -> tuple[bool, str]:
        with self.lock:
            status = dict(self.system_status)
            qdrant = dict(self.qdrant_status)
        parts = []
        if not status.get("clock_topic"):
            parts.append("missing /clock publisher")
        if not status.get("map_topic"):
            parts.append("missing /map")
        if not status.get("navigate_action"):
            parts.append("missing /navigate_to_pose")
        if not status.get("goal_frame"):
            parts.append(f"missing {self.app_args.goal_frame}->{self.app_args.robot_base_frame}")
        if not qdrant.get("ready"):
            parts.append(str(qdrant.get("summary", "Qdrant unknown")))
        if parts:
            return False, "; ".join(parts)
        return True, "Nav2 action/map and Qdrant ready"

    def recording_ready(self) -> tuple[bool, str]:
        with self.lock:
            status = dict(self.system_status)
            qdrant = dict(self.qdrant_status)
        parts = []
        if not status.get("clock_topic"):
            parts.append("missing /clock publisher")
        if not status.get("map_topic"):
            parts.append("missing /map")
        if not status.get("goal_frame"):
            parts.append(f"missing {self.app_args.goal_frame}->{self.app_args.robot_base_frame}")
        if not qdrant.get("ready"):
            parts.append(str(qdrant.get("summary", "Qdrant unknown")))
        if parts:
            return False, "; ".join(parts)
        return True, "SLAM map/TF and Qdrant ready"

    def set_qdrant_status(self, status: dict) -> None:
        with self.lock:
            self.qdrant_status = status


class CameraSubscriber(Node):
    def __init__(self, state: SharedState, args: argparse.Namespace) -> None:
        super().__init__("semantic_nav_web_camera")
        self.state = state
        self.args = args
        self.last_encoded_wall_time = 0.0
        self.min_frame_period = 1.0 / args.max_stream_fps if args.max_stream_fps > 0 else 0.0
        reliability = (
            ReliabilityPolicy.RELIABLE
            if args.image_reliability == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        )
        qos = QoSProfile(
            reliability=reliability,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Image, args.image_topic, self.on_image, qos)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.create_timer(1.0, self.update_system_status)
        self.create_timer(2.0, self.update_qdrant_status)
        self.get_logger().info(
            f"Streaming camera topic: {args.image_topic} "
            f"(reliability={args.image_reliability})"
        )

    def on_image(self, msg: Image) -> None:
        try:
            now = time.monotonic()
            if (
                self.min_frame_period > 0
                and self.last_encoded_wall_time
                and now - self.last_encoded_wall_time < self.min_frame_period
            ):
                return
            self.last_encoded_wall_time = now
            image = image_msg_to_bgr(msg)
            if self.args.stream_width and image.shape[1] > self.args.stream_width:
                scale = self.args.stream_width / image.shape[1]
                image = cv2.resize(
                    image,
                    (self.args.stream_width, int(image.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality],
            )
            if ok:
                self.state.set_frame(encoded.tobytes(), msg)
        except Exception as exc:
            self.state.add_log("camera", f"Failed to encode image: {exc}")

    def update_system_status(self) -> None:
        clock_topic = bool(self.get_publishers_info_by_topic("/clock"))
        map_topic = bool(self.get_publishers_info_by_topic(self.args.map_topic))
        action_prefix = self.args.action_name.rstrip("/")
        action_status_topic = f"{action_prefix}/_action/status"
        navigate_action = bool(self.get_publishers_info_by_topic(action_status_topic))
        goal_frame = bool(self.tf_buffer.can_transform(
            self.args.goal_frame,
            self.args.robot_base_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=0.0),
        ))

        missing = []
        if not clock_topic:
            missing.append("/clock publisher")
        if not map_topic:
            missing.append(self.args.map_topic)
        if not navigate_action:
            missing.append(self.args.action_name)
        if not goal_frame:
            missing.append(f"{self.args.goal_frame}->{self.args.robot_base_frame}")
        ready = not missing
        summary = "SLAM/Nav2 ready" if ready else "Missing: " + ", ".join(missing)
        self.state.set_system_status(
            {
                "clock_topic": clock_topic,
                "map_topic": map_topic,
                "navigate_action": navigate_action,
                "goal_frame": goal_frame,
                "ready": ready,
                "summary": summary,
            }
        )

    def update_qdrant_status(self) -> None:
        try:
            with urlopen(self.args.qdrant_url, timeout=0.5) as response:
                ready = 200 <= response.status < 300
                summary = "Qdrant ready" if ready else f"Qdrant HTTP {response.status}"
        except (OSError, URLError) as exc:
            ready = False
            summary = f"Qdrant unavailable: {exc}"
        self.state.set_qdrant_status(
            {
                "ready": ready,
                "summary": summary,
                "url": self.args.qdrant_url,
            }
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


def build_nav_command(args: argparse.Namespace, query: str) -> list[str]:
    script = PROJECT_ROOT / "semantic_nav_to_pose.py"
    python_path = PROJECT_ROOT / ".ruclip_venv" / "bin" / "python"
    command = [
        str(python_path),
        str(script),
        query,
        "--collection",
        args.collection,
        "--top-k",
        str(args.top_k),
        "--goal-frame",
        args.goal_frame,
        "--map-topic",
        args.map_topic,
        "--global-costmap-topic",
        args.global_costmap_topic,
        "--action-name",
        args.action_name,
        "--robot-base-frame",
        args.robot_base_frame,
        "--image-topic",
        args.image_topic,
        "--mission-match-threshold",
        str(args.mission_match_threshold),
        "--mission-check-period-sec",
        str(args.mission_check_period_sec),
        "--result-timeout-sec",
        str(args.result_timeout_sec),
        "--feedback-log-period-sec",
        str(args.feedback_log_period_sec),
    ]
    if args.max_goal_distance and args.max_goal_distance > 0.0:
        command.extend(["--max-goal-distance", str(args.max_goal_distance)])
    return command


def build_slam_command(args: argparse.Namespace) -> list[str]:
    return [
        "ros2",
        "launch",
        str(PROJECT_ROOT / "launch" / "slam_nav2_launch.py"),
        "log_level:=" + args.slam_log_level,
    ]


def build_qdrant_start_command() -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(PROJECT_ROOT / "compose.qdrant.yml"),
        "up",
        "-d",
    ]


def build_qdrant_stop_command() -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(PROJECT_ROOT / "compose.qdrant.yml"),
        "down",
    ]


def build_record_command(args: argparse.Namespace, keyframes_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "ros_keyframe_recorder.py"),
        "--output-dir",
        str(keyframes_dir),
        "--image-topic",
        args.image_topic,
        "--odom-topic",
        args.record_odom_topic,
        "--min-time-delta-sec",
        str(args.record_min_time_delta_sec),
        "--min-translation-delta-m",
        str(args.record_min_translation_delta_m),
        "--min-rotation-delta-rad",
        str(args.record_min_rotation_delta_rad),
        "--max-pose-age-sec",
        str(args.record_max_pose_age_sec),
        "--target-frame",
        args.goal_frame,
        "--tf-timeout-sec",
        str(args.record_tf_timeout_sec),
    ]


def point_payload_to_memory_item(point: dict, score: float | None = None) -> dict:
    payload = point.get("payload") or {}
    pose = payload.get("pose") or {}
    position = pose.get("position") or {}
    orientation = pose.get("orientation") or {}
    return {
        "id": str(point.get("id", "")),
        "score": score if score is not None else point.get("score"),
        "memory_id": payload.get("memory_id") or str(point.get("id", "")),
        "image_path": payload.get("image_path") or "",
        "run_id": payload.get("run_id") or run_id_from_path(payload.get("image_path") or ""),
        "timestamp": payload.get("timestamp"),
        "image_topic": payload.get("image_topic"),
        "image_frame": payload.get("image_frame"),
        "pose_topic": payload.get("pose_topic"),
        "pose_frame": payload.get("pose_frame"),
        "source_pose_frame": payload.get("source_pose_frame"),
        "child_frame_id": payload.get("child_frame_id"),
        "pose_age_sec": payload.get("pose_age_sec"),
        "position": {
            "x": position.get("x"),
            "y": position.get("y"),
            "z": position.get("z"),
        },
        "orientation": {
            "x": orientation.get("x"),
            "y": orientation.get("y"),
            "z": orientation.get("z"),
            "w": orientation.get("w"),
        },
        "status": payload.get("status"),
        "embedding_model": payload.get("embedding_model"),
        "map": payload.get("map"),
        "map_yaml": payload.get("map_yaml"),
        "map_image": payload.get("map_image"),
    }


def run_id_from_path(raw_path: str) -> str | None:
    if not raw_path:
        return None
    parts = Path(raw_path).parts
    for index, part in enumerate(parts[:-1]):
        if part == "recordings" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_command_capture(command: list[str], timeout_sec: float) -> str:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    output = result.stdout or ""
    if result.returncode != 0:
        raise RuntimeError(output.strip() or f"command exited with code {result.returncode}")
    return output


def add_recording_run_id(metadata_path: Path, run_id: str) -> int:
    records = read_jsonl(metadata_path)
    temp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        for record in records:
            record["run_id"] = run_id
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(metadata_path)
    return len(records)


def load_recording_into_qdrant(args: argparse.Namespace, run_dir: Path, collection_name: str) -> dict:
    keyframes_dir = run_dir / "keyframes"
    metadata_path = keyframes_dir / "metadata.jsonl"
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        raise FileNotFoundError(f"Recording has no metadata: {metadata_path}")
    if not collection_name:
        raise ValueError("target collection is empty")
    record_count = add_recording_run_id(metadata_path, run_dir.name)
    embeddings_path = keyframes_dir / "embeddings.jsonl"
    python_path = PROJECT_ROOT / ".ruclip_venv" / "bin" / "python"

    embed_output = run_command_capture(
        [
            str(python_path),
            str(PROJECT_ROOT / "ruclip_embed_keyframes.py"),
            "--metadata",
            str(metadata_path),
            "--output",
            str(embeddings_path),
        ],
        args.memory_load_timeout_sec,
    )

    load_command = [
        str(python_path),
        str(PROJECT_ROOT / "qdrant_load_keyframes.py"),
        "--embeddings",
        str(embeddings_path),
        "--url",
        args.qdrant_url,
        "--collection",
        collection_name,
    ]
    if args.record_recreate_collection:
        load_command.append("--recreate")
    load_output = run_command_capture(load_command, args.memory_load_timeout_sec)
    return {
        "run": run_dir.name,
        "collection": collection_name,
        "records": record_count,
        "metadata": str(metadata_path),
        "embeddings": str(embeddings_path),
        "embed_output": embed_output,
        "load_output": load_output,
    }


def qdrant_collections(args: argparse.Namespace) -> list[str]:
    url = args.qdrant_url.rstrip("/") + "/collections"
    with urlopen(url, timeout=args.qdrant_timeout_sec) as response:
        data = json.loads(response.read().decode("utf-8"))
    collections = (data.get("result") or {}).get("collections") or []
    return sorted(str(item.get("name", "")) for item in collections if item.get("name"))


def delete_qdrant_collection(args: argparse.Namespace, collection_name: str) -> None:
    if not collection_name:
        raise ValueError("collection name is empty")
    url = args.qdrant_url.rstrip("/") + "/collections/" + quote(collection_name, safe="")
    request = Request(url, method="DELETE")
    with urlopen(request, timeout=args.qdrant_timeout_sec) as response:
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"Qdrant returned HTTP {response.status}")


def qdrant_scroll_memory(args: argparse.Namespace) -> dict:
    limit = max(1, min(args.memory_limit, 512))
    collections = qdrant_collections(args)
    if args.collection not in collections:
        return {
            "items": [],
            "limit": limit,
            "collection": args.collection,
            "collections": collections,
            "collection_missing": True,
            "next_page_offset": None,
            "qdrant_run_counts": {},
        }
    url = args.qdrant_url.rstrip("/") + f"/collections/{args.collection}/points/scroll"
    body = json.dumps(
        {
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
    ).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=args.qdrant_timeout_sec) as response:
        data = json.loads(response.read().decode("utf-8"))
    points = (data.get("result") or {}).get("points") or []
    items = [point_payload_to_memory_item(point) for point in points]
    run_counts: dict[str, int] = {}
    for item in items:
        run_id = item.get("run_id") or "<unknown>"
        run_counts[run_id] = run_counts.get(run_id, 0) + 1
    return {
        "items": items,
        "limit": limit,
        "collection": args.collection,
        "collections": collections,
        "collection_missing": False,
        "next_page_offset": (data.get("result") or {}).get("next_page_offset"),
        "qdrant_run_counts": run_counts,
    }


def safe_image_path(raw_path: str) -> Path:
    if not raw_path:
        raise ValueError("empty image path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("image path is outside project root")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    if resolved.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("unsupported image type")
    return resolved


def search_memory_preview(args: argparse.Namespace, query: str) -> dict:
    python_path = PROJECT_ROOT / ".ruclip_venv" / "bin" / "python"
    command = [
        str(python_path),
        str(PROJECT_ROOT / "semantic_search_qdrant.py"),
        query,
        "--url",
        args.qdrant_url,
        "--collection",
        args.collection,
        "--top-k",
        "1",
        "--json",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=args.memory_search_timeout_sec,
        check=False,
    )
    if result.returncode != 0:
        output = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(output or f"semantic search exited with code {result.returncode}")
    stdout_lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    if not stdout_lines:
        raise RuntimeError("semantic search returned empty output")
    data = json.loads(stdout_lines[-1])
    results = data.get("results") or []
    selected = None
    if results:
        point = results[0]
        selected = point_payload_to_memory_item(point, score=point.get("score"))
    return {
        "query": query,
        "selected": selected,
        "raw_count": len(results),
    }


def stream_process_output(state: SharedState, proc: subprocess.Popen, kind: str) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        state.add_log(kind, line)
    exit_code = proc.wait()
    with state.lock:
        if kind == "nav" and state.nav_process is proc:
            state.nav_exit_code = exit_code
            state.nav_process = None
            query = state.nav_query
            state.nav_query = ""
            suffix = f" {query!r}".strip()
        elif kind == "slam" and state.slam_process is proc:
            state.slam_exit_code = exit_code
            state.slam_process = None
            suffix = ""
        elif kind == "record" and state.record_process is proc:
            state.record_exit_code = exit_code
            state.record_process = None
            suffix = ""
        else:
            suffix = ""
    state.add_log(kind, f"process exited with code {exit_code}{suffix}")


def finalize_recording(
    state: SharedState,
    args: argparse.Namespace,
    proc: subprocess.Popen,
    run_dir: Path,
    collection_name: str,
) -> None:
    while proc.poll() is None:
        time.sleep(0.2)
    with state.lock:
        if state.record_finalizing:
            return
        state.record_finalizing = True
        state.record_last_result = None
    try:
        state.add_log(
            "record",
            f"embedding and loading recording into Qdrant: {run_dir.name} -> {collection_name}",
        )
        result = load_recording_into_qdrant(args, run_dir, collection_name)
        for line in (result.get("embed_output") or "").splitlines():
            state.add_log("record", line)
        for line in (result.get("load_output") or "").splitlines():
            state.add_log("record", line)
        with state.lock:
            args.collection = result["collection"]
            state.record_last_result = {
                "ok": True,
                "run": result["run"],
                "records": result["records"],
                "collection": result["collection"],
            }
        state.add_log(
            "record",
            f"recording loaded: run={result['run']} records={result['records']} collection={result['collection']}",
        )
    except Exception as exc:
        with state.lock:
            state.record_last_result = {"ok": False, "error": str(exc), "run": run_dir.name}
        state.add_log("record", f"failed to load recording: {exc}")
    finally:
        with state.lock:
            state.record_finalizing = False


class ControlHandler(BaseHTTPRequestHandler):
    server_version = "SemanticNavWeb/0.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.app_args.client_timeout_sec)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_html(INDEX_HTML)
        elif path == "/camera.jpg":
            self.send_camera_snapshot()
        elif path == "/camera.mjpg":
            self.stream_camera()
        elif path == "/api/status":
            self.send_json(self.state.status())
        elif path == "/api/logs":
            self.send_json({"logs": self.get_logs()})
        elif path == "/api/events":
            self.stream_events()
        elif path == "/api/memory":
            self.handle_memory_list()
        elif path == "/api/memory/image":
            self.handle_memory_image()
        elif path == "/api/qdrant/collections":
            self.handle_qdrant_collections()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_head_response("text/html; charset=utf-8", len(INDEX_HTML.encode("utf-8")))
        elif path in {"/camera.jpg", "/api/camera.jpg"}:
            with self.state.lock:
                jpeg = self.state.latest_jpeg
            self.send_head_response("image/jpeg", len(jpeg or PLACEHOLDER_JPEG), no_store=True)
        elif path == "/api/status":
            payload = json.dumps(self.state.status(), ensure_ascii=False).encode("utf-8")
            self.send_head_response("application/json; charset=utf-8", len(payload))
        elif path == "/api/logs":
            payload = json.dumps({"logs": self.get_logs()}, ensure_ascii=False).encode("utf-8")
            self.send_head_response("application/json; charset=utf-8", len(payload))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/navigate":
            self.handle_navigate()
        elif path == "/api/stop":
            self.handle_stop()
        elif path == "/api/slam/start":
            self.handle_slam_start()
        elif path == "/api/slam/stop":
            self.handle_slam_stop()
        elif path == "/api/qdrant/start":
            self.handle_qdrant_start()
        elif path == "/api/qdrant/stop":
            self.handle_qdrant_stop()
        elif path == "/api/qdrant/collections/select":
            self.handle_qdrant_select_collection()
        elif path == "/api/qdrant/collections/delete":
            self.handle_qdrant_delete_collection()
        elif path == "/api/record/start":
            self.handle_record_start()
        elif path == "/api/record/stop":
            self.handle_record_stop()
        elif path == "/api/memory/search":
            self.handle_memory_search()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    @property
    def state(self) -> SharedState:
        return self.server.state

    @property
    def app_args(self) -> argparse.Namespace:
        return self.server.app_args

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length) if length else b"{}"
        return json.loads(data.decode("utf-8"))

    def send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return

    def send_head_response(
        self,
        content_type: str,
        content_length: int,
        status: HTTPStatus = HTTPStatus.OK,
        no_store: bool = False,
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(content_length))
            if no_store:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return

    def send_bytes(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return

    def get_logs(self) -> list[dict]:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        last_id = int((query.get("last_id") or query.get("after") or ["0"])[0])
        with self.state.lock:
            return [item for item in self.state.logs if item["id"] > last_id]

    def handle_memory_list(self) -> None:
        try:
            try:
                payload = qdrant_scroll_memory(self.app_args)
            except Exception as exc:
                collections = []
                try:
                    collections = qdrant_collections(self.app_args)
                except Exception:
                    pass
                payload = {
                    "items": [],
                    "collection": self.app_args.collection,
                    "collections": collections,
                    "qdrant_run_counts": {},
                    "qdrant_error": str(exc),
                }
            self.send_json(payload)
        except Exception as exc:
            self.state.add_log("web", f"Failed to load memory DB: {exc}")
            self.send_json(
                {"items": [], "error": str(exc)},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )

    def handle_memory_image(self) -> None:
        parsed = urlparse(self.path)
        raw_path = parse_qs(parsed.query).get("path", [""])[0]
        try:
            image_path = safe_image_path(raw_path)
            content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
            data = image_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return
        except Exception as exc:
            self.state.add_log("web", f"Failed to serve memory image: {exc}")
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.NOT_FOUND)

    def handle_memory_search(self) -> None:
        try:
            payload = self.read_json()
            query = str(payload.get("query", "")).strip()
            if not query:
                self.send_json({"ok": False, "error": "empty query"}, HTTPStatus.BAD_REQUEST)
                return
            preview = search_memory_preview(self.app_args, query)
            self.send_json({"ok": True, **preview})
        except subprocess.TimeoutExpired:
            message = "semantic memory preview timed out"
            self.state.add_log("web", message)
            self.send_json({"ok": False, "error": message}, HTTPStatus.GATEWAY_TIMEOUT)
        except Exception as exc:
            self.state.add_log("web", f"Failed to search memory preview: {exc}")
            self.send_json(
                {"ok": False, "error": str(exc)},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def handle_navigate(self) -> None:
        try:
            payload = self.read_json()
            query = str(payload.get("query", "")).strip()
            if not query:
                self.send_json({"ok": False, "error": "empty query"}, HTTPStatus.BAD_REQUEST)
                return
            with self.state.lock:
                running = (
                    self.state.nav_process is not None
                    and self.state.nav_process.poll() is None
                )
                record_busy = (
                    (
                        self.state.record_process is not None
                        and self.state.record_process.poll() is None
                    )
                    or self.state.record_finalizing
                )
            if running:
                self.send_json(
                    {"ok": False, "error": "navigation already running"},
                    HTTPStatus.CONFLICT,
                )
                return
            if record_busy:
                self.send_json(
                    {"ok": False, "error": "database recording is active"},
                    HTTPStatus.CONFLICT,
                )
                return
            if self.app_args.require_nav_ready:
                ready, summary = self.state.navigation_request_ready()
                if not ready:
                    message = (
                        "Navigation request is not ready. Start SLAM/Nav2 "
                        "and Qdrant first. "
                        f"{summary}"
                    )
                    self.state.add_log("web", message)
                    self.send_json(
                        {"ok": False, "error": message},
                        HTTPStatus.CONFLICT,
                    )
                    return

            with self.state.lock:
                command = build_nav_command(self.app_args, query)
                env = os.environ.copy()
                env.setdefault("PYTHONUNBUFFERED", "1")
                proc = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                self.state.nav_process = proc
                self.state.nav_started_at = time.monotonic()
                self.state.nav_query = query
                self.state.nav_exit_code = None
            self.state.add_log("web", "started: " + " ".join(shlex.quote(part) for part in command))
            threading.Thread(
                target=stream_process_output,
                args=(self.state, proc, "nav"),
                daemon=True,
            ).start()
            self.send_json({"ok": True, "query": query})
        except Exception as exc:
            self.state.add_log("web", f"Failed to start navigation: {exc}")
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_stop(self) -> None:
        with self.state.lock:
            proc = self.state.nav_process
            running = proc is not None and proc.poll() is None
        if not running or proc is None:
            self.send_json({"ok": True, "stopped": False})
            return

        self.state.add_log("web", "stopping active navigation process")
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        self.send_json({"ok": True, "stopped": True})

    def handle_slam_start(self) -> None:
        try:
            already_running = False
            already_external = False
            external_log_needed = False
            with self.state.lock:
                proc = self.state.slam_process
                running = proc is not None and proc.poll() is None
                system_status = dict(self.state.system_status)
                if running:
                    already_running = True
                elif (
                    system_status.get("map_topic")
                    and system_status.get("navigate_action")
                ):
                    already_external = True
                    external_log_needed = True
                if already_running or already_external:
                    command = None
                    proc = None
                else:
                    command = build_slam_command(self.app_args)
                    env = os.environ.copy()
                    env.setdefault("PYTHONUNBUFFERED", "1")
                    proc = subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        start_new_session=True,
                    )
                    self.state.slam_process = proc
                    self.state.slam_started_at = time.monotonic()
                    self.state.slam_exit_code = None
            if already_running or already_external:
                if external_log_needed:
                    self.state.add_log(
                        "web",
                        "SLAM/Nav2 appears to be already running outside this panel",
                    )
                self.send_json(
                    {
                        "ok": True,
                        "started": False,
                        "running": True,
                        "external": already_external,
                    }
                )
                return

            assert command is not None and proc is not None
            self.state.add_log("web", "started SLAM/Nav2: " + " ".join(shlex.quote(part) for part in command))
            threading.Thread(
                target=stream_process_output,
                args=(self.state, proc, "slam"),
                daemon=True,
            ).start()
            self.send_json({"ok": True, "started": True})
        except Exception as exc:
            self.state.add_log("web", f"Failed to start SLAM/Nav2: {exc}")
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_slam_stop(self) -> None:
        with self.state.lock:
            nav_proc = self.state.nav_process
            nav_running = nav_proc is not None and nav_proc.poll() is None
            slam_proc = self.state.slam_process
            slam_running = slam_proc is not None and slam_proc.poll() is None

        if nav_running and nav_proc is not None:
            self.state.add_log("web", "stopping active navigation before SLAM/Nav2 shutdown")
            try:
                os.killpg(nav_proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

        if not slam_running or slam_proc is None:
            self.send_json({"ok": True, "stopped": False})
            return

        self.state.add_log("web", "stopping SLAM/Nav2 launch")
        try:
            os.killpg(slam_proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        self.send_json({"ok": True, "stopped": True})

    def run_short_command(self, source: str, command: list[str], timeout_sec: float = 60.0) -> tuple[bool, int, str]:
        self.state.add_log(source, "running: " + " ".join(shlex.quote(part) for part in command))
        try:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            self.state.add_log(source, output)
            self.state.add_log(source, f"command timed out after {timeout_sec}s")
            return False, 124, output
        if result.stdout:
            for line in result.stdout.splitlines():
                self.state.add_log(source, line)
        self.state.add_log(source, f"command exited with code {result.returncode}")
        return result.returncode == 0, result.returncode, result.stdout or ""

    def handle_qdrant_start(self) -> None:
        command = build_qdrant_start_command()
        ok, code, _output = self.run_short_command("qdrant", command)
        if ok:
            self.send_json({"ok": True, "started": True})
        else:
            self.send_json(
                {"ok": False, "error": f"Qdrant start failed with code {code}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def handle_qdrant_stop(self) -> None:
        command = build_qdrant_stop_command()
        ok, code, _output = self.run_short_command("qdrant", command)
        if ok:
            self.send_json({"ok": True, "stopped": True})
        else:
            self.send_json(
                {"ok": False, "error": f"Qdrant stop failed with code {code}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def handle_qdrant_collections(self) -> None:
        try:
            self.send_json(
                {
                    "ok": True,
                    "collection": self.app_args.collection,
                    "collections": qdrant_collections(self.app_args),
                }
            )
        except Exception as exc:
            self.state.add_log("qdrant", f"Failed to list collections: {exc}")
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)

    def handle_qdrant_select_collection(self) -> None:
        try:
            payload = self.read_json()
            collection = str(payload.get("collection", "")).strip()
            if not collection:
                self.send_json({"ok": False, "error": "empty collection"}, HTTPStatus.BAD_REQUEST)
                return
            collections = qdrant_collections(self.app_args)
            if collection not in collections:
                self.send_json(
                    {"ok": False, "error": f"collection not found: {collection}", "collections": collections},
                    HTTPStatus.NOT_FOUND,
                )
                return
            with self.state.lock:
                self.app_args.collection = collection
            self.state.add_log("qdrant", f"selected collection: {collection}")
            self.send_json({"ok": True, "collection": collection, "collections": collections})
        except Exception as exc:
            self.state.add_log("qdrant", f"Failed to select collection: {exc}")
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_qdrant_delete_collection(self) -> None:
        try:
            payload = self.read_json()
            collection = str(payload.get("collection", "")).strip()
            if not collection:
                self.send_json({"ok": False, "error": "empty collection"}, HTTPStatus.BAD_REQUEST)
                return
            with self.state.lock:
                record_busy = (
                    (
                        self.state.record_process is not None
                        and self.state.record_process.poll() is None
                    )
                    or self.state.record_finalizing
                )
                nav_running = (
                    self.state.nav_process is not None
                    and self.state.nav_process.poll() is None
                )
            if record_busy or nav_running:
                self.send_json(
                    {"ok": False, "error": "stop navigation/recording before deleting collections"},
                    HTTPStatus.CONFLICT,
                )
                return
            delete_qdrant_collection(self.app_args, collection)
            collections = qdrant_collections(self.app_args)
            with self.state.lock:
                if self.app_args.collection == collection:
                    self.app_args.collection = collections[0] if collections else self.app_args.initial_collection
            self.state.add_log("qdrant", f"deleted collection: {collection}")
            self.send_json(
                {
                    "ok": True,
                    "deleted": collection,
                    "collection": self.app_args.collection,
                    "collections": collections,
                }
            )
        except Exception as exc:
            self.state.add_log("qdrant", f"Failed to delete collection: {exc}")
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_record_start(self) -> None:
        try:
            ready, summary = self.state.recording_ready()
            if not ready:
                message = "Cannot record to DB until SLAM map/TF and Qdrant are ready. " + summary
                self.state.add_log("web", message)
                self.send_json({"ok": False, "error": message}, HTTPStatus.CONFLICT)
                return

            with self.state.lock:
                nav_running = (
                    self.state.nav_process is not None
                    and self.state.nav_process.poll() is None
                )
                record_busy = (
                    (
                        self.state.record_process is not None
                        and self.state.record_process.poll() is None
                    )
                    or self.state.record_finalizing
                )
                if nav_running:
                    command = None
                    proc = None
                    blocked_error = "semantic navigation already running"
                    already_running = False
                elif record_busy:
                    command = None
                    proc = None
                    blocked_error = ""
                    already_running = True
                else:
                    blocked_error = ""
                    already_running = False
                    run_dir = RECORDINGS_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
                    keyframes_dir = run_dir / "keyframes"
                    keyframes_dir.mkdir(parents=True, exist_ok=False)
                    target_collection = f"{self.app_args.record_collection_prefix}{run_dir.name}"
                    command = build_record_command(self.app_args, keyframes_dir)
                    env = os.environ.copy()
                    env.setdefault("PYTHONUNBUFFERED", "1")
                    proc = subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        start_new_session=True,
                    )
                    self.state.record_process = proc
                    self.state.record_started_at = time.monotonic()
                    self.state.record_exit_code = None
                    self.state.record_output_dir = str(run_dir)
                    self.state.record_target_collection = target_collection
                    self.state.record_last_result = None
            if blocked_error:
                self.send_json(
                    {"ok": False, "error": blocked_error},
                    HTTPStatus.CONFLICT,
                )
                return
            if already_running:
                self.send_json({"ok": True, "started": False, "running": True})
                return
            if command is None or proc is None:
                self.send_json(
                    {"ok": False, "error": "recording start did not create a process"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self.state.add_log(
                "web",
                f"started DB recording for {target_collection}: "
                + " ".join(shlex.quote(part) for part in command),
            )
            threading.Thread(
                target=stream_process_output,
                args=(self.state, proc, "record"),
                daemon=True,
            ).start()
            self.send_json(
                {
                    "ok": True,
                    "started": True,
                    "output_dir": str(run_dir),
                    "collection": target_collection,
                }
            )
        except Exception as exc:
            self.state.add_log("web", f"Failed to start DB recording: {exc}")
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_record_stop(self) -> None:
        with self.state.lock:
            proc = self.state.record_process
            running = proc is not None and proc.poll() is None
            output_dir = self.state.record_output_dir
            target_collection = self.state.record_target_collection
        if not running or proc is None:
            self.send_json({"ok": True, "stopped": False})
            return

        self.state.add_log("web", "stopping DB recording")
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        if output_dir:
            threading.Thread(
                target=finalize_recording,
                args=(self.state, self.app_args, proc, Path(output_dir), target_collection),
                daemon=True,
            ).start()
        self.send_json({"ok": True, "stopped": True, "loading": bool(output_dir)})

    def send_camera_snapshot(self) -> None:
        with self.state.lock:
            jpeg = self.state.latest_jpeg
        self.send_bytes(jpeg or PLACEHOLDER_JPEG, "image/jpeg")

    def stream_camera(self) -> None:
        boundary = "frame"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        last_count = -1
        try:
            while True:
                with self.state.frame_condition:
                    self.state.frame_condition.wait_for(
                        lambda: self.state.frame_count != last_count,
                        timeout=2.0,
                    )
                    jpeg = self.state.latest_jpeg
                    last_count = self.state.frame_count
                if jpeg is None:
                    jpeg = PLACEHOLDER_JPEG
                self.wfile.write(f"--{boundary}\r\n".encode("ascii"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return

    def stream_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last_id = 0
        try:
            last_id = int(self.headers.get("Last-Event-ID", "0") or "0")
        except ValueError:
            last_id = 0
        if last_id <= 0:
            with self.state.lock:
                newest_id = self.state.next_log_id - 1
            last_id = max(0, newest_id - EVENT_STREAM_INITIAL_BACKLOG)
        try:
            while True:
                with self.state.lock:
                    items = [item for item in self.state.logs if item["id"] > last_id]
                for item in items:
                    last_id = item["id"]
                    payload = json.dumps(item, ensure_ascii=False)
                    self.wfile.write(f"id: {last_id}\n".encode("utf-8"))
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.4)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return

    def log_message(self, fmt: str, *args) -> None:
        message = fmt % args
        noisy_paths = (
            "GET /api/status ",
            "GET /api/logs",
            "GET /api/events ",
            "GET /camera.jpg",
            "GET /camera.mjpg ",
        )
        if any(path in message for path in noisy_paths):
            return
        self.state.add_log("http", message)


class ControlServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, address, handler, state: SharedState, args: argparse.Namespace):
        super().__init__(address, handler)
        self.state = state
        self.app_args = args


def make_placeholder_jpeg() -> bytes:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :] = (28, 31, 36)
    cv2.putText(
        image,
        "Waiting for robot camera",
        (125, 178),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (210, 220, 225),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return encoded.tobytes() if ok else b""


PLACEHOLDER_JPEG = make_placeholder_jpeg()


INDEX_HTML = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Semantic Nav Control</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #121416;
      --panel: #1b1f24;
      --panel-2: #222831;
      --text: #f0f3f5;
      --muted: #aab3bd;
      --line: #343b44;
      --accent: #2fb6a3;
      --danger: #e05f55;
      --warn: #d8a441;
    }
    * { box-sizing: border-box; }
    html { min-height: 100%; overflow: auto; }
    body {
      margin: 0;
      min-height: 100%;
      overflow: auto;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    main {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 0;
      overflow: visible;
      align-items: start;
    }
    .visual-column {
      min-width: 0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      border-right: 1px solid var(--line);
    }
    .camera {
      min-width: 0;
      height: clamp(420px, 68vh, 820px);
      display: flex;
      flex-direction: column;
      border-bottom: 1px solid var(--line);
    }
    .topbar {
      min-height: 44px;
      flex: 0 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: #171a1e;
    }
    .title {
      font-weight: 650;
      font-size: 14px;
      line-height: 1.2;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--danger);
      box-shadow: 0 0 0 3px rgba(224, 95, 85, 0.15);
    }
    .dot.ok {
      background: var(--accent);
      box-shadow: 0 0 0 3px rgba(47, 182, 163, 0.17);
    }
    .stream {
      flex: 1;
      min-height: 0;
      display: grid;
      place-items: center;
      background: #0d0f11;
      overflow: hidden;
    }
    .stream img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
    aside {
      min-width: 0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      background: var(--panel);
    }
    .control-panel {
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    .control {
      flex: 0 0 auto;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }
    .query-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
    }
    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }
    input, select {
      width: 100%;
      min-width: 0;
      height: 36px;
      border: 1px solid var(--line);
      background: #101316;
      color: var(--text);
      border-radius: 6px;
      padding: 0 12px;
      font-size: 13px;
      outline: none;
    }
    input:focus, select:focus { border-color: var(--accent); }
    button {
      height: 36px;
      border: 0;
      border-radius: 6px;
      padding: 0 14px;
      background: var(--accent);
      color: #061412;
      font-weight: 700;
      font-size: 13px;
      cursor: pointer;
    }
    button.secondary {
      width: 100%;
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--line);
    }
    button.danger {
      color: #fff;
      background: var(--danger);
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .selected-match {
      margin-top: 12px;
      display: none;
      grid-template-columns: 88px minmax(0, 1fr);
      gap: 10px;
      align-items: start;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
    }
    .selected-match.visible { display: grid; }
    .selected-match img {
      width: 88px;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      border-radius: 4px;
      background: #0d0f11;
    }
    .selected-match strong {
      display: block;
      font-size: 13px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .selected-match span {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.3;
      overflow-wrap: anywhere;
    }
    .metrics {
      flex: 0 0 auto;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }
    .metric {
      min-height: 50px;
      padding: 8px;
      border-radius: 6px;
      background: var(--panel-2);
      border: 1px solid var(--line);
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }
    .metric strong {
      display: block;
      font-size: 13px;
      font-weight: 650;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .logs {
      flex: 0 0 42vh;
      min-height: 300px;
      max-height: 520px;
      display: flex;
      flex-direction: column;
      border-bottom: 1px solid var(--line);
    }
    .logs h2 {
      margin: 0;
      padding: 13px 18px;
      font-size: 14px;
      border-bottom: 1px solid var(--line);
      font-weight: 650;
    }
    pre {
      min-height: 0;
      margin: 0;
      padding: 12px 14px;
      overflow: auto;
      flex: 1;
      font-size: 12px;
      line-height: 1.45;
      color: #d8dee5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #101316;
    }
    .log-footer {
      flex: 0 0 auto;
      padding: 8px 14px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.2;
      background: #171a1e;
    }
    .side-fill {
      flex: 1;
      min-height: 180px;
      padding: 12px;
      display: grid;
      align-content: start;
      gap: 10px;
      background: #171a1e;
    }
    .side-fill h2 {
      margin: 0;
      font-size: 14px;
      font-weight: 650;
    }
    .side-row {
      display: grid;
      grid-template-columns: 88px minmax(0, 1fr);
      gap: 8px;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
    }
    .side-row span {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.25;
    }
    .side-row strong {
      min-width: 0;
      font-size: 12px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    body::-webkit-scrollbar,
    pre::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }
    body::-webkit-scrollbar-thumb,
    pre::-webkit-scrollbar-thumb {
      background: #4a535f;
      border-radius: 999px;
      border: 2px solid #101316;
    }
    body::-webkit-scrollbar-track,
    pre::-webkit-scrollbar-track {
      background: #101316;
    }
    .memory-section {
      display: flex;
      flex-direction: column;
      min-height: 520px;
      background: var(--panel);
    }
    .memory-toolbar {
      flex: 0 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    .memory-actions {
      display: flex;
      gap: 6px;
      align-items: center;
    }
    .memory-actions button {
      min-height: 32px;
      padding: 0 10px;
      font-size: 12px;
    }
    .memory-run {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 6px;
    }
    .memory-run span {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .memory-detail {
      flex: 0 0 auto;
      display: none;
      grid-template-columns: minmax(260px, 420px) minmax(0, 1fr);
      gap: 10px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: #171a1e;
    }
    .memory-detail.visible { display: grid; }
    .memory-detail img {
      width: 100%;
      aspect-ratio: 4 / 3;
      max-height: 320px;
      object-fit: contain;
      border-radius: 4px;
      background: #0d0f11;
    }
    .memory-detail h2 {
      margin: 0 0 6px;
      font-size: 14px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .memory-detail p {
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .memory-grid {
      min-height: 0;
      overflow: visible;
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      align-content: start;
      gap: 12px;
      background: #101316;
    }
    .memory-card {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(96px, auto);
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      overflow: hidden;
      cursor: pointer;
    }
    .memory-card.selected { border-color: var(--accent); }
    .memory-card img {
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      display: block;
      background: #0d0f11;
    }
    .memory-card div { padding: 8px; }
    .memory-card strong {
      display: block;
      font-size: 12px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .memory-card span {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.3;
      overflow-wrap: anywhere;
    }
    .src-web { color: var(--accent); }
    .src-nav { color: #8ab4ff; }
    .src-slam { color: #d8a441; }
    .src-qdrant { color: #c49bff; }
    .src-record { color: #70d36b; }
    .src-camera { color: var(--warn); }
    .src-http { color: var(--muted); }
    @media (max-width: 900px) {
      main {
        min-height: 100vh;
        grid-template-columns: 1fr;
        overflow: visible;
      }
      .visual-column { min-height: 0; border-right: 0; }
      .camera { height: min(58vh, 620px); }
      aside { min-height: 0; }
      .control-panel { min-height: 0; }
      .logs {
        flex-basis: 360px;
        max-height: none;
      }
      .memory-detail { grid-template-columns: 1fr; }
      .memory-grid {
        grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="visual-column">
      <section class="camera">
        <div class="topbar">
          <div class="title">Semantic Nav Control</div>
          <div class="status"><span id="cameraDot" class="dot"></span><span id="cameraStatus">camera offline</span></div>
        </div>
        <div class="stream"><img id="cameraImage" src="/camera.jpg" alt="robot camera"></div>
      </section>
      <section id="panelMemory" class="memory-section">
        <div class="memory-toolbar">
          <strong id="memoryCount">Memory DB</strong>
          <div class="memory-actions">
            <button id="refreshMemory" class="secondary" type="button">Refresh</button>
            <button id="deleteCollection" class="secondary danger" type="button">Delete</button>
          </div>
          <div class="memory-run">
            <select id="collectionSelect"></select>
            <span id="collectionInfo">-</span>
          </div>
        </div>
        <div id="memoryDetail" class="memory-detail">
          <img id="memoryDetailImage" alt="memory frame">
          <div>
            <h2 id="memoryDetailTitle">No image selected</h2>
            <p id="memoryDetailMeta">-</p>
          </div>
        </div>
        <div id="memoryGrid" class="memory-grid"></div>
      </section>
    </section>
    <aside>
      <section id="panelControl" class="control-panel">
        <section class="control">
          <label for="query">Semantic command</label>
          <div class="query-row">
            <input id="query" type="text" value="дорога" autocomplete="off">
            <button id="go">Go</button>
          </div>
          <div id="selectedMatch" class="selected-match">
            <img id="selectedMatchImage" alt="selected memory frame">
            <div>
              <strong id="selectedMatchTitle">No selected image</strong>
              <span id="selectedMatchMeta">-</span>
            </div>
          </div>
          <div class="button-row">
            <button id="startQdrant" class="secondary">Start Qdrant</button>
            <button id="stopQdrant" class="secondary danger">Stop Qdrant</button>
          </div>
          <div class="button-row">
            <button id="startSlam" class="secondary">Start SLAM/Nav2</button>
            <button id="stopSlam" class="secondary danger">Stop SLAM/Nav2</button>
          </div>
          <div class="button-row">
            <button id="startRecord" class="secondary">Start DB recording</button>
            <button id="stopRecord" class="secondary danger">Stop DB recording</button>
          </div>
          <button id="stop" class="secondary danger">Stop active goal</button>
        </section>
        <section class="metrics">
          <div class="metric"><span>Navigation</span><strong id="navState">idle</strong></div>
          <div class="metric"><span>Runtime</span><strong id="runtime">-</strong></div>
          <div class="metric"><span>Qdrant</span><strong id="qdrantState">checking</strong></div>
          <div class="metric"><span>Qdrant URL</span><strong id="qdrantUrl">-</strong></div>
          <div class="metric"><span>Recording</span><strong id="recordState">idle</strong></div>
          <div class="metric"><span>Record runtime</span><strong id="recordRuntime">-</strong></div>
          <div class="metric"><span>SLAM process</span><strong id="slamProcess">idle</strong></div>
          <div class="metric"><span>SLAM runtime</span><strong id="slamRuntime">-</strong></div>
          <div class="metric"><span>SLAM/Nav2</span><strong id="systemState">checking</strong></div>
          <div class="metric"><span>TF map</span><strong id="tfState">-</strong></div>
          <div class="metric"><span>Frames</span><strong id="frames">0</strong></div>
          <div class="metric"><span>Image frame</span><strong id="imageFrame">-</strong></div>
        </section>
        <section class="logs">
          <h2>Execution Log</h2>
          <pre id="log"></pre>
          <div class="log-footer">Log panel</div>
        </section>
        <section class="side-fill">
          <h2>Workspace</h2>
          <div class="side-row"><span>Camera</span><strong id="sideCamera">-</strong></div>
          <div class="side-row"><span>Memory</span><strong id="sideMemory">-</strong></div>
          <div class="side-row"><span>Qdrant</span><strong id="sideQdrant">-</strong></div>
        </section>
      </section>
    </aside>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const logEl = $("log");
    let lastLogId = 0;
    const maxLogNodes = 260;
    let statusInFlight = false;
    let logPollInFlight = false;
    let collectionSelectInFlight = false;
    let activeCollectionMissing = false;

    function addLog(item) {
      if (item.id <= lastLogId) return;
      lastLogId = item.id;
      const cls = `src-${item.source}`;
      const line = `[${item.time}] ${item.source}: ${item.message}`;
      const span = document.createElement("span");
      span.className = cls;
      span.textContent = line + "\n";
      logEl.appendChild(span);
      while (logEl.childNodes.length > maxLogNodes) {
        logEl.removeChild(logEl.firstChild);
      }
      logEl.scrollTop = logEl.scrollHeight;
    }

    function imageUrl(item) {
      return item && item.image_path ? `/api/memory/image?path=${encodeURIComponent(item.image_path)}` : "";
    }

    function fmt(value, digits = 2) {
      const number = Number(value);
      return Number.isFinite(number) ? number.toFixed(digits) : "-";
    }

    function itemMeta(item) {
      const pos = item.position || {};
      const score = item.score === null || item.score === undefined ? "-" : fmt(item.score, 4);
      return [
        `score=${score}`,
        `run=${item.run_id || "-"}`,
        `frame=${item.pose_frame || "-"}`,
        `x=${fmt(pos.x)} y=${fmt(pos.y)} z=${fmt(pos.z)}`,
        `image=${item.image_frame || "-"}`
      ].join(" · ");
    }

    function showSelectedMatch(item) {
      const box = $("selectedMatch");
      if (!item) {
        box.classList.remove("visible");
        return;
      }
      $("selectedMatchImage").src = imageUrl(item);
      $("selectedMatchTitle").textContent = item.memory_id || "selected memory";
      $("selectedMatchMeta").textContent = itemMeta(item);
      box.classList.add("visible");
      showMemoryDetail(item);
    }

    function showMemoryDetail(item) {
      const detail = $("memoryDetail");
      if (!item) {
        detail.classList.remove("visible");
        return;
      }
      $("memoryDetailImage").src = imageUrl(item);
      $("memoryDetailTitle").textContent = item.memory_id || "memory item";
      const pos = item.position || {};
      $("memoryDetailMeta").textContent = [
        itemMeta(item),
        `timestamp=${item.timestamp ?? "-"}`,
        `topic=${item.image_topic || "-"}`,
        `source_frame=${item.source_pose_frame || "-"}`,
        `map=${item.map_yaml || "-"}`,
        `path=${item.image_path || "-"}`
      ].join("\n");
      detail.classList.add("visible");
      document.querySelectorAll(".memory-card").forEach((card) => {
        card.classList.toggle("selected", card.dataset.memoryId === item.memory_id);
      });
    }

    function qdrantInfo(runCounts) {
      const entries = Object.entries(runCounts || {});
      if (!entries.length) return "Qdrant source: empty/unknown";
      return `Qdrant source: ${entries.map(([run, count]) => `${run} (${count})`).join(", ")}`;
    }

    function populateCollectionSelector(collections, selectedCollection) {
      const select = $("collectionSelect");
      const previous = select.value;
      select.textContent = "";
      const names = collections && collections.length ? collections : [selectedCollection].filter(Boolean);
      for (const name of names) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
      }
      if (selectedCollection && [...select.options].some((option) => option.value === selectedCollection)) {
        select.value = selectedCollection;
      } else if (previous && [...select.options].some((option) => option.value === previous)) {
        select.value = previous;
      }
      $("deleteCollection").disabled = !select.value;
    }

    function renderMemory(data) {
      const items = data.items || [];
      const grid = $("memoryGrid");
      grid.textContent = "";
      populateCollectionSelector(data.collections || [], data.collection);
      activeCollectionMissing = Boolean(data.collection_missing);
      $("memoryCount").textContent = `Memory DB (${items.length})`;
      const selectedLabel = data.collection || "-";
      const errorLabel = data.qdrant_error ? ` · ${data.qdrant_error}` : "";
      const missingLabel = activeCollectionMissing ? " · collection missing" : "";
      $("collectionInfo").textContent = `${selectedLabel} · ${qdrantInfo(data.qdrant_run_counts)}${missingLabel}${errorLabel}`;
      $("sideMemory").textContent = `${selectedLabel} · ${items.length} images`;
      if (!items.length) {
        const empty = document.createElement("div");
        empty.className = "memory-card";
        empty.innerHTML = activeCollectionMissing
          ? "<div><strong>Collection missing</strong><span>Use Start DB recording, then Stop DB recording to create and load it</span></div>"
          : "<div><strong>No memory records</strong><span>The selected collection has no points</span></div>";
        grid.appendChild(empty);
        showMemoryDetail(null);
        return;
      }
      for (const item of items) {
        const card = document.createElement("article");
        card.className = "memory-card";
        card.dataset.memoryId = item.memory_id || "";
        const img = document.createElement("img");
        img.src = imageUrl(item);
        img.alt = item.memory_id || "memory frame";
        const body = document.createElement("div");
        const title = document.createElement("strong");
        title.textContent = item.memory_id || "memory item";
        const meta = document.createElement("span");
        meta.textContent = itemMeta(item);
        body.append(title, meta);
        card.append(img, body);
        card.addEventListener("click", () => showMemoryDetail(item));
        grid.appendChild(card);
      }
      showMemoryDetail(items[0]);
    }

    async function loadMemory() {
      try {
        const res = await fetch("/api/memory", { cache: "no-store" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Memory request failed");
        renderMemory(data);
      } catch (err) {
        $("memoryCount").textContent = "Memory DB unavailable";
        $("collectionInfo").textContent = String(err);
        $("memoryGrid").textContent = "";
        showMemoryDetail(null);
      }
    }

    async function selectCollection() {
      if (collectionSelectInFlight) return;
      const collection = $("collectionSelect").value;
      if (!collection) return;
      collectionSelectInFlight = true;
      try {
        const res = await fetch("/api/qdrant/collections/select", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ collection })
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || "Collection select failed");
        await loadMemory();
        refreshStatus();
      } catch (err) {
        $("collectionInfo").textContent = String(err);
      } finally {
        collectionSelectInFlight = false;
      }
    }

    async function previewSemanticMatch(query) {
      try {
        const res = await fetch("/api/memory/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query })
        });
        const data = await res.json();
        if (res.ok && data.selected) showSelectedMatch(data.selected);
      } catch (err) {
        console.warn("Semantic preview failed", err);
      }
    }

    async function refreshStatus() {
      if (statusInFlight) return;
      statusInFlight = true;
      try {
        const res = await fetch("/api/status", { cache: "no-store" });
        const data = await res.json();
        const camera = data.camera;
        const nav = data.navigation;
        const slam = data.slam;
        const qdrant = data.qdrant;
        const recording = data.recording;
        const system = data.system;
        const fresh = camera.has_frame && camera.frame_age_sec !== null && camera.frame_age_sec < 2.5;
        $("cameraDot").classList.toggle("ok", fresh);
        $("cameraStatus").textContent = fresh ? `camera live (${camera.frame_age_sec}s)` : "camera offline";
        $("navState").textContent = nav.running ? `running: ${nav.query}` : `idle (${nav.last_exit_code ?? "-"})`;
        $("runtime").textContent = nav.runtime_sec === null ? "-" : `${nav.runtime_sec}s`;
        $("qdrantState").textContent = qdrant.ready ? "ready" : qdrant.summary;
        $("qdrantUrl").textContent = qdrant.url || "-";
        $("sideCamera").textContent = fresh ? `${camera.frame_age_sec}s · ${camera.image_frame || "-"}` : "offline";
        $("sideQdrant").textContent = qdrant.ready ? qdrant.url || "ready" : qdrant.summary;
        $("recordState").textContent = recording.finalizing
          ? "loading to DB"
          : (recording.running ? "recording" : `idle (${recording.last_exit_code ?? "-"})`);
        $("recordRuntime").textContent = recording.runtime_sec === null ? "-" : `${recording.runtime_sec}s`;
        $("slamProcess").textContent = slam.running ? "running" : `idle (${slam.last_exit_code ?? "-"})`;
        $("slamRuntime").textContent = slam.runtime_sec === null ? "-" : `${slam.runtime_sec}s`;
        $("systemState").textContent = system.ready ? "ready" : system.summary;
        $("tfState").textContent = system.goal_frame
          ? (system.clock_topic ? "map -> base_link" : "no /clock")
          : "missing";
        $("frames").textContent = String(camera.frame_count);
        $("imageFrame").textContent = camera.image_frame || "-";
        const navRequestReady = Boolean(
          system.clock_topic && system.map_topic && system.navigate_action && qdrant.ready
        );
        const recordBusy = Boolean(recording.running || recording.finalizing);
        $("go").disabled = nav.running || recordBusy || !navRequestReady;
        $("stop").disabled = !nav.running;
        $("startQdrant").disabled = qdrant.ready || recordBusy;
        $("stopQdrant").disabled = !qdrant.ready || recordBusy;
        $("startSlam").disabled = slam.running || recordBusy;
        $("stopSlam").disabled = !slam.running || recordBusy;
        $("startRecord").disabled = nav.running || recordBusy || !qdrant.ready || !system.clock_topic || !system.map_topic || !system.goal_frame;
        $("stopRecord").disabled = !recording.running;
        $("deleteCollection").disabled = recordBusy || nav.running || activeCollectionMissing || !$("collectionSelect").value || !qdrant.ready;
      } catch (err) {
        $("cameraDot").classList.remove("ok");
        $("cameraStatus").textContent = "panel disconnected";
      } finally {
        statusInFlight = false;
      }
    }

    async function refreshCameraImage() {
      $("cameraImage").src = `/camera.jpg?ts=${Date.now()}`;
    }

    async function pollLogs() {
      if (logPollInFlight) return;
      logPollInFlight = true;
      try {
        const res = await fetch(`/api/logs?last_id=${lastLogId}`, { cache: "no-store" });
        const data = await res.json();
        for (const item of data.logs || []) addLog(item);
      } catch (err) {
        console.warn("Log polling failed", err);
      } finally {
        logPollInFlight = false;
      }
    }

    async function navigate() {
      const query = $("query").value.trim();
      if (!query) return;
      previewSemanticMatch(query);
      const res = await fetch("/api/navigate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        console.warn(data.error || "Navigation request failed");
      }
      refreshStatus();
    }

    async function stopNav() {
      await fetch("/api/stop", { method: "POST" });
      refreshStatus();
    }

    async function startSlam() {
      await fetch("/api/slam/start", { method: "POST" });
      refreshStatus();
    }

    async function stopSlam() {
      await fetch("/api/slam/stop", { method: "POST" });
      refreshStatus();
    }

    async function startQdrant() {
      await fetch("/api/qdrant/start", { method: "POST" });
      refreshStatus();
    }

    async function stopQdrant() {
      await fetch("/api/qdrant/stop", { method: "POST" });
      refreshStatus();
    }

    async function startRecord() {
      await fetch("/api/record/start", { method: "POST" });
      refreshStatus();
    }

    async function stopRecord() {
      await fetch("/api/record/stop", { method: "POST" });
      refreshStatus();
    }

    async function deleteCollection() {
      const collection = $("collectionSelect").value;
      if (!collection) return;
      if (!window.confirm(`Delete Qdrant collection "${collection}"?`)) return;
      const res = await fetch("/api/qdrant/collections/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ collection })
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        console.warn(data.error || "Collection delete failed");
      }
      await loadMemory();
      refreshStatus();
    }

    $("go").addEventListener("click", navigate);
    $("stop").addEventListener("click", stopNav);
    $("startQdrant").addEventListener("click", startQdrant);
    $("stopQdrant").addEventListener("click", stopQdrant);
    $("startSlam").addEventListener("click", startSlam);
    $("stopSlam").addEventListener("click", stopSlam);
    $("startRecord").addEventListener("click", startRecord);
    $("stopRecord").addEventListener("click", stopRecord);
    $("collectionSelect").addEventListener("change", selectCollection);
    $("deleteCollection").addEventListener("click", deleteCollection);
    $("refreshMemory").addEventListener("click", loadMemory);
    $("query").addEventListener("keydown", (event) => {
      if (event.key === "Enter") navigate();
    });

    setInterval(refreshStatus, 800);
    setInterval(refreshCameraImage, 450);
    setInterval(pollLogs, 700);
    refreshStatus();
    pollLogs();
    loadMemory();
  </script>
</body>
</html>
"""


def spin_ros(node: Node) -> None:
    try:
        rclpy.spin(node)
    except Exception as exc:
        print(f"ROS spin stopped: {exc}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web control panel for semantic navigation.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--map-topic", default="/map")
    parser.add_argument("--global-costmap-topic", default="/global_costmap/costmap")
    parser.add_argument("--action-name", default="/navigate_to_pose")
    parser.add_argument("--robot-base-frame", default="base_link")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="semantic_visual_memory")
    parser.add_argument("--qdrant-timeout-sec", type=float, default=2.0)
    parser.add_argument("--memory-limit", type=int, default=200)
    parser.add_argument("--memory-search-timeout-sec", type=float, default=45.0)
    parser.add_argument("--memory-load-timeout-sec", type=float, default=600.0)
    parser.add_argument("--stream-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--max-stream-fps", type=float, default=3.0)
    parser.add_argument(
        "--image-reliability",
        choices=("reliable", "best_effort"),
        default="reliable",
        help="QoS reliability for the camera image subscription.",
    )
    parser.add_argument("--client-timeout-sec", type=float, default=6.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--goal-frame", default="map")
    parser.add_argument("--mission-match-threshold", type=float, default=0.30)
    parser.add_argument("--mission-check-period-sec", type=float, default=1.5)
    parser.add_argument(
        "--max-goal-distance",
        type=float,
        default=0.0,
        help="0 disables semantic goal distance limiting.",
    )
    parser.add_argument(
        "--result-timeout-sec",
        type=float,
        default=0.0,
        help="0 disables client-side Nav2 result timeout.",
    )
    parser.add_argument("--feedback-log-period-sec", type=float, default=4.0)
    parser.add_argument("--slam-log-level", default="info")
    parser.add_argument("--record-odom-topic", default="/chassis/odom")
    parser.add_argument("--record-min-time-delta-sec", type=float, default=1.0)
    parser.add_argument("--record-min-translation-delta-m", type=float, default=0.25)
    parser.add_argument("--record-min-rotation-delta-rad", type=float, default=0.25)
    parser.add_argument("--record-max-pose-age-sec", type=float, default=0.25)
    parser.add_argument("--record-tf-timeout-sec", type=float, default=2.0)
    parser.add_argument(
        "--record-collection-prefix",
        default="semantic_visual_memory_",
        help="Prefix for Qdrant collections created from new DB recordings.",
    )
    parser.add_argument(
        "--record-recreate-collection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recreate the target Qdrant collection when a recording is loaded.",
    )
    parser.add_argument(
        "--require-nav-ready",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject web navigation requests until /map, /navigate_to_pose, and map->base_link are available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.initial_collection = args.collection
    state = SharedState()
    state.app_args = args
    rclpy.init()
    node = CameraSubscriber(state, args)
    ros_thread = threading.Thread(target=spin_ros, args=(node,), daemon=True)
    ros_thread.start()

    server = ControlServer((args.host, args.port), ControlHandler, state, args)
    state.add_log("web", f"panel listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        with state.lock:
            proc = state.nav_process
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        with state.lock:
            slam_proc = state.slam_process
        if slam_proc is not None and slam_proc.poll() is None:
            try:
                os.killpg(slam_proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        with state.lock:
            record_proc = state.record_process
        if record_proc is not None and record_proc.poll() is None:
            try:
                os.killpg(record_proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
