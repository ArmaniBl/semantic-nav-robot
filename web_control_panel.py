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
from urllib.parse import parse_qs, urlparse
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
EXPLORATION_RUNS_DIR = PROJECT_ROOT / "data" / "exploration_runs"


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
        self.logs: deque[dict] = deque(maxlen=1000)
        self.next_log_id = 1
        self.nav_process: subprocess.Popen | None = None
        self.nav_started_at = 0.0
        self.nav_query = ""
        self.nav_exit_code: int | None = None
        self.slam_process: subprocess.Popen | None = None
        self.slam_started_at = 0.0
        self.slam_exit_code: int | None = None
        self.explore_process: subprocess.Popen | None = None
        self.explore_started_at = 0.0
        self.explore_exit_code: int | None = None
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
        with self.lock:
            item = {
                "id": self.next_log_id,
                "time": time.strftime("%H:%M:%S"),
                "source": source,
                "message": line,
            }
            self.next_log_id += 1
            self.logs.append(item)

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
            slam_running = slam_proc is not None and slam_proc.poll() is None
            slam_runtime = None
            if slam_running:
                slam_runtime = round(time.monotonic() - self.slam_started_at, 1)
            explore_proc = self.explore_process
            explore_running = explore_proc is not None and explore_proc.poll() is None
            explore_runtime = None
            if explore_running:
                explore_runtime = round(time.monotonic() - self.explore_started_at, 1)
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
                    "runtime_sec": slam_runtime,
                    "last_exit_code": self.slam_exit_code,
                },
                "explore": {
                    "running": explore_running,
                    "runtime_sec": explore_runtime,
                    "last_exit_code": self.explore_exit_code,
                },
                "qdrant": dict(self.qdrant_status),
                "system": dict(self.system_status),
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
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Image, args.image_topic, self.on_image, qos)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.create_timer(1.0, self.update_system_status)
        self.create_timer(2.0, self.update_qdrant_status)
        self.get_logger().info(f"Streaming camera topic: {args.image_topic}")

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
        topic_names = {name for name, _types in self.get_topic_names_and_types()}
        map_topic = self.args.map_topic in topic_names
        action_prefix = self.args.action_name.rstrip("/")
        navigate_action = f"{action_prefix}/_action/status" in topic_names
        goal_frame = bool(self.tf_buffer.can_transform(
            self.args.goal_frame,
            self.args.robot_base_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=0.0),
        ))

        missing = []
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
    return [
        str(python_path),
        str(script),
        query,
        "--collection",
        args.collection,
        "--top-k",
        str(args.top_k),
        "--goal-frame",
        args.goal_frame,
        "--action-name",
        args.action_name,
        "--robot-base-frame",
        args.robot_base_frame,
        "--max-goal-distance",
        str(args.max_goal_distance),
        "--result-timeout-sec",
        str(args.result_timeout_sec),
        "--feedback-log-period-sec",
        str(args.feedback_log_period_sec),
    ]


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


def build_explore_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "exploration_mission.py"),
        "--initial-tf-timeout-sec",
        str(args.explore_initial_tf_timeout_sec),
        "--max-goals",
        str(args.explore_max_goals),
        "--max-goal-distance",
        str(args.explore_max_goal_distance),
        "--goal-timeout-sec",
        str(args.explore_goal_timeout_sec),
        "--return-timeout-sec",
        str(args.explore_return_timeout_sec),
        "--feedback-log-period-sec",
        str(args.feedback_log_period_sec),
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
        "run_id": run_id_from_path(payload.get("image_path") or ""),
        "timestamp": payload.get("timestamp"),
        "image_topic": payload.get("image_topic"),
        "image_frame": payload.get("image_frame"),
        "pose_topic": payload.get("pose_topic"),
        "pose_frame": payload.get("pose_frame"),
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
    }


def run_id_from_path(raw_path: str) -> str | None:
    if not raw_path:
        return None
    parts = Path(raw_path).parts
    for index, part in enumerate(parts[:-1]):
        if part == "exploration_runs" and index + 1 < len(parts):
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


def list_exploration_runs() -> list[dict]:
    runs = []
    if not EXPLORATION_RUNS_DIR.exists():
        return runs
    for run_dir in sorted(EXPLORATION_RUNS_DIR.glob("run_*")):
        if not run_dir.is_dir():
            continue
        keyframes_dir = run_dir / "keyframes"
        metadata_path = keyframes_dir / "metadata.jsonl"
        images_dir = keyframes_dir / "images"
        image_count = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0
        embeddings_path = keyframes_dir / "embeddings.jsonl"
        runs.append(
            {
                "id": run_dir.name,
                "path": str(run_dir),
                "metadata": str(metadata_path),
                "has_metadata": metadata_path.exists(),
                "has_embeddings": embeddings_path.exists(),
                "image_count": image_count,
                "mtime": run_dir.stat().st_mtime,
            }
        )
    runs.sort(key=lambda item: (item["mtime"], item["id"]), reverse=True)
    return runs


def find_exploration_run(run_id: str) -> dict:
    runs = list_exploration_runs()
    if run_id in ("", "latest"):
        if not runs:
            raise FileNotFoundError("No exploration runs found")
        return runs[0]
    for run in runs:
        if run["id"] == run_id:
            return run
    raise FileNotFoundError(f"Exploration run not found: {run_id}")


def metadata_record_to_memory_item(record: dict, run_id: str) -> dict:
    pose = record.get("pose") or {}
    position = pose.get("position") or {}
    orientation = pose.get("orientation") or {}
    return {
        "id": record.get("memory_id") or "",
        "score": None,
        "memory_id": record.get("memory_id") or "",
        "image_path": record.get("image_path") or "",
        "run_id": run_id,
        "timestamp": record.get("timestamp"),
        "image_topic": record.get("image_topic"),
        "image_frame": record.get("image_frame"),
        "pose_topic": record.get("pose_topic"),
        "pose_frame": record.get("pose_frame"),
        "child_frame_id": record.get("child_frame_id"),
        "pose_age_sec": record.get("pose_age_sec"),
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
        "status": record.get("status", "active"),
        "embedding_model": None,
    }


def load_run_memory(run_id: str) -> dict:
    runs = list_exploration_runs()
    try:
        run = find_exploration_run(run_id)
    except FileNotFoundError:
        return {"items": [], "runs": runs, "selected_run": None}
    metadata_path = Path(run["metadata"])
    records = read_jsonl(metadata_path) if metadata_path.exists() else []
    return {
        "items": [metadata_record_to_memory_item(record, run["id"]) for record in records],
        "runs": runs,
        "selected_run": run["id"],
        "selected_run_info": run,
    }


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


def load_run_into_qdrant(args: argparse.Namespace, run_id: str) -> dict:
    run = find_exploration_run(run_id)
    metadata_path = Path(run["metadata"])
    if not metadata_path.exists():
        raise FileNotFoundError(f"Run has no metadata: {metadata_path}")
    embeddings_path = metadata_path.parent / "embeddings.jsonl"
    python_path = PROJECT_ROOT / ".ruclip_venv" / "bin" / "python"

    embed_output = ""
    if not embeddings_path.exists():
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

    load_output = run_command_capture(
        [
            str(python_path),
            str(PROJECT_ROOT / "qdrant_load_keyframes.py"),
            "--embeddings",
            str(embeddings_path),
            "--url",
            args.qdrant_url,
            "--collection",
            args.collection,
            "--recreate",
        ],
        args.memory_load_timeout_sec,
    )
    return {
        "run": run["id"],
        "metadata": str(metadata_path),
        "embeddings": str(embeddings_path),
        "embedded": bool(embed_output),
        "embed_output": embed_output,
        "load_output": load_output,
    }


def qdrant_scroll_memory(args: argparse.Namespace) -> dict:
    limit = max(1, min(args.memory_limit, 512))
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
        elif kind == "explore" and state.explore_process is proc:
            state.explore_exit_code = exit_code
            state.explore_process = None
            suffix = ""
        else:
            suffix = ""
    state.add_log(kind, f"process exited with code {exit_code}{suffix}")


class ControlHandler(BaseHTTPRequestHandler):
    server_version = "SemanticNavWeb/0.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.app_args.client_timeout_sec)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_html(INDEX_HTML)
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
        elif path == "/api/explore/start":
            self.handle_explore_start()
        elif path == "/api/explore/stop":
            self.handle_explore_stop()
        elif path == "/api/memory/search":
            self.handle_memory_search()
        elif path == "/api/memory/load-run":
            self.handle_memory_load_run()
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

    def get_logs(self) -> list[dict]:
        parsed = urlparse(self.path)
        last_id = int(parse_qs(parsed.query).get("after", ["0"])[0])
        with self.state.lock:
            return [item for item in self.state.logs if item["id"] > last_id]

    def handle_memory_list(self) -> None:
        try:
            parsed = urlparse(self.path)
            run_id = parse_qs(parsed.query).get("run", ["latest"])[0]
            try:
                qdrant_memory = qdrant_scroll_memory(self.app_args)
            except Exception as exc:
                qdrant_memory = {"items": [], "qdrant_run_counts": {}, "qdrant_error": str(exc)}
            if run_id == "qdrant":
                payload = qdrant_memory
                payload["runs"] = list_exploration_runs()
                payload["selected_run"] = "qdrant"
            else:
                payload = load_run_memory(run_id)
                payload["qdrant_run_counts"] = qdrant_memory.get("qdrant_run_counts", {})
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

    def handle_memory_load_run(self) -> None:
        try:
            payload = self.read_json()
            run_id = str(payload.get("run", "")).strip()
            if run_id in ("", "latest"):
                run_id = find_exploration_run(run_id)["id"]
            if run_id == "qdrant":
                self.send_json({"ok": True, "loaded": False, "run": "qdrant"})
                return
            self.state.add_log("web", f"loading memory run into Qdrant: {run_id}")
            result = load_run_into_qdrant(self.app_args, run_id)
            self.state.add_log("web", f"loaded memory run into Qdrant: {run_id}")
            self.send_json({"ok": True, "loaded": True, **result})
        except subprocess.TimeoutExpired:
            message = "memory run load timed out"
            self.state.add_log("web", message)
            self.send_json({"ok": False, "error": message}, HTTPStatus.GATEWAY_TIMEOUT)
        except Exception as exc:
            self.state.add_log("web", f"Failed to load memory run: {exc}")
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
                explore_running = (
                    self.state.explore_process is not None
                    and self.state.explore_process.poll() is None
                )
            if running:
                self.send_json(
                    {"ok": False, "error": "navigation already running"},
                    HTTPStatus.CONFLICT,
                )
                return
            if explore_running:
                self.send_json(
                    {"ok": False, "error": "exploration mission already running"},
                    HTTPStatus.CONFLICT,
                )
                return
            if self.app_args.require_nav_ready:
                ready, summary = self.state.navigation_ready()
                if not ready:
                    message = (
                        "SLAM/Nav2 is not ready. Start "
                        "`ros2 launch launch/slam_nav2_launch.py` first. "
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
            with self.state.lock:
                proc = self.state.slam_process
                running = proc is not None and proc.poll() is None
                if running:
                    self.send_json({"ok": True, "started": False, "running": True})
                    return

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

    def handle_explore_start(self) -> None:
        try:
            ready, summary = self.state.navigation_ready()
            if not ready:
                message = "Cannot explore until SLAM/Nav2 and Qdrant are ready. " + summary
                self.state.add_log("web", message)
                self.send_json({"ok": False, "error": message}, HTTPStatus.CONFLICT)
                return

            with self.state.lock:
                nav_running = (
                    self.state.nav_process is not None
                    and self.state.nav_process.poll() is None
                )
                explore_running = (
                    self.state.explore_process is not None
                    and self.state.explore_process.poll() is None
                )
                if nav_running:
                    self.send_json(
                        {"ok": False, "error": "semantic navigation already running"},
                        HTTPStatus.CONFLICT,
                    )
                    return
                if explore_running:
                    self.send_json({"ok": True, "started": False, "running": True})
                    return

                command = build_explore_command(self.app_args)
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
                self.state.explore_process = proc
                self.state.explore_started_at = time.monotonic()
                self.state.explore_exit_code = None
            self.state.add_log("web", "started exploration: " + " ".join(shlex.quote(part) for part in command))
            threading.Thread(
                target=stream_process_output,
                args=(self.state, proc, "explore"),
                daemon=True,
            ).start()
            self.send_json({"ok": True, "started": True})
        except Exception as exc:
            self.state.add_log("web", f"Failed to start exploration: {exc}")
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_explore_stop(self) -> None:
        with self.state.lock:
            proc = self.state.explore_process
            running = proc is not None and proc.poll() is None
        if not running or proc is None:
            self.send_json({"ok": True, "stopped": False})
            return

        self.state.add_log("web", "stopping active exploration mission")
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        self.send_json({"ok": True, "stopped": True})

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
            "GET /api/events ",
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
    html { height: 100%; overflow: hidden; }
    body {
      margin: 0;
      height: 100%;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    main {
      height: 100vh;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 0;
      overflow: hidden;
    }
    .camera {
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
      border-right: 1px solid var(--line);
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
      min-height: 0;
      height: 100vh;
      display: flex;
      flex-direction: column;
      background: var(--panel);
      overflow: hidden;
    }
    .tabs {
      flex: 0 0 auto;
      display: grid;
      grid-template-columns: 1fr 1fr;
      border-bottom: 1px solid var(--line);
      background: #171a1e;
    }
    .tab {
      height: 44px;
      border-radius: 0;
      background: transparent;
      color: var(--muted);
      border: 0;
      border-right: 1px solid var(--line);
    }
    .tab:last-child { border-right: 0; }
    .tab.active {
      color: var(--text);
      background: var(--panel);
      box-shadow: inset 0 -2px 0 var(--accent);
    }
    .tab-panel {
      min-height: 0;
      flex: 1;
      display: none;
      flex-direction: column;
      overflow: hidden;
    }
    .tab-panel.active { display: flex; }
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
      min-height: 0;
      flex: 1;
      display: flex;
      flex-direction: column;
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
    .memory-grid::-webkit-scrollbar,
    pre::-webkit-scrollbar,
    .tab-panel::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }
    .memory-grid::-webkit-scrollbar-thumb,
    pre::-webkit-scrollbar-thumb,
    .tab-panel::-webkit-scrollbar-thumb {
      background: #4a535f;
      border-radius: 999px;
      border: 2px solid #101316;
    }
    .memory-grid::-webkit-scrollbar-track,
    pre::-webkit-scrollbar-track,
    .tab-panel::-webkit-scrollbar-track {
      background: #101316;
    }
    .memory-toolbar {
      flex: 0 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
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
      grid-template-columns: 1fr;
      gap: 10px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: #171a1e;
    }
    .memory-detail.visible { display: grid; }
    .memory-detail img {
      width: 100%;
      max-height: 260px;
      object-fit: cover;
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
      flex: 1;
      overflow: auto;
      padding: 12px;
      display: grid;
      grid-template-columns: 1fr;
      align-content: start;
      gap: 10px;
      background: #101316;
    }
    .memory-card {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      overflow: hidden;
      cursor: pointer;
    }
    .memory-card.selected { border-color: var(--accent); }
    .memory-card img {
      width: 100%;
      height: clamp(180px, 24vh, 280px);
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
    .src-explore { color: #70d36b; }
    .src-camera { color: var(--warn); }
    .src-http { color: var(--muted); }
    @media (max-width: 900px) {
      html, body { overflow: auto; }
      main {
        height: auto;
        min-height: 100vh;
        grid-template-columns: 1fr;
        grid-template-rows: 58vh minmax(420px, 42vh);
        overflow: visible;
      }
      .camera { border-right: 0; border-bottom: 1px solid var(--line); }
      aside { height: auto; min-height: 420px; }
    }
  </style>
</head>
<body>
  <main>
    <section class="camera">
      <div class="topbar">
        <div class="title">Semantic Nav Control</div>
        <div class="status"><span id="cameraDot" class="dot"></span><span id="cameraStatus">camera offline</span></div>
      </div>
      <div class="stream"><img src="/camera.mjpg" alt="robot camera"></div>
    </section>
    <aside>
      <nav class="tabs">
        <button id="tabControl" class="tab active" type="button">Control</button>
        <button id="tabMemory" class="tab" type="button">Memory DB</button>
      </nav>
      <section id="panelControl" class="tab-panel active">
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
            <button id="startExplore" class="secondary">Explore</button>
            <button id="stopExplore" class="secondary danger">Stop Explore</button>
          </div>
          <button id="stop" class="secondary danger">Stop active goal</button>
        </section>
        <section class="metrics">
          <div class="metric"><span>Navigation</span><strong id="navState">idle</strong></div>
          <div class="metric"><span>Runtime</span><strong id="runtime">-</strong></div>
          <div class="metric"><span>Qdrant</span><strong id="qdrantState">checking</strong></div>
          <div class="metric"><span>Qdrant URL</span><strong id="qdrantUrl">-</strong></div>
          <div class="metric"><span>Explore</span><strong id="exploreState">idle</strong></div>
          <div class="metric"><span>Explore runtime</span><strong id="exploreRuntime">-</strong></div>
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
        </section>
      </section>
      <section id="panelMemory" class="tab-panel">
        <div class="memory-toolbar">
          <strong id="memoryCount">Memory DB</strong>
          <button id="refreshMemory" class="secondary" type="button">Refresh</button>
          <div class="memory-run">
            <select id="memoryRun"></select>
            <span id="memoryRunInfo">-</span>
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
    </aside>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const logEl = $("log");
    let lastLogId = 0;
    let currentMemoryRun = "latest";

    function addLog(item) {
      if (item.id <= lastLogId) return;
      lastLogId = item.id;
      const cls = `src-${item.source}`;
      const line = `[${item.time}] ${item.source}: ${item.message}`;
      const span = document.createElement("span");
      span.className = cls;
      span.textContent = line + "\n";
      logEl.appendChild(span);
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

    function showTab(name) {
      const control = name === "control";
      $("tabControl").classList.toggle("active", control);
      $("tabMemory").classList.toggle("active", !control);
      $("panelControl").classList.toggle("active", control);
      $("panelMemory").classList.toggle("active", !control);
      if (!control) loadMemory();
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

    function populateRunSelector(runs, selectedRun) {
      const select = $("memoryRun");
      const previous = select.value;
      select.textContent = "";
      const qdrantOption = document.createElement("option");
      qdrantOption.value = "qdrant";
      qdrantOption.textContent = "Current Qdrant collection";
      select.appendChild(qdrantOption);
      for (const run of runs || []) {
        const option = document.createElement("option");
        option.value = run.id;
        const suffix = run.has_embeddings ? "embeddings" : "metadata only";
        option.textContent = `${run.id} · ${run.image_count} images · ${suffix}`;
        select.appendChild(option);
      }
      if (selectedRun && [...select.options].some((option) => option.value === selectedRun)) {
        select.value = selectedRun;
      } else if (previous && [...select.options].some((option) => option.value === previous)) {
        select.value = previous;
      } else if (runs && runs.length) {
        select.value = runs[0].id;
      } else {
        select.value = "qdrant";
      }
      currentMemoryRun = select.value;
    }

    async function ensureSelectedRunForGo() {
      if (!currentMemoryRun || currentMemoryRun === "qdrant") return true;
      const goButton = $("go");
      const oldText = goButton.textContent;
      goButton.disabled = true;
      goButton.textContent = "Load DB";
      try {
        const res = await fetch("/api/memory/load-run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run: currentMemoryRun })
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          console.warn(data.error || "Failed to load selected run");
          return false;
        }
        currentMemoryRun = "qdrant";
        await loadMemory();
        return true;
      } finally {
        goButton.textContent = oldText;
      }
    }

    function renderMemory(data) {
      const items = data.items || [];
      const grid = $("memoryGrid");
      grid.textContent = "";
      populateRunSelector(data.runs || [], data.selected_run);
      $("memoryCount").textContent = `Memory DB (${items.length})`;
      const selectedLabel = data.selected_run === "qdrant" ? "Current Qdrant collection" : (data.selected_run || "-");
      $("memoryRunInfo").textContent = `${selectedLabel} · ${qdrantInfo(data.qdrant_run_counts)}`;
      if (!items.length) {
        const empty = document.createElement("div");
        empty.className = "memory-card";
        empty.innerHTML = "<div><strong>No memory records</strong><span>The selected run has no metadata/images</span></div>";
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
        const res = await fetch(`/api/memory?run=${encodeURIComponent(currentMemoryRun)}`, { cache: "no-store" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Memory request failed");
        renderMemory(data);
      } catch (err) {
        $("memoryCount").textContent = "Memory DB unavailable";
        $("memoryRunInfo").textContent = String(err);
        $("memoryGrid").textContent = "";
        showMemoryDetail(null);
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
      try {
        const res = await fetch("/api/status", { cache: "no-store" });
        const data = await res.json();
        const camera = data.camera;
        const nav = data.navigation;
        const slam = data.slam;
        const qdrant = data.qdrant;
        const explore = data.explore;
        const system = data.system;
        const fresh = camera.has_frame && camera.frame_age_sec !== null && camera.frame_age_sec < 2.5;
        $("cameraDot").classList.toggle("ok", fresh);
        $("cameraStatus").textContent = fresh ? `camera live (${camera.frame_age_sec}s)` : "camera offline";
        $("navState").textContent = nav.running ? `running: ${nav.query}` : `idle (${nav.last_exit_code ?? "-"})`;
        $("runtime").textContent = nav.runtime_sec === null ? "-" : `${nav.runtime_sec}s`;
        $("qdrantState").textContent = qdrant.ready ? "ready" : qdrant.summary;
        $("qdrantUrl").textContent = qdrant.url || "-";
        $("exploreState").textContent = explore.running ? "running" : `idle (${explore.last_exit_code ?? "-"})`;
        $("exploreRuntime").textContent = explore.runtime_sec === null ? "-" : `${explore.runtime_sec}s`;
        $("slamProcess").textContent = slam.running ? "running" : `idle (${slam.last_exit_code ?? "-"})`;
        $("slamRuntime").textContent = slam.runtime_sec === null ? "-" : `${slam.runtime_sec}s`;
        $("systemState").textContent = system.ready ? "ready" : system.summary;
        $("tfState").textContent = system.goal_frame ? "map -> base_link" : "missing";
        $("frames").textContent = String(camera.frame_count);
        $("imageFrame").textContent = camera.image_frame || "-";
        $("go").disabled = nav.running || explore.running || !system.ready || !qdrant.ready;
        $("stop").disabled = !nav.running;
        $("startQdrant").disabled = qdrant.ready || explore.running;
        $("stopQdrant").disabled = !qdrant.ready || explore.running;
        $("startSlam").disabled = slam.running || explore.running;
        $("stopSlam").disabled = !slam.running || explore.running;
        $("startExplore").disabled = nav.running || explore.running || !system.ready || !qdrant.ready;
        $("stopExplore").disabled = !explore.running;
      } catch (err) {
        $("cameraDot").classList.remove("ok");
        $("cameraStatus").textContent = "panel disconnected";
      }
    }

    async function navigate() {
      const query = $("query").value.trim();
      if (!query) return;
      const ready = await ensureSelectedRunForGo();
      if (!ready) {
        refreshStatus();
        return;
      }
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

    async function startExplore() {
      await fetch("/api/explore/start", { method: "POST" });
      refreshStatus();
    }

    async function stopExplore() {
      await fetch("/api/explore/stop", { method: "POST" });
      refreshStatus();
    }

    $("go").addEventListener("click", navigate);
    $("stop").addEventListener("click", stopNav);
    $("startQdrant").addEventListener("click", startQdrant);
    $("stopQdrant").addEventListener("click", stopQdrant);
    $("startSlam").addEventListener("click", startSlam);
    $("stopSlam").addEventListener("click", stopSlam);
    $("startExplore").addEventListener("click", startExplore);
    $("stopExplore").addEventListener("click", stopExplore);
    $("tabControl").addEventListener("click", () => showTab("control"));
    $("tabMemory").addEventListener("click", () => showTab("memory"));
    $("refreshMemory").addEventListener("click", loadMemory);
    $("memoryRun").addEventListener("change", () => {
      currentMemoryRun = $("memoryRun").value || "latest";
      loadMemory();
    });
    $("query").addEventListener("keydown", (event) => {
      if (event.key === "Enter") navigate();
    });

    const events = new EventSource("/api/events");
    events.onmessage = (event) => addLog(JSON.parse(event.data));
    setInterval(refreshStatus, 800);
    refreshStatus();
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
    parser.add_argument("--client-timeout-sec", type=float, default=6.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--goal-frame", default="map")
    parser.add_argument("--max-goal-distance", type=float, default=4.0)
    parser.add_argument("--result-timeout-sec", type=float, default=120.0)
    parser.add_argument("--feedback-log-period-sec", type=float, default=4.0)
    parser.add_argument("--slam-log-level", default="info")
    parser.add_argument("--explore-initial-tf-timeout-sec", type=float, default=45.0)
    parser.add_argument("--explore-max-goals", type=int, default=1)
    parser.add_argument("--explore-max-goal-distance", type=float, default=4.0)
    parser.add_argument("--explore-goal-timeout-sec", type=float, default=60.0)
    parser.add_argument("--explore-return-timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--require-nav-ready",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject web navigation requests until /map, /navigate_to_pose, and map->base_link are available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = SharedState()
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
            explore_proc = state.explore_process
        if explore_proc is not None and explore_proc.poll() is None:
            try:
                os.killpg(explore_proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
