#!/usr/bin/env python3
import argparse
import html
import json
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
from urllib.request import urlopen

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
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def get_logs(self) -> list[dict]:
        parsed = urlparse(self.path)
        last_id = int(parse_qs(parsed.query).get("after", ["0"])[0])
        with self.state.lock:
            return [item for item in self.state.logs if item["id"] > last_id]

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
      grid-template-columns: minmax(0, 1fr) 380px;
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
      min-height: 56px;
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
      font-size: 16px;
      line-height: 1.2;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
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
    .control {
      flex: 0 0 auto;
      padding: 18px;
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
      margin-top: 10px;
    }
    input {
      width: 100%;
      min-width: 0;
      height: 42px;
      border: 1px solid var(--line);
      background: #101316;
      color: var(--text);
      border-radius: 6px;
      padding: 0 12px;
      font-size: 15px;
      outline: none;
    }
    input:focus { border-color: var(--accent); }
    button {
      height: 42px;
      border: 0;
      border-radius: 6px;
      padding: 0 14px;
      background: var(--accent);
      color: #061412;
      font-weight: 700;
      font-size: 14px;
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
    .metrics {
      flex: 0 0 auto;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
    }
    .metric {
      min-height: 58px;
      padding: 10px;
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
      font-size: 14px;
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
      <section class="control">
        <label for="query">Semantic command</label>
        <div class="query-row">
          <input id="query" type="text" value="дорога" autocomplete="off">
          <button id="go">Go</button>
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
    </aside>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const logEl = $("log");
    let lastLogId = 0;

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
