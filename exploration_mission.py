#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros import Buffer, TransformException, TransformListener


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "data" / "exploration_runs"


class ExplorationMission(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("exploration_mission")
        self.args = args
        self.processes: list[subprocess.Popen] = []
        self.current_goal_handle = None
        self.last_feedback_log_time = 0.0
        if self.has_parameter("use_sim_time"):
            self.set_parameters(
                [Parameter("use_sim_time", Parameter.Type.BOOL, args.use_sim_time)]
            )
        else:
            self.declare_parameter("use_sim_time", args.use_sim_time)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.action_client = ActionClient(self, NavigateToPose, args.action_name)

    def wait_for_robot_pose(self) -> PoseStamped:
        deadline = time.monotonic() + self.args.initial_tf_timeout_sec
        last_error = None
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.args.goal_frame,
                    self.args.robot_base_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=self.args.tf_timeout_sec),
                )
                t = transform.transform.translation
                r = transform.transform.rotation
                pose = PoseStamped()
                pose.header.frame_id = self.args.goal_frame
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position.x = t.x
                pose.pose.position.y = t.y
                pose.pose.position.z = 0.0
                pose.pose.orientation.x = r.x
                pose.pose.orientation.y = r.y
                pose.pose.orientation.z = r.z
                pose.pose.orientation.w = r.w
                return pose
            except TransformException as exc:
                last_error = exc
                rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError(
            f"No TF {self.args.goal_frame} -> {self.args.robot_base_frame}: {last_error}"
        )

    def wait_for_future(self, future, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return future.done()

    def return_to_start(self, start_pose: PoseStamped) -> bool:
        self.get_logger().info(
            f"Returning to start: frame={start_pose.header.frame_id} "
            f"x={start_pose.pose.position.x:.3f} y={start_pose.pose.position.y:.3f}"
        )
        if not self.action_client.wait_for_server(timeout_sec=self.args.action_timeout_sec):
            self.get_logger().error(f"Nav2 action server unavailable: {self.args.action_name}")
            return False

        start_pose.header.stamp = self.get_clock().now().to_msg()
        goal = NavigateToPose.Goal()
        goal.pose = start_pose
        goal_future = self.action_client.send_goal_async(goal, feedback_callback=self.on_feedback)
        if not self.wait_for_future(goal_future, self.args.action_timeout_sec):
            self.get_logger().error("Timed out while sending return goal")
            return False

        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Nav2 rejected return goal")
            return False

        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        if not self.wait_for_future(result_future, self.args.return_timeout_sec):
            self.get_logger().warn("Return goal timed out; requesting cancel")
            cancel_future = goal_handle.cancel_goal_async()
            self.wait_for_future(cancel_future, self.args.action_timeout_sec)
            self.current_goal_handle = None
            return False

        result = result_future.result()
        self.current_goal_handle = None
        if (
            result is not None
            and result.status == GoalStatus.STATUS_SUCCEEDED
            and result.result.error_code == NavigateToPose.Result.NONE
        ):
            self.get_logger().info("Returned to start")
            return True

        if result is None:
            self.get_logger().error("Nav2 returned no result for return goal")
        else:
            self.get_logger().warn(
                f"Return failed: status={result.status}, "
                f"error_code={result.result.error_code}, error_msg={result.result.error_msg!r}"
            )
        return False

    def on_feedback(self, feedback_msg) -> None:
        now = time.monotonic()
        if now - self.last_feedback_log_time < self.args.feedback_log_period_sec:
            return
        self.last_feedback_log_time = now
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f"Return feedback: distance_remaining={feedback.distance_remaining:.3f}, "
            f"recoveries={feedback.number_of_recoveries}"
        )

    def launch_process(self, name: str, command: list[str]) -> subprocess.Popen:
        self.get_logger().info(f"Starting {name}: {' '.join(command)}")
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
        self.processes.append(proc)
        threading.Thread(target=self.stream_process, args=(name, proc), daemon=True).start()
        return proc

    def stream_process(self, name: str, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{name}] {line}", end="", flush=True)

    def stop_process(self, name: str, proc: subprocess.Popen, timeout_sec: float = 10.0) -> int:
        if proc.poll() is not None:
            return proc.returncode
        self.get_logger().info(f"Stopping {name}")
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            return proc.poll() or 0
        deadline = time.monotonic() + timeout_sec
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if proc.poll() is None:
            self.get_logger().warn(f"{name} did not stop after SIGINT; terminating")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return proc.wait(timeout=timeout_sec)

    def run_command(self, name: str, command: list[str]) -> int:
        self.get_logger().info(f"Running {name}: {' '.join(command)}")
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
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{name}] {line}", end="", flush=True)
        return proc.wait()

    def build_run_dir(self) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.args.output_root / f"run_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def run(self) -> int:
        run_dir = self.build_run_dir()
        keyframes_dir = run_dir / "keyframes"
        metadata_path = keyframes_dir / "metadata.jsonl"
        embeddings_path = keyframes_dir / "embeddings.jsonl"
        self.get_logger().info(f"Exploration run dir: {run_dir}")

        start_pose = self.wait_for_robot_pose()
        self.get_logger().info(
            f"Start pose: frame={start_pose.header.frame_id} "
            f"x={start_pose.pose.position.x:.3f} y={start_pose.pose.position.y:.3f}"
        )

        recorder = self.launch_process(
            "recorder",
            [
                sys.executable,
                str(PROJECT_ROOT / "ros_keyframe_recorder.py"),
                "--output-dir",
                str(keyframes_dir),
                "--min-time-delta-sec",
                str(self.args.keyframe_min_time_delta_sec),
                "--min-translation-delta-m",
                str(self.args.keyframe_min_translation_delta_m),
                "--min-rotation-delta-rad",
                str(self.args.keyframe_min_rotation_delta_rad),
            ],
        )

        frontier_code = 1
        try:
            frontier = self.launch_process(
                "frontier",
                [
                    sys.executable,
                    str(PROJECT_ROOT / "frontier_explorer.py"),
                    "--initial-tf-timeout-sec",
                    str(self.args.initial_tf_timeout_sec),
                    "--max-goals",
                    str(self.args.max_goals),
                    "--max-goal-distance",
                    str(self.args.max_goal_distance),
                    "--goal-timeout-sec",
                    str(self.args.goal_timeout_sec),
                    "--feedback-log-period-sec",
                    str(self.args.feedback_log_period_sec),
                    "--min-frontier-size",
                    str(self.args.min_frontier_size),
                ],
            )
            frontier_code = frontier.wait()
            self.get_logger().info(f"Frontier explorer exited with code {frontier_code}")
            returned = self.return_to_start(start_pose)
        finally:
            self.stop_process("recorder", recorder)

        if not metadata_path.exists() or metadata_path.stat().st_size == 0:
            self.get_logger().error(f"No keyframes recorded: {metadata_path}")
            return 5

        embed_code = self.run_command(
            "embed",
            [
                str(PROJECT_ROOT / ".ruclip_venv" / "bin" / "python"),
                str(PROJECT_ROOT / "ruclip_embed_keyframes.py"),
                "--metadata",
                str(metadata_path),
                "--output",
                str(embeddings_path),
            ],
        )
        if embed_code != 0:
            self.get_logger().error(f"Embedding failed with code {embed_code}")
            return embed_code

        load_command = [
            str(PROJECT_ROOT / ".ruclip_venv" / "bin" / "python"),
            str(PROJECT_ROOT / "qdrant_load_keyframes.py"),
            "--embeddings",
            str(embeddings_path),
        ]
        if self.args.recreate_collection:
            load_command.append("--recreate")
        load_code = self.run_command("qdrant_load", load_command)
        if load_code != 0:
            self.get_logger().error(f"Qdrant load failed with code {load_code}")
            return load_code

        self.get_logger().info(f"Mission complete: returned={returned}, run_dir={run_dir}")
        return 0 if returned and frontier_code == 0 else 6

    def cancel_current_goal(self) -> None:
        if self.current_goal_handle is None:
            return
        try:
            cancel_future = self.current_goal_handle.cancel_goal_async()
            self.wait_for_future(cancel_future, self.args.action_timeout_sec)
        finally:
            self.current_goal_handle = None

    def close(self) -> None:
        self.cancel_current_goal()
        for proc in list(self.processes):
            if proc.poll() is None:
                self.stop_process("child", proc)
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
    parser = argparse.ArgumentParser(
        description="Explore with frontier navigation, record keyframes, rebuild Qdrant memory, and return home."
    )
    parser.add_argument("--output-root", type=Path, default=RUNS_DIR)
    parser.add_argument("--action-name", default="/navigate_to_pose")
    parser.add_argument("--goal-frame", default="map")
    parser.add_argument("--robot-base-frame", default="base_link")
    parser.add_argument("--use-sim-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-tf-timeout-sec", type=float, default=20.0)
    parser.add_argument("--tf-timeout-sec", type=float, default=2.0)
    parser.add_argument("--action-timeout-sec", type=float, default=10.0)
    parser.add_argument("--return-timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-goals", type=int, default=3)
    parser.add_argument("--max-goal-distance", type=float, default=4.0)
    parser.add_argument("--goal-timeout-sec", type=float, default=110.0)
    parser.add_argument("--feedback-log-period-sec", type=float, default=4.0)
    parser.add_argument("--min-frontier-size", type=int, default=8)
    parser.add_argument("--keyframe-min-time-delta-sec", type=float, default=1.0)
    parser.add_argument("--keyframe-min-translation-delta-m", type=float, default=0.25)
    parser.add_argument("--keyframe-min-rotation-delta-rad", type=float, default=0.25)
    parser.add_argument("--recreate-collection", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = ExplorationMission(args)
    try:
        exit_code = node.run()
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as exc:
        node.get_logger().error(str(exc))
        exit_code = 1
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
