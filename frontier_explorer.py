#!/usr/bin/env python3
import argparse
import math
import time
from collections import deque
from dataclasses import dataclass

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass(frozen=True)
class FrontierCluster:
    cells: list[tuple[int, int]]
    goal_cell: tuple[int, int]
    centroid_x: float
    centroid_y: float
    goal_x: float
    goal_y: float
    distance: float
    score: float


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


class FrontierExplorer(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("frontier_explorer")
        self.args = args
        self.map_msg: OccupancyGrid | None = None
        self.failed_goals: list[tuple[float, float]] = []
        self.last_feedback_log_time = 0.0
        self.warned_latest_tf_compose = False

        if self.has_parameter("use_sim_time"):
            self.set_parameters(
                [Parameter("use_sim_time", Parameter.Type.BOOL, args.use_sim_time)]
            )
        else:
            self.declare_parameter("use_sim_time", args.use_sim_time)

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(OccupancyGrid, args.map_topic, self.on_map, map_qos)
        self.action_client = ActionClient(self, NavigateToPose, args.action_name)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

    def on_map(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg

    def wait_for_map(self) -> bool:
        deadline = time.monotonic() + self.args.map_timeout_sec
        while rclpy.ok() and self.map_msg is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.map_msg is None:
            self.get_logger().error(f"No OccupancyGrid received on {self.args.map_topic}")
            return False
        return True

    def robot_xy(self) -> tuple[float, float]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.args.goal_frame,
                self.args.robot_base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.args.tf_timeout_sec),
            )
            t = transform.transform.translation
            return t.x, t.y
        except TransformException:
            if not self.args.allow_latest_tf_compose:
                raise
            return self.robot_xy_from_latest_edges()

    def robot_xy_from_latest_edges(self) -> tuple[float, float]:
        if self.args.goal_frame == self.args.odom_frame:
            odom_base = self.tf_buffer.lookup_transform(
                self.args.odom_frame,
                self.args.robot_base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.args.tf_timeout_sec),
            )
            t = odom_base.transform.translation
            return t.x, t.y

        map_odom = self.tf_buffer.lookup_transform(
            self.args.goal_frame,
            self.args.odom_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=self.args.tf_timeout_sec),
        )
        odom_base = self.tf_buffer.lookup_transform(
            self.args.odom_frame,
            self.args.robot_base_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=self.args.tf_timeout_sec),
        )
        if not self.warned_latest_tf_compose:
            self.get_logger().warn(
                "Using latest-edge TF compose for robot pose; direct TF lookup had "
                "no exact timestamp overlap yet"
            )
            self.warned_latest_tf_compose = True

        map_odom_t = map_odom.transform.translation
        odom_base_t = odom_base.transform.translation
        yaw = yaw_from_quaternion(map_odom.transform.rotation)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return (
            map_odom_t.x + odom_base_t.x * cos_yaw - odom_base_t.y * sin_yaw,
            map_odom_t.y + odom_base_t.x * sin_yaw + odom_base_t.y * cos_yaw,
        )

    def wait_for_robot_pose(self) -> bool:
        deadline = time.monotonic() + self.args.initial_tf_timeout_sec
        last_error = None
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                self.robot_xy()
                return True
            except TransformException as exc:
                last_error = exc
                rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().error(
            f"No TF {self.args.goal_frame} -> {self.args.robot_base_frame}: {last_error}"
        )
        return False

    def index(self, x: int, y: int, width: int) -> int:
        return y * width + x

    def cell_to_world(self, grid: OccupancyGrid, x: float, y: float) -> tuple[float, float]:
        info = grid.info
        resolution = info.resolution
        origin = info.origin.position
        yaw = yaw_from_quaternion(info.origin.orientation)
        local_x = (x + 0.5) * resolution
        local_y = (y + 0.5) * resolution
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return (
            origin.x + local_x * cos_yaw - local_y * sin_yaw,
            origin.y + local_x * sin_yaw + local_y * cos_yaw,
        )

    def is_free(self, value: int) -> bool:
        return 0 <= value <= self.args.free_threshold

    def is_frontier_cell(self, grid: OccupancyGrid, x: int, y: int) -> bool:
        width = grid.info.width
        height = grid.info.height
        data = grid.data
        if not self.is_free(data[self.index(x, y, width)]):
            return False
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height:
                if data[self.index(nx, ny, width)] == -1:
                    return True
        return False

    def find_frontier_cells(self, grid: OccupancyGrid) -> set[tuple[int, int]]:
        width = grid.info.width
        height = grid.info.height
        frontiers: set[tuple[int, int]] = set()
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if self.is_frontier_cell(grid, x, y):
                    frontiers.add((x, y))
        return frontiers

    def cluster_frontiers(
        self,
        grid: OccupancyGrid,
        frontier_cells: set[tuple[int, int]],
        robot_x: float,
        robot_y: float,
    ) -> list[FrontierCluster]:
        remaining = set(frontier_cells)
        clusters: list[FrontierCluster] = []
        neighbors = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]

        while remaining:
            seed = remaining.pop()
            queue: deque[tuple[int, int]] = deque([seed])
            cells = [seed]
            while queue:
                x, y = queue.popleft()
                for dx, dy in neighbors:
                    neighbor = (x + dx, y + dy)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
                        cells.append(neighbor)

            if len(cells) < self.args.min_frontier_size:
                continue

            centroid_cell_x = sum(cell[0] for cell in cells) / len(cells)
            centroid_cell_y = sum(cell[1] for cell in cells) / len(cells)
            centroid_x, centroid_y = self.cell_to_world(
                grid, centroid_cell_x, centroid_cell_y
            )
            goal_cell = min(
                cells,
                key=lambda cell: (cell[0] - centroid_cell_x) ** 2
                + (cell[1] - centroid_cell_y) ** 2,
            )
            goal_x, goal_y = self.cell_to_world(grid, goal_cell[0], goal_cell[1])
            distance = math.hypot(goal_x - robot_x, goal_y - robot_y)
            if distance < self.args.min_goal_distance:
                continue
            if self.args.max_goal_distance is not None and distance > self.args.max_goal_distance:
                continue
            if self.is_blacklisted(goal_x, goal_y):
                continue

            score = (
                len(cells) * self.args.frontier_size_weight
                - distance * self.args.distance_weight
            )
            clusters.append(
                FrontierCluster(
                    cells=cells,
                    goal_cell=goal_cell,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    goal_x=goal_x,
                    goal_y=goal_y,
                    distance=distance,
                    score=score,
                )
            )

        clusters.sort(key=lambda cluster: cluster.score, reverse=True)
        return clusters

    def is_blacklisted(self, x: float, y: float) -> bool:
        radius = self.args.failed_goal_blacklist_radius
        return any(math.hypot(x - bx, y - by) <= radius for bx, by in self.failed_goals)

    def build_goal(self, cluster: FrontierCluster, robot_x: float, robot_y: float) -> PoseStamped:
        yaw = math.atan2(cluster.goal_y - robot_y, cluster.goal_x - robot_x)
        qx, qy, qz, qw = quaternion_from_yaw(yaw)

        goal = PoseStamped()
        goal.header.frame_id = self.args.goal_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = cluster.goal_x
        goal.pose.position.y = cluster.goal_y
        goal.pose.position.z = 0.0
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        return goal

    def select_candidates(self) -> list[FrontierCluster]:
        if self.map_msg is None:
            return []
        robot_x, robot_y = self.robot_xy()
        frontier_cells = self.find_frontier_cells(self.map_msg)
        clusters = self.cluster_frontiers(self.map_msg, frontier_cells, robot_x, robot_y)
        self.get_logger().info(
            f"Frontiers: cells={len(frontier_cells)}, clusters={len(clusters)}, "
            f"robot=({robot_x:.2f}, {robot_y:.2f})"
        )
        for rank, cluster in enumerate(clusters[: self.args.log_top_k], start=1):
            self.get_logger().info(
                f"Candidate #{rank}: goal=({cluster.goal_x:.2f}, {cluster.goal_y:.2f}), "
                f"distance={cluster.distance:.2f}m, size={len(cluster.cells)}, "
                f"score={cluster.score:.2f}"
            )
        return clusters

    def send_goal(self, goal_pose: PoseStamped, cluster: FrontierCluster) -> bool:
        if self.args.dry_run:
            self.get_logger().info(
                f"Dry run: would send frontier goal x={goal_pose.pose.position.x:.3f} "
                f"y={goal_pose.pose.position.y:.3f}, size={len(cluster.cells)}"
            )
            return True

        if not self.action_client.wait_for_server(timeout_sec=self.args.action_timeout_sec):
            self.get_logger().error(f"Nav2 action server is not available: {self.args.action_name}")
            return False

        goal = NavigateToPose.Goal()
        goal.pose = goal_pose
        goal.behavior_tree = self.args.behavior_tree
        self.get_logger().info(
            f"Sending frontier goal: frame={goal_pose.header.frame_id} "
            f"x={goal_pose.pose.position.x:.3f} y={goal_pose.pose.position.y:.3f}"
        )

        goal_future = self.action_client.send_goal_async(
            goal,
            feedback_callback=self.on_feedback,
        )
        if not self.wait_for_future(goal_future, self.args.action_timeout_sec):
            self.get_logger().error("Timed out while sending frontier goal")
            return False

        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Nav2 rejected frontier goal")
            return False

        result_future = goal_handle.get_result_async()
        if not self.wait_for_future(result_future, self.args.goal_timeout_sec):
            self.get_logger().warn("Frontier goal timed out; requesting cancel")
            cancel_future = goal_handle.cancel_goal_async()
            self.wait_for_future(cancel_future, self.args.action_timeout_sec)
            return False

        result = result_future.result()
        if result is None:
            self.get_logger().error("Nav2 returned no result")
            return False

        if (
            result.status == GoalStatus.STATUS_SUCCEEDED
            and result.result.error_code == NavigateToPose.Result.NONE
        ):
            self.get_logger().info("Frontier goal reached")
            return True

        self.get_logger().warn(
            f"Frontier goal failed: status={result.status}, "
            f"error_code={result.result.error_code}, error_msg={result.result.error_msg!r}"
        )
        return False

    def wait_for_future(self, future, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return future.done()

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

    def run(self) -> int:
        if not self.wait_for_map():
            return 1
        if not self.wait_for_robot_pose():
            return 2

        goals_sent = 0
        while rclpy.ok():
            try:
                candidates = self.select_candidates()
            except TransformException as exc:
                self.get_logger().error(f"TF lookup failed: {exc}")
                return 2

            if not candidates:
                self.get_logger().warn("No usable frontier candidates found")
                return 3 if goals_sent == 0 else 0

            goal_sent_in_round = False
            robot_x, robot_y = self.robot_xy()
            for cluster in candidates[: self.args.try_top_k]:
                if self.args.max_goals is not None and goals_sent >= self.args.max_goals:
                    self.get_logger().info(f"Reached max goals limit: {self.args.max_goals}")
                    return 0
                goal_pose = self.build_goal(cluster, robot_x, robot_y)
                success = self.send_goal(goal_pose, cluster)
                goals_sent += 1
                goal_sent_in_round = True
                if success and self.args.once:
                    return 0
                if success:
                    break
                self.failed_goals.append((cluster.goal_x, cluster.goal_y))

            if self.args.dry_run or self.args.once:
                return 0 if goal_sent_in_round else 4
            if self.args.max_goals is not None and goals_sent >= self.args.max_goals:
                self.get_logger().info(f"Reached max goals limit: {self.args.max_goals}")
                return 0

            rclpy.spin_once(self, timeout_sec=self.args.replan_delay_sec)

        return 130

    def close(self) -> None:
        self.tf_listener.unregister()
        executor = getattr(self.tf_listener, "executor", None)
        if executor is not None:
            executor.shutdown()
        thread = getattr(self.tf_listener, "dedicated_listener_thread", None)
        if thread is not None:
            thread.join(timeout=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal frontier exploration client for the SLAM/Nav2 profile."
    )
    parser.add_argument("--map-topic", default="/map")
    parser.add_argument("--action-name", default="/navigate_to_pose")
    parser.add_argument("--goal-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--robot-base-frame", default="base_link")
    parser.add_argument("--behavior-tree", default="")
    parser.add_argument("--use-sim-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--map-timeout-sec", type=float, default=10.0)
    parser.add_argument("--initial-tf-timeout-sec", type=float, default=15.0)
    parser.add_argument("--tf-timeout-sec", type=float, default=2.0)
    parser.add_argument(
        "--allow-latest-tf-compose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback to latest map->odom and odom->base_link transforms if direct lookup has no timestamp overlap.",
    )
    parser.add_argument("--action-timeout-sec", type=float, default=10.0)
    parser.add_argument("--goal-timeout-sec", type=float, default=90.0)
    parser.add_argument("--feedback-log-period-sec", type=float, default=2.0)
    parser.add_argument("--replan-delay-sec", type=float, default=1.0)
    parser.add_argument("--free-threshold", type=int, default=20)
    parser.add_argument("--min-frontier-size", type=int, default=8)
    parser.add_argument("--min-goal-distance", type=float, default=0.8)
    parser.add_argument("--max-goal-distance", type=float, default=4.0)
    parser.add_argument("--frontier-size-weight", type=float, default=0.05)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--failed-goal-blacklist-radius", type=float, default=0.75)
    parser.add_argument("--try-top-k", type=int, default=5)
    parser.add_argument("--log-top-k", type=int, default=5)
    parser.add_argument("--max-goals", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = FrontierExplorer(args)
    try:
        exit_code = node.run()
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
