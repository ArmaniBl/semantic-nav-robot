#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time


DEFAULT_NODES = (
    "/controller_server",
    "/planner_server",
    "/behavior_server",
    "/bt_navigator",
)


def run_ros2(args: list[str], timeout_sec: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ros2", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
        check=False,
    )


def lifecycle_state(node: str, timeout_sec: float) -> str | None:
    result = run_ros2(["lifecycle", "get", node], timeout_sec)
    if result.returncode != 0:
        return None
    first = (result.stdout or "").strip().splitlines()[0]
    return first.split()[0] if first else None


def set_transition(node: str, transition: str, timeout_sec: float) -> None:
    result = run_ros2(["lifecycle", "set", node, transition], timeout_sec)
    output = (result.stdout or "").strip()
    if result.returncode != 0:
        raise RuntimeError(f"{node} {transition} failed: {output}")
    print(f"{node}: {transition}: {output or 'ok'}", flush=True)


def wait_for_node(node: str, deadline: float, timeout_sec: float) -> str:
    while time.monotonic() < deadline:
        state = lifecycle_state(node, timeout_sec)
        if state is not None:
            return state
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for lifecycle node {node}")


def ensure_active(node: str, deadline: float, timeout_sec: float) -> None:
    state = wait_for_node(node, deadline, timeout_sec)
    if state == "active":
        print(f"{node}: already active", flush=True)
        return
    if state == "unconfigured":
        set_transition(node, "configure", timeout_sec)
        state = wait_for_node(node, deadline, timeout_sec)
    if state == "inactive":
        set_transition(node, "activate", timeout_sec)
        state = wait_for_node(node, deadline, timeout_sec)
    if state != "active":
        raise RuntimeError(f"{node} is {state}, expected active")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure Nav2 lifecycle nodes become active.")
    parser.add_argument("--nodes", nargs="+", default=list(DEFAULT_NODES))
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--service-timeout-sec", type=float, default=10.0)
    parser.add_argument("--start-delay-sec", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    time.sleep(args.start_delay_sec)
    try:
        for node in args.nodes:
            deadline = time.monotonic() + args.timeout_sec
            ensure_active(node, deadline, args.service_timeout_sec)
    except Exception as exc:
        print(f"Nav2 lifecycle activation failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print("Nav2 lifecycle nodes are active", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
