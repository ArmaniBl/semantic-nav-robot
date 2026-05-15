# Semantic Navigation Robot Control

ROS 2 Jazzy project for semantic robot navigation with:

- Isaac Sim / ROS camera streaming;
- SLAM Toolbox + Nav2;
- RuCLIP visual embeddings;
- Qdrant semantic visual memory;
- a local web control panel for camera view, semantic navigation, and exploration.

## Quick Start

```bash
cd /home/arman/test/diplom
source /opt/ros/jazzy/setup.bash
docker compose -f compose.qdrant.yml up -d
python3 web_control_panel.py --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:8765/
```

From the web panel:

1. Wait for `Qdrant` to become `ready`.
2. Click `Start SLAM/Nav2`.
3. Wait for `SLAM/Nav2` to become `ready`.
4. Use `Explore` to collect keyframes and rebuild semantic memory.
5. Use the semantic command input and `Go` to navigate by text query.

The `Memory DB` tab shows keyframes from exploration runs, including image, pose
frame, coordinates, score when available, topics, and file path. The newest
`data/exploration_runs/run_*` folder is selected by default, and the tab also
shows which run is currently loaded into Qdrant. When `Go` is pressed, the panel
shows the image selected by semantic search for that text query.

## Stop Runtime Processes

```bash
pkill -f web_control_panel.py || true
pkill -f exploration_mission.py || true
pkill -f frontier_explorer.py || true
pkill -f ros_keyframe_recorder.py || true
pkill -f semantic_nav_to_pose.py || true
pkill -f slam_nav2_launch.py || true
docker compose -f compose.qdrant.yml down
```

More detailed progress notes are in `CURRENT_PROGRESS.md`.
