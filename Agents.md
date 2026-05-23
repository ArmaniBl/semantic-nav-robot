# Project Notes

Этот файл хранит рабочие правила проекта и не является частью runtime.

## Цель

Собрать управляемый сценарий навигации мобильного робота:

1. записать визуальные keyframes с pose;
2. загрузить записи в Qdrant;
3. выбрать collection;
4. отправить робота к найденной позе наблюдения через Nav2;
5. проверить live camera во время движения и после прибытия.

## Текущий Стек

```text
OS: Ubuntu/Linux
ROS2: Jazzy
Simulator: NVIDIA Isaac Sim 6.0
Navigation: Nav2 Jazzy
ROS client library: rclpy
Vector storage: Qdrant
Computer vision utilities: OpenCV, NumPy
```

## Runtime

Основной runtime запускается через:

```bash
./restart_semantic_nav.sh
```

Web UI:

```text
http://127.0.0.1:8765/
```

Основной launch для движения:

```text
launch/nav2_odom_launch.py
```

Цели отправляются в `odom`.

## Навигация

Основной программный интерфейс Nav2:

```text
Action: /navigate_to_pose
Type: nav2_msgs/action/NavigateToPose
```

`/goal_pose` использовать только для ручной отладки.

Navigation goals должны быть `geometry_msgs/PoseStamped` с корректными:

- `header.frame_id`;
- `header.stamp`;
- `pose.position`;
- `pose.orientation`.

## Данные Keyframe

Каждый payload в Qdrant должен содержать:

```json
{
  "memory_id": "keyframe_000001",
  "run_id": "run_YYYYMMDD_HHMMSS",
  "image_path": "...",
  "pose_frame": "odom",
  "pose": {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
  },
  "status": "active"
}
```

Разрешенные значения `status`:

- `active`;
- `stale`;
- `uncertain`.

## Правила Работы

- Не смешивать старые collections с новым запуском симуляции без проверки.
- Не отправлять goal в frame, для которого нет TF.
- Не считать keyframe координатой объекта: это поза наблюдения.
- Не менять unrelated файлы при точечных правках.
- После изменения Python-файлов запускать `python3 -m py_compile`.
- После изменения shell-скриптов запускать `bash -n`.

## Поведение После Прибытия

После прибытия к candidate:

1. робот поворачивается на месте до 360 градусов;
2. фактический угол считается по TF yaw;
3. live camera проверяется во время поворота;
4. если совпадение найдено, поворот сразу прерывается и робот останавливается;
5. если совпадения нет, пробуется следующий candidate.

## Диагностика

Основные логи:

```text
logs/web/process.log
logs/web/events_YYYYMMDD_HHMMSS.log
```

Проверка TF:

```bash
source /opt/ros/jazzy/setup.bash
ros2 run tf2_ros tf2_echo odom base_link
```
