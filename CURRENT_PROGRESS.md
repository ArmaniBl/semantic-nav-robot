# Текущий Прогресс

## Состояние Runtime

Проект работает в ручном workflow через web-панель:

- web-панель является основной точкой управления;
- Qdrant запускается и останавливается из UI;
- Nav2 запускается в `odom`-frame профиле;
- база наблюдений создается через ручную DB recording-сессию;
- `Go` отправляет выбранную goal-точку и при необходимости пробует до 3 candidates;
- во время движения live camera проверяется по текстовому запросу;
- порог совпадения зависит от количества слов в запросе;
- после прибытия робот поворачивается на месте до 360 градусов;
- при совпадении во время поворота робот сразу останавливается.

## Основные Изменения

- `web_control_panel.py`
  - управление Qdrant, Nav2, DB recording и navigation mission;
  - просмотр и удаление Qdrant collections;
  - проверка `/clock`, `/navigate_to_pose`, TF `goal_frame -> base_link`;
  - сохранение последней выбранной collection в `data/runtime_state.json`.

- `semantic_nav_to_pose.py`
  - поиск visual memory в Qdrant;
  - отправка координат в Nav2;
  - проверка live camera во время движения;
  - динамический порог совпадения по количеству слов;
  - client-side arrival stop по `distance_remaining`;
  - поворот после прибытия с контролем угла по TF yaw;
  - немедленная остановка поворота при совпадении;
  - расшифровка Nav2 error codes.

- `ros_keyframe_recorder.py`
  - запись image + pose metadata;
  - сохранение pose в target frame;
  - подавление штатного `ExternalShutdownException` при остановке.

- `launch/nav2_odom_launch.py`
  - основной launch runtime для текущего workflow.

- `config/nav2_odom_params.yaml`
  - спокойный профиль движения;
  - параметры planner/controller для локальной схемы координат.

## Проверки

```bash
python3 -m py_compile \
  web_control_panel.py \
  ros_keyframe_recorder.py \
  qdrant_load_keyframes.py \
  semantic_nav_to_pose.py \
  semantic_search_qdrant.py \
  semantic_search_offline.py \
  wait_for_slam_ready.py \
  launch/slam_nav2_launch.py \
  launch/localization_nav2_launch.py \
  launch/nav2_odom_launch.py

bash -n restart_semantic_nav.sh stop_all.sh
```

Для launch argument sanity check:

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch launch/nav2_odom_launch.py --show-args
```
