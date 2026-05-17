# Текущий Прогресс

## Состояние Runtime

Проект переведен на ручной workflow semantic navigation:

- autonomous frontier exploration удален из runtime;
- web-панель является основной точкой управления;
- semantic memory создается через ручную DB recording-сессию;
- Nav2 стартует только после готовности SLAM `/map` и TF;
- `Go` отправляет одну выбранную semantic goal-точку и не переключается на другие candidates по умолчанию;
- во время движения live camera проверяется RuCLIP’ом, и миссия может завершиться раньше координатной цели, если визуальный запрос совпал.

## Основные Изменения

- `web_control_panel.py`
  - управление Qdrant, SLAM/Nav2, DB recording, semantic navigation;
  - просмотр и удаление Qdrant collections;
  - проверка `/clock`, `/map`, `/navigate_to_pose`, TF `map -> base_link`;
  - защита от использования старой collection до текущего SLAM-сеанса.

- `semantic_nav_to_pose.py`
  - поиск visual memory в Qdrant;
  - отправка точных координат в Nav2;
  - live visual mission completion по камере;
  - расшифровка Nav2 error codes;
  - один selected candidate на mission run по умолчанию.

- `ros_keyframe_recorder.py`
  - запись image + pose metadata;
  - сохранение pose в target frame;
  - подавление штатного `ExternalShutdownException` при остановке.

- `launch/slam_nav2_launch.py`
  - запуск `wait_for_slam_ready.py` перед Nav2;
  - Nav2 не стартует, пока нет `/clock`, `/scan`, `/map`, TF.

- `config/nav2_slam_params.yaml`
  - более спокойный профиль Regulated Pure Pursuit;
  - большая rolling global costmap;
  - увеличенный lifecycle `bond_timeout`.

- `obstacle_right_turn.py`
  - простой независимый тест: ехать вперед и поворачивать вправо на 90 градусов при препятствии.

## Очистка

Удалены из runtime:

- `frontier_explorer.py`
- `exploration_mission.py`
- старые tracked maps в `data/maps/`

Локальный мусор удален из рабочей директории:

- `__pycache__/`
- `launch/__pycache__/`
- Windows `Zone.Identifier`
- неиспользуемый untracked `example_scene.usd`

## Проверки

```bash
python3 -m py_compile \
  web_control_panel.py \
  ros_keyframe_recorder.py \
  ruclip_embed_keyframes.py \
  qdrant_load_keyframes.py \
  semantic_nav_to_pose.py \
  semantic_search_qdrant.py \
  semantic_search_offline.py \
  obstacle_right_turn.py \
  wait_for_slam_ready.py \
  launch/slam_nav2_launch.py \
  launch/nav2_odom_launch.py

bash -n restart_semantic_nav.sh stop_all.sh
```

Для launch argument sanity check:

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch launch/slam_nav2_launch.py --show-args
```
