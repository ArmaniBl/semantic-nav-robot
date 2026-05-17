# Semantic Navigation Robot Control

Проект для семантической навигации робота в ROS 2 Jazzy / Isaac Sim.

Система строит SLAM-карту, записывает визуальные keyframes с позой робота,
считает RuCLIP-эмбеддинги, загружает их в Qdrant и затем позволяет ехать по
текстовому запросу: например `pumpkin`, `door`, `boxes`, `дорога`.

## Что Умеет

- Показывает live camera stream в локальной web-панели.
- Запускает и останавливает Qdrant.
- Запускает и останавливает SLAM Toolbox + Nav2.
- Записывает semantic memory через `Start DB recording` / `Stop DB recording`.
- После записи автоматически считает RuCLIP embeddings и загружает их в Qdrant.
- Ищет похожие visual keyframes по тексту.
- Отправляет Nav2 цель в координатах найденного keyframe.
- Пока робот едет к координатам, проверяет live-camera RuCLIP score; если текущий кадр совпал с запросом, goal отменяется и миссия считается выполненной.
- Управляет Qdrant collections из UI.

## Важная Модель Координат

`map` в SLAM не является абсолютной мировой системой Isaac Sim.

При новом запуске SLAM координаты `map` строятся заново. Поэтому visual memory,
записанная в одном SLAM-сеансе, может вести робота не туда в другом SLAM-сеансе.

Практическое правило:

1. Запусти симуляцию.
2. Запусти SLAM/Nav2.
3. Запиши DB recording.
4. Используй именно collection, созданную в этом же SLAM-сеансе.

Если симуляция или SLAM были перезапущены, лучше записать новую DB.

## Структура

```text
web_control_panel.py          Web UI и HTTP API управления runtime
semantic_nav_to_pose.py       Поиск Qdrant по тексту и отправка Nav2 goal
ros_keyframe_recorder.py      Запись RGB keyframes + pose metadata
ruclip_embed_keyframes.py     Расчет RuCLIP image embeddings
qdrant_load_keyframes.py      Загрузка embeddings в Qdrant
semantic_search_qdrant.py     CLI semantic search по Qdrant
semantic_search_offline.py    Offline search по локальным embeddings
obstacle_right_turn.py        Простой тестовый reactive controller: ехать/повернуть
wait_for_slam_ready.py        Gate перед запуском Nav2: ждет /clock, /scan, /map, TF
restart_semantic_nav.sh       Полный restart web + Qdrant + SLAM/Nav2
stop_all.sh                   Остановка runtime-процессов

launch/slam_nav2_launch.py    SLAM Toolbox + pointcloud_to_laserscan + gated Nav2
launch/nav2_odom_launch.py    Nav2 в odom-frame профиле

config/slam_toolbox_params.yaml
config/nav2_slam_params.yaml
config/nav2_odom_params.yaml
compose.qdrant.yml
```

## Требования

- Ubuntu/Linux с ROS 2 Jazzy.
- Isaac Sim или другой источник ROS topics:
  - `/clock`
  - `/front_stereo_camera/left/image_raw`
  - `/front_3d_lidar/lidar_points`
  - `/chassis/odom`
- Docker Compose для Qdrant.
- Python окружение проекта:
  - системный `python3` с ROS пакетами для web/recorder/launch helper;
  - `.ruclip_venv` для RuCLIP/Qdrant semantic scripts.

Модель RuCLIP кэшируется в:

```text
.cache/ruclip/
```

Runtime данные Qdrant хранятся в:

```text
qdrant_storage/
```

Эти директории не коммитятся.

## Быстрый Старт

Из корня проекта:

```bash
cd /home/arman/test/diplom
source /opt/ros/jazzy/setup.bash
./restart_semantic_nav.sh
```

Открыть:

```text
http://127.0.0.1:8765/
```

Если нужен доступ с другой машины:

```bash
HOST=0.0.0.0 ./restart_semantic_nav.sh
```

Потом открыть адрес VM, например:

```text
http://<VM-IP>:8765/
```

## Рабочий Сценарий Через Web

1. Запусти Isaac Sim и дождись публикации `/clock`.
2. Открой web-панель.
3. Убедись, что Qdrant `ready`.
4. Нажми `Start SLAM/Nav2`.
5. Дождись `SLAM/Nav2 ready`.
6. Нажми `Start DB recording`.
7. Провези или поставь робота в местах, которые нужны для semantic memory.
8. Нажми `Stop DB recording`.
9. Дождись загрузки записи в Qdrant.
10. Введи текстовый запрос.
11. Нажми `Go`.

Записи сохраняются в:

```text
data/recordings/run_YYYYMMDD_HHMMSS/keyframes/
```

Каждая запись создает Qdrant collection:

```text
semantic_visual_memory_run_YYYYMMDD_HHMMSS
```

## Как Работает `Go`

`semantic_nav_to_pose.py` выполняет один mission run:

1. Кодирует текстовый запрос RuCLIP text encoder.
2. Ищет top-k похожих image embeddings в выбранной Qdrant collection.
3. Берет первый semantic candidate.
4. Достает сохраненную pose из payload.
5. Преобразует pose в `goal_frame`, обычно `map`.
6. Отправляет точные координаты в Nav2 action `/navigate_to_pose`.
7. Пока Nav2 едет, периодически проверяет live camera:
   - кадр кодируется RuCLIP image encoder;
   - считается similarity с исходным текстом;
   - если score >= `--mission-match-threshold`, Nav2 goal отменяется и миссия считается успешной.
8. Если Nav2 доехал до goal, миссия считается успешной.
9. Если выбранный goal не сработал, по умолчанию mission run завершается ошибкой и не прыгает на другие keyframes.

Это важно: один `Go` теперь означает одну выбранную цель, без автоматического переключения на следующий semantic candidate.

Для возврата старого поведения можно вручную запускать:

```bash
./.ruclip_venv/bin/python semantic_nav_to_pose.py "pumpkin" --try-next-candidates
```

## Параметры Semantic Mission

Основные параметры `semantic_nav_to_pose.py`:

```bash
--collection                      Qdrant collection
--top-k                           сколько кандидатов запросить из Qdrant
--goal-frame                      frame для Nav2 goal, по умолчанию map
--image-topic                     live camera topic
--complete-on-visual-match        завершать миссию при совпадении live camera
--mission-match-threshold         порог совпадения, по умолчанию 0.30
--mission-check-period-sec        период проверки live camera, по умолчанию 1.5
--max-goal-distance               0 = ехать к точным координатам
--try-next-candidates             пробовать следующие keyframes при ошибке Nav2
```

Web-панель запускает semantic navigation через `.ruclip_venv/bin/python`.

## SLAM/Nav2 Bringup

`launch/slam_nav2_launch.py` запускает:

1. `static_transform_publisher` для `base_link -> front_3d_lidar`;
2. `pointcloud_to_laserscan`, который делает `/scan` из `/front_3d_lidar/lidar_points`;
3. `slam_toolbox`;
4. `wait_for_slam_ready.py`;
5. Nav2 nodes только после готовности SLAM.

`wait_for_slam_ready.py` ждет:

- `/clock` publisher;
- `/scan` publisher;
- `/map` publisher;
- TF `map -> base_link`.

Это предотвращает ранний старт Nav2, когда `map` еще не существует.

## Nav2 Профиль

В `config/nav2_slam_params.yaml` включен спокойный профиль движения:

- сниженная линейная скорость;
- увеличенный lookahead;
- отключен velocity-scaled lookahead;
- уменьшено угловое ускорение;
- увеличен `bond_timeout`, чтобы lifecycle manager не гасил Nav2 из-за короткой просадки heartbeat;
- global costmap сделана большой rolling window, чтобы planner мог принимать дальние точные координаты.

## Recording Pipeline

Когда пользователь нажимает `Start DB recording`, web-панель запускает:

```bash
python3 ros_keyframe_recorder.py ...
```

Recorder:

- слушает RGB image topic;
- слушает odometry;
- получает pose в target frame, обычно `map`;
- сохраняет PNG keyframes;
- пишет `metadata.jsonl`.

После `Stop DB recording` web-панель запускает:

```bash
.ruclip_venv/bin/python ruclip_embed_keyframes.py ...
.ruclip_venv/bin/python qdrant_load_keyframes.py ...
```

В Qdrant payload сохраняются:

- `memory_id`
- `run_id`
- image path/topic/frame
- pose frame/source frame
- pose position/orientation
- embedding model
- status

## Qdrant

Запуск:

```bash
docker compose -f compose.qdrant.yml up -d
```

Остановка:

```bash
docker compose -f compose.qdrant.yml down
```

Web-панель может:

- показать collections;
- выбрать active collection;
- удалить collection;
- показать images/poses из выбранной collection.

## Управляющие Скрипты

Полный restart runtime:

```bash
./restart_semantic_nav.sh
```

Остановка всего runtime:

```bash
./stop_all.sh
```

Ручной запуск web-панели:

```bash
source /opt/ros/jazzy/setup.bash
python3 web_control_panel.py --host 127.0.0.1 --port 8765
```

## Reactive Obstacle Test

`obstacle_right_turn.py` — отдельный простой тестовый скрипт, не часть semantic mission.

Поведение:

- ждет `/scan`;
- едет вперед;
- если видит препятствие ближе `--stop-distance-m`, поворачивает вправо на `--turn-angle-deg`, по умолчанию 90 градусов;
- если скан протух, останавливается.

Пример:

```bash
source /opt/ros/jazzy/setup.bash
./obstacle_right_turn.py --stop-distance-m 2.0 --linear-speed 0.25 --angular-speed 0.4
```

## Важные Правила Эксплуатации

### Не перезапускать симуляцию под живым SLAM/Nav2

Если перезапустить Isaac Sim, пока SLAM/Nav2 работают:

- `/clock` сбросится или пропадет;
- TF-buffer останется со старым временем;
- могут появиться `Extrapolation Error`;
- `map -> base_link` может стать неконсистентным;
- старые keyframes могут перестать соответствовать текущей карте.

Правильный порядок:

```bash
./stop_all.sh
# перезапустить Isaac Sim
./restart_semantic_nav.sh
```

### Не смешивать collections между SLAM-сеансами

Если SLAM был перезапущен, новая `map` может иметь другой ноль и поворот.
Используй collection, записанную после текущего запуска SLAM.

## Troubleshooting

### Кнопка `Go` неактивна

Проверь `/api/status`:

```bash
curl -fsS http://127.0.0.1:8765/api/status | python3 -m json.tool
```

`Go` включается, когда:

- Qdrant ready;
- `/clock` публикуется;
- `/map` публикуется;
- `/navigate_to_pose` доступен;
- TF `map -> base_link` доступен;
- нет активной записи DB;
- нет активной navigation mission.

### `ModuleNotFoundError: rclpy`

Нужно открыть ROS environment:

```bash
source /opt/ros/jazzy/setup.bash
```

### `GOAL_OUTSIDE_MAP`

Nav2 planner считает, что цель вне global costmap. В текущем профиле global costmap 80x80 rolling window. Если цель дальше, нужно:

- записать актуальную DB ближе к текущему SLAM-сеансу;
- или увеличить global costmap;
- или включить `--max-goal-distance`, если нужна промежуточная цель.

### Робот едет визуально не туда

Проверь:

- выбрана ли collection текущего SLAM-сеанса;
- не перезапускалась ли симуляция после записи DB;
- какие координаты отправлены в log строке `Sending keyframe...`;
- текущий TF:

```bash
source /opt/ros/jazzy/setup.bash
ros2 run tf2_ros tf2_echo map base_link
```

### Робот виляет

Параметры движения находятся в:

```text
config/nav2_slam_params.yaml
config/nav2_odom_params.yaml
```

Уже выставлен спокойный профиль. Если все еще виляет, можно снизить:

```yaml
desired_linear_vel: 0.08
```

## Проверки Перед Коммитом

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

source /opt/ros/jazzy/setup.bash
ros2 launch launch/slam_nav2_launch.py --show-args
```

## Git Hygiene

Не коммитятся:

- виртуальные окружения;
- RuCLIP model cache;
- Qdrant storage;
- recordings/keyframes/embeddings;
- Python/ROS build caches;
- local logs.

Это настроено в `.gitignore`.
