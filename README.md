# Robot Navigation Control

Проект для управления мобильным роботом в ROS 2 Jazzy / Isaac Sim.

Система записывает визуальные keyframes с позой робота, загружает записи в
Qdrant и позволяет отправлять робота к сохраненной позе наблюдения по
текстовому запросу.

## Что Умеет

- Показывает live camera stream в локальной web-панели.
- Запускает и останавливает Qdrant.
- Запускает и останавливает Nav2 в `odom`-frame профиле.
- Записывает keyframes через `Start DB recording` / `Stop DB recording`.
- Загружает записи в Qdrant collection.
- Ищет похожие keyframes по текстовому запросу.
- Отправляет Nav2 цель в координатах найденного keyframe.
- Проверяет live camera во время движения и после прибытия.
- После прибытия к цели поворачивает робота на месте до 360 градусов.
- Если совпадение найдено во время поворота, робот сразу останавливается.
- Управляет Qdrant collections из UI.

## Текущая Схема Координат

Основной runtime работает в `odom`.

Запись сохраняет pose keyframe в `odom`, и `Go` отправляет цель в тот же frame.
Это снижает риск смешивания старой `map` с текущим запуском симуляции.

Практическое правило:

1. Запусти симуляцию.
2. Запусти `Start Map/Nav2`.
3. Запиши DB recording.
4. Используй collection, созданную для текущего запуска.

Если симуляция была перезапущена, лучше сделать новую запись или убедиться, что
робот стартовал в той же позе относительно сохраненной системы координат.

## Структура

```text
web_control_panel.py          Web UI и HTTP API управления runtime
semantic_nav_to_pose.py       Поиск Qdrant по тексту и отправка Nav2 goal
ros_keyframe_recorder.py      Запись RGB keyframes + pose metadata
qdrant_load_keyframes.py      Загрузка записей в Qdrant
semantic_search_qdrant.py     CLI-поиск по Qdrant
semantic_search_offline.py    Offline-поиск по локальным данным
wait_for_slam_ready.py        Gate перед запуском Nav2: ждет /clock, /scan, /map, TF
restart_semantic_nav.sh       Полный restart web + Qdrant + Nav2
stop_all.sh                   Остановка runtime-процессов

launch/nav2_odom_launch.py    Nav2 в odom-frame профиле
launch/slam_nav2_launch.py    SLAM Toolbox + pointcloud_to_laserscan + gated Nav2
launch/localization_nav2_launch.py Saved map + AMCL localization + gated Nav2

config/nav2_odom_params.yaml
config/nav2_slam_params.yaml
config/nav2_localization_params.yaml
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
  - локальное окружение для Qdrant и обработки записей.

Runtime данные Qdrant хранятся в:

```text
qdrant_storage/
```

Эта директория не коммитится.

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
4. Нажми `Start Map/Nav2`.
5. Дождись готовности Nav2.
6. Нажми `Start DB recording`.
7. Провези или поставь робота в местах, которые нужны для базы наблюдений.
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

Последняя выбранная collection сохраняется локально в:

```text
data/runtime_state.json
```

## Как Работает `Go`

`semantic_nav_to_pose.py` выполняет один mission run:

1. Принимает текстовый запрос.
2. Ищет top-k похожих keyframes в выбранной Qdrant collection.
3. Берет первый подходящий candidate.
4. Достает сохраненную pose из payload.
5. Отправляет pose в `goal_frame`, обычно `odom`.
6. Отправляет координаты в Nav2 action `/navigate_to_pose`.
7. Пока Nav2 едет, периодически проверяет live camera.
8. Порог совпадения зависит от количества слов в запросе:
   - 1 слово = `0.25`;
   - 2 слова = `0.26`;
   - дальше порог растет медленнее;
   - максимум = `0.8`.
9. Если совпадение найдено во время движения, Nav2 goal отменяется и робот останавливается.
10. Если Nav2 доехал до goal, робот поворачивается на месте до 360 градусов.
11. Поворот контролируется по TF yaw, а не только по таймеру.
12. Если совпадение найдено во время поворота, робот сразу останавливается.
13. Если совпадения нет, пробуется следующий candidate, максимум 3 попытки.

## Параметры Mission

Основные параметры `semantic_nav_to_pose.py`:

```bash
--collection                       Qdrant collection
--top-k                            сколько кандидатов запросить из Qdrant
--goal-frame                       frame для Nav2 goal, по умолчанию odom
--image-topic                      live camera topic
--complete-on-visual-match         завершать миссию при совпадении live camera
--mission-check-period-sec         период проверки live camera, по умолчанию 1.5
--arrival-spin-angle-deg           угол поворота после прибытия, по умолчанию 360
--arrival-spin-angular-vel-rad-sec скорость поворота после прибытия, по умолчанию 0.45
--max-goal-distance                0 = ехать к точным координатам
--max-candidate-attempts           максимум candidates за один Go, по умолчанию 3
```

Параметр `--mission-match-threshold` оставлен только для совместимости старых
команд. Фактический порог рассчитывается автоматически по текстовому запросу.

## Nav2

Основной режим web-панели запускает `launch/nav2_odom_launch.py`.

Этот профиль:

- использует `odom` как рабочий frame для целей;
- ждет `/clock`, `/scan` и нужный TF;
- использует спокойные параметры движения;
- слушает `/scan` для остановки перед близким препятствием.

Дополнительные launch-файлы оставлены для ручных проверок:

- `launch/slam_nav2_launch.py`;
- `launch/localization_nav2_launch.py`.

## Recording Pipeline

Когда пользователь нажимает `Start DB recording`, web-панель запускает:

```bash
python3 ros_keyframe_recorder.py ...
```

Recorder:

- слушает RGB image topic;
- слушает odometry;
- получает pose в target frame, обычно `odom`;
- сохраняет PNG keyframes;
- пишет `metadata.jsonl`.

После остановки записи web-панель проверяет непрерывность траектории в выбранном
`goal_frame`. Если между соседними keyframes обнаружен скачок больше
`--record-max-map-step-m` (по умолчанию `5.0 m`), запись не загружается в
Qdrant.

В Qdrant payload сохраняются:

- `memory_id`
- `run_id`
- image path/topic/frame
- pose frame/source frame
- pose position/orientation
- map_yaml/map_image
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

Логи web-интерфейса пишутся в:

```text
logs/web/process.log
logs/web/events_YYYYMMDD_HHMMSS.log
```

## Важные Правила Эксплуатации

### Не перезапускать симуляцию под живым Nav2

Если перезапустить Isaac Sim, пока runtime работает:

- `/clock` сбросится или пропадет;
- TF-buffer останется со старым временем;
- могут появиться `Extrapolation Error`;
- старые keyframes могут перестать соответствовать текущей позе робота.

Правильный порядок:

```bash
./stop_all.sh
# перезапустить Isaac Sim
./restart_semantic_nav.sh
```

### Не смешивать старые collections без проверки

Если симуляция или одометрия стартовали иначе, старая collection может вести
робота не туда. Для стабильного теста используй collection, записанную в том же
сеансе.

## Troubleshooting

### Кнопка `Go` неактивна

Проверь `/api/status`:

```bash
curl -fsS http://127.0.0.1:8765/api/status | python3 -m json.tool
```

`Go` включается, когда:

- Qdrant ready;
- `/clock` публикуется;
- `/navigate_to_pose` доступен;
- TF `goal_frame -> base_link` доступен;
- нет активной записи DB;
- нет активной navigation mission.

### `ModuleNotFoundError: rclpy`

Нужно открыть ROS environment:

```bash
source /opt/ros/jazzy/setup.bash
```

### Робот врезается при движении к цели

Runtime слушает `/scan`. Если впереди в секторе 70 градусов есть препятствие
ближе `0.85 m`, mission отменяет Nav2 goal и публикует ноль в `/cmd_vel`.

Порог можно изменить при запуске web-панели:

```bash
python3 web_control_panel.py --front-obstacle-stop-distance-m 1.0
```

### Робот едет визуально не туда

Проверь:

- выбрана ли правильная collection;
- не перезапускалась ли симуляция после записи DB;
- какие координаты отправлены в log строке `Sending keyframe...`;
- текущий TF:

```bash
source /opt/ros/jazzy/setup.bash
ros2 run tf2_ros tf2_echo odom base_link
```

### Робот виляет

Параметры движения находятся в:

```text
config/nav2_odom_params.yaml
config/nav2_slam_params.yaml
```

Если все еще виляет, можно снизить:

```yaml
desired_linear_vel: 0.08
```

## Проверки Перед Коммитом

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

source /opt/ros/jazzy/setup.bash
ros2 launch launch/nav2_odom_launch.py --show-args
```

## Git Hygiene

Не коммитятся:

- виртуальные окружения;
- Qdrant storage;
- recordings/keyframes;
- local map files in `data/maps/`;
- local runtime state in `data/runtime_state.json`;
- Python/ROS build caches;
- local logs.

Это настроено в `.gitignore`.
