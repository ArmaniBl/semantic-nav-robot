# Текущий прогресс по разработке проекта

## 2026-04-25

- Прочитаны `Agents.md` и `README_CONTEXT.md`.
- Текущая сцена Isaac Sim / Nova Carter уже публикует базовые ROS2-топики: stereo RGB, `camera_info`, IMU, lidar, `/chassis/odom`, `/tf`, `/cmd_vel`.
- Следующий практический этап выбран как нижний слой MVP: запись keyframes из `/front_stereo_camera/left/image_raw` с позой наблюдения из `/chassis/odom`.
- Добавлен скрипт `ros_keyframe_recorder.py`.
  - Входы: `/front_stereo_camera/left/image_raw`, `/chassis/odom`.
  - Выходы: PNG-кадры в `data/keyframes/images/` и JSONL metadata в `data/keyframes/metadata.jsonl`.
  - Отбор keyframes: минимальное время, смещение, поворот; сохраняется `pose_frame`, `image_frame`, timestamp и pose.
  - Для image и odom используется best-effort QoS, чтобы лучше совпадать с Isaac Sim sensor publishers.
- Синтаксис `ros_keyframe_recorder.py` проверен через `python3 -m py_compile`.
- Исправлена нумерация в `ros_keyframe_recorder.py`: при новом запуске recorder продолжает с максимального существующего `keyframe_*.png` и не перетирает старые кадры.
- Собрана первая рабочая пачка визуальной памяти:
  - `43` PNG-кадра в `data/keyframes/images/`;
  - `43` JSONL-записи в `data/keyframes/metadata.jsonl`;
  - дубликатов `memory_id` нет;
  - все `image_path` существуют;
  - `pose_frame = odom`;
  - `image_frame = front_stereo_camera_left_optical`;
  - `pose_age_sec` в диапазоне примерно `0.0167..0.0333`;
  - траектория по odom: примерно `x=0.002..26.981`, `y=-0.000..0.797`.
- Добавлен offline-слой semantic memory:
  - `ruclip_embed_keyframes.py` считает RuCLIP image embeddings для всех keyframes из `metadata.jsonl`;
  - результат сохраняется в `data/keyframes/embeddings.jsonl`;
  - каждая запись содержит `memory_id`, `image_path`, `embedding_model`, `vector_dim`, `vector` и `payload` с pose/status metadata;
  - `semantic_search_offline.py` считает RuCLIP text embedding для русского запроса и выводит top-k ближайших keyframes по cosine similarity без Qdrant.
- Создано локальное окружение `.ruclip_venv` для запуска RuCLIP-скриптов, потому что текущий системный `python3` не видел пакет `ruclip`, а старое `.venv` содержит shebang на несуществующий путь `/home/arm/diplom/.venv/bin/python`.
- Установлен `ruclip` в `.ruclip_venv`; для сборки зависимости `youtokentome` понадобились `Cython` и `--no-build-isolation`.
- Выполнен расчёт embeddings:

```bash
.ruclip_venv/bin/python ruclip_embed_keyframes.py
```

  - создан `data/keyframes/embeddings.jsonl`;
  - обработано `43` keyframes;
  - устройство инференса: `cuda`;
  - размерность vector: `512`;
  - нормы сохранённых vectors: `1.000000`, то есть embeddings нормализованы для cosine similarity.
- Проверен offline semantic search:

```bash
.ruclip_venv/bin/python semantic_search_offline.py "дорога" --top-k 5
.ruclip_venv/bin/python semantic_search_offline.py "стена" --top-k 5
```

  - запрос `"дорога"` вернул top-1 `keyframe_000033` со score `0.1912`;
  - запрос `"стена"` вернул top-1 `keyframe_000042` со score `0.1559`;
  - результаты содержат `memory_id`, score, `pose_frame=odom`, координаты pose и путь к изображению.

## Ближайший следующий шаг

Визуально проверить top-k картинки для нескольких русских запросов и решить, достаточно ли осмысленное ранжирование. Если да, следующий слой MVP — загрузка vectors + payload в Qdrant collection и перенос offline search на Qdrant search API.

## 2026-04-26

- Прочитаны текущие Markdown-файлы проекта: `CURRENT_PROGRESS.md`, `README_CONTEXT.md`, `Agents.md`.
- Зафиксировано уточнение: текущая целевая ROS2-дистрибуция проекта — Jazzy.
- Аккуратно установлен Qdrant через Docker Compose:
  - добавлен `compose.qdrant.yml`;
  - используется pinned image `qdrant/qdrant:v1.17.1-unprivileged`;
  - REST/gRPC порты проброшены только на localhost: `127.0.0.1:6333`, `127.0.0.1:6334`;
  - включены лимиты `cpus: 1.0`, `mem_limit: 1g`;
  - автоперезапуск отключён: `restart: "no"`;
  - telemetry отключена через `QDRANT__TELEMETRY_DISABLED=true`;
  - storage вынесен в `qdrant_storage/`.
- Добавлен `.gitignore` для локальных окружений, cache и `qdrant_storage/`.
- Установлен `qdrant-client==1.17.1` в существующее окружение `.ruclip_venv`.
- Проверки:
  - Qdrant отвечает на `http://127.0.0.1:6333/`;
  - лог содержит `Telemetry reporting disabled`;
  - Python-клиент успешно получает список коллекций: `collections=[]`;
  - контейнер в простое потребляет примерно `13 MiB / 1 GiB`;
  - текущий `qdrant_storage/` занимает примерно `16K`;
  - Docker image `qdrant/qdrant:v1.17.1-unprivileged` занимает примерно `286MB`.

## Ближайший следующий шаг после Qdrant install

Создать скрипт загрузки `data/keyframes/embeddings.jsonl` в Qdrant collection с vector size `512`, distance `Cosine`, payload из существующих записей, затем перенести semantic search с offline cosine на Qdrant search/query API.

## 2026-04-26 — Qdrant visual memory layer

- Добавлен `qdrant_load_keyframes.py`.
  - Читает `data/keyframes/embeddings.jsonl`.
  - Создаёт/проверяет collection `semantic_visual_memory`.
  - Vector config: size `512`, distance `Cosine`.
  - Загружает points через deterministic UUID, построенный из `memory_id`.
  - Оставляет исходный `memory_id` в payload.
  - Создаёт payload indexes для `memory_id`, `status`, `embedding_model`, `pose_frame`.
- Добавлен `semantic_search_qdrant.py`.
  - Кодирует русский text query через тот же RuCLIP model/cache, что offline baseline.
  - Выполняет `QdrantClient.query_points`.
  - По умолчанию исключает `status == stale`.
  - Поддерживает фильтры `--status`, `--embedding-model`, `--pose-frame`.
- Выполнена загрузка embeddings:

```bash
.ruclip_venv/bin/python qdrant_load_keyframes.py --recreate
```

  - загружено `43/43` points;
  - collection `semantic_visual_memory`;
  - `points_count = 43`;
  - collection status: `green`;
  - payload schema keys: `embedding_model`, `memory_id`, `pose_frame`, `status`.
- Проверен Qdrant semantic search:

```bash
.ruclip_venv/bin/python semantic_search_qdrant.py "дорога" --top-k 5
.ruclip_venv/bin/python semantic_search_qdrant.py "стена" --top-k 5
```

  - запрос `"дорога"` вернул top-1 `keyframe_000033` со score `0.1912`;
  - запрос `"стена"` вернул top-1 `keyframe_000042` со score `0.1559`;
  - top-1 совпадает с предыдущим offline semantic search.

## Ближайший следующий шаг

Сделать визуальную проверку top-k результатов уже из Qdrant для нескольких русских запросов, затем оформить Qdrant-поиск как слой/ноду semantic query для дальнейшей интеграции с Nav2. Отдельно нужно решить архитектурный вопрос `pose_frame=odom` vs Nav2 goal frame `map`.

## 2026-04-26 — Nav2 integration start

- Уточнено пользователем: текущая ROS2-дистрибуция проекта — Jazzy.
- Обновлён Markdown-контекст:
  - `Agents.md` больше не описывает Humble как текущий профиль;
  - `README_CONTEXT.md` обновлён под текущий профиль `Ubuntu 24.04 + ROS2 Jazzy + Python 3.12`;
  - `CURRENT_PROGRESS.md` фиксирует Jazzy как целевую дистрибуцию.
- Добавлен standalone ROS2 Jazzy script/node `semantic_nav_to_pose.py`.
  - Вход: русский текстовый query CLI argument.
  - Кодирует query через RuCLIP.
  - Ищет top-k candidates в Qdrant collection `semantic_visual_memory`.
  - Берёт observation pose из payload.
  - Проверяет quaternion.
  - Если `pose_frame != goal_frame`, пытается выполнить TF lookup через `tf2_ros.Buffer`.
  - Отправляет первый валидный candidate в Nav2 action `/navigate_to_pose`.
  - Action type: `nav2_msgs/action/NavigateToPose`.
  - Поддерживает `--dry-run`, `--goal-frame`, `--top-k`, `--include-stale`, `--no-wait-result`.
  - По умолчанию `--goal-frame map`, чтобы не смешивать `odom` и `map` молча.
- Проверено окружение:
  - `/opt/ros/jazzy/setup.bash` существует;
  - после `source /opt/ros/jazzy/setup.bash` окружение `.ruclip_venv` видит `rclpy`, `nav2_msgs`, `geometry_msgs`, `tf2_ros`, `action_msgs`;
  - `semantic_nav_to_pose.py` проходит `py_compile` в sourced Jazzy окружении.
- Проверен безопасный dry-run для текущих данных в `odom`:

```bash
source /opt/ros/jazzy/setup.bash
.ruclip_venv/bin/python semantic_nav_to_pose.py "дорога" --top-k 1 --goal-frame odom --dry-run
```

  - найден `keyframe_000033`;
  - score `0.1912`;
  - dry-run goal: `frame=odom`, `x=16.925`, `y=0.405`.
- Проверен failure mode для дефолтного `map` без live TF:

```bash
.ruclip_venv/bin/python semantic_nav_to_pose.py "дорога" --top-k 1 --goal-frame map --dry-run
```

  - candidate пропущен с ошибкой TF lookup: frame `map` не существует;
  - goal не отправлен.

## Ближайший следующий шаг

Запустить Isaac Sim + Nav2 Jazzy так, чтобы были доступны `/tf`, `map -> odom` и action server `/navigate_to_pose`, затем выполнить `semantic_nav_to_pose.py` без `--dry-run`. Для MVP без глобальной карты можно временно проверять `--goal-frame odom`, но для корректной Nav2-интеграции нужен согласованный `map` frame.

## 2026-04-26 — Live Nav2 smoke-test with Isaac Sim

- Пользователь запустил Isaac Sim scene.
- Live ROS2 graph:
  - есть `/chassis/odom`, `/tf`, `/cmd_vel`, RGB/lidar/IMU topics;
  - `/tf` публикует `odom -> base_link`;
  - `/tf_static` не публикуется;
  - static transforms к sensor frames, например `base_link -> front_3d_lidar`, сейчас недоступны;
  - до запуска bringup `/navigate_to_pose` отсутствовал.
- Проверено, что Nav2 Jazzy packages уже установлены:
  - `nav2_bringup`, `nav2_controller`, `nav2_planner`, `nav2_bt_navigator`, `nav2_behaviors`, `nav2_lifecycle_manager`, `slam_toolbox`, `robot_localization`.
- Добавлен временный MVP bringup:
  - `config/nav2_odom_params.yaml`;
  - `launch/nav2_odom_launch.py`.
- Назначение временного bringup:
  - odom-frame Nav2 smoke-test без карты;
  - `global_frame=odom`;
  - `robot_base_frame=base_link`;
  - `odom_topic=/chassis/odom`;
  - controller output remapped to `/cmd_vel`;
  - costmaps без obstacle layers, потому что sensor static TF пока отсутствует.
- Nav2 bringup успешно стартует:

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch launch/nav2_odom_launch.py
```

  - lifecycle nodes переходят в `active`;
  - появляется `/navigate_to_pose [nav2_msgs/action/NavigateToPose]`;
  - `/cmd_vel` получает publishers от Nav2 и subscriber от Isaac differential drive.
- Выполнен live semantic goal:

```bash
.ruclip_venv/bin/python semantic_nav_to_pose.py "дорога" --top-k 1 --goal-frame odom
```

  - Qdrant top-1: `keyframe_000033`, score `0.1912`;
  - Nav2 принял goal;
  - robot начал движение от примерно `x=1.12` к `x=16.92`;
  - дошёл примерно до `x=2.71`;
  - затем Nav2 получил `Failed to make progress` и начал recovery.
- Добавлены улучшения в `semantic_nav_to_pose.py`:
  - `TransformListener(..., spin_thread=True)`, чтобы TF lookup работал до action spin;
  - `--max-goal-distance` для безопасных коротких live smoke-tests;
  - `--feedback-log-period-sec` для throttling feedback logs;
  - `--robot-base-frame`.
- Проверен короткий live smoke-test:

```bash
.ruclip_venv/bin/python semantic_nav_to_pose.py "дорога" \
  --top-k 1 \
  --goal-frame odom \
  --max-goal-distance 2.0 \
  --feedback-log-period-sec 2.0
```

  - goal был ограничен с `14.220m` до `2.000m`;
  - Nav2 принял goal;
  - Nav2 публиковал `/cmd_vel`, например `linear.x=0.45`;
  - odom не продвигался вперёд, distance remaining оставался примерно `2.050`.
- Проверена граница проблемы прямой публикацией `/cmd_vel` без Nav2:
  - forward command `linear.x=0.3` почти не изменил odom;
  - reverse command `linear.x=-0.3` сдвинул робота назад примерно с `x=2.71` до `x=2.51`;
  - вывод: ROS/Nav2/Qdrant pipeline до `/cmd_vel` работает, текущий блокер — состояние симуляции/контакт/препятствие/drive behavior в Isaac scene при движении вперёд из текущей позы.
- Все запущенные в ходе проверки Nav2/semantic client процессы остановлены.

## Ближайший следующий шаг

В Isaac Sim вернуть робота в свободную стартовую позу или проверить, не упёрся ли он в геометрию/коллизию около `x≈2.7`, затем повторить короткий smoke-test. После этого добавить static transforms для sensor frames и включать obstacle layer / полноценный `map` или SLAM-based Nav2 profile.

## 2026-04-26 — Lidar obstacle layer and successful short semantic navigation

- Подтверждено пользователем: робот упёрся в препятствие, потому что предыдущий временный профиль вёл его только вперёд.
- Обновлён `launch/nav2_odom_launch.py`:
  - добавлен `tf2_ros/static_transform_publisher`;
  - публикует приблизительный static TF `base_link -> front_3d_lidar`;
  - параметры доступны через launch args: `lidar_x`, `lidar_y`, `lidar_z`, `lidar_roll`, `lidar_pitch`, `lidar_yaw`;
  - дефолт: `x=0.40`, `y=0.00`, `z=0.45`, identity rotation.
- Обновлён `config/nav2_odom_params.yaml`:
  - local/global costmaps теперь используют `obstacle_layer` + `inflation_layer`;
  - obstacle source: `/front_3d_lidar/lidar_points`;
  - data type: `PointCloud2`.
- Проверки:
  - `base_link -> front_3d_lidar` TF появился;
  - local/global costmaps подписались на `front_lidar`;
  - lifecycle nodes перешли в `active`;
  - missing-transform ошибок для lidar при запуске не было.
- Выполнен успешный короткий semantic navigation smoke-test:

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch launch/nav2_odom_launch.py

.ruclip_venv/bin/python semantic_nav_to_pose.py "дорога" \
  --top-k 1 \
  --goal-frame odom \
  --max-goal-distance 2.0 \
  --feedback-log-period-sec 2.0
```

  - Qdrant top-1: `keyframe_000033`, score `0.1912`;
  - goal был ограничен с `16.797m` до `2.000m`;
  - Nav2 принял goal;
  - distance remaining уменьшался примерно `1.858 -> 0.358`;
  - Nav2 сообщил `Reached the goal`;
  - semantic client сообщил `Navigation succeeded for keyframe_000033`.
- Исправлен shutdown `semantic_nav_to_pose.py`:
  - добавлен `close()` для остановки background TF listener executor/thread;
  - dry-run после правки завершается без traceback.
- Временный Nav2 launch остановлен после проверки.

## Ближайший следующий шаг

Подстроить точный static TF `base_link -> front_3d_lidar` по Isaac/Nova Carter модели, затем пробовать более длинный semantic goal без `--max-goal-distance` или с постепенным увеличением лимита. Для полноценного MVP далее нужен `map`/SLAM profile вместо odom-only smoke-test.

## 2026-04-26 — SLAM mapping profile for ROS2 Jazzy

- Добавлен SLAM/Nav2 bringup:
  - `launch/slam_nav2_launch.py`;
  - `config/slam_toolbox_params.yaml`;
  - `config/nav2_slam_params.yaml`.
- Профиль запускает:
  - static TF `base_link -> front_3d_lidar`;
  - `pointcloud_to_laserscan` из `/front_3d_lidar/lidar_points` в `/scan`;
  - `slam_toolbox` в mapping mode с lifecycle configure/activate;
  - Nav2 controller/planner/behavior/bt navigator в `map` frame.
- Важная правка для Isaac lidar:
  - `scan_angle_min=-pi`, `scan_angle_max=pi`;
  - фронтальные `[-90°, +90°]` давали пустой `/scan` после разворота робота, потому что точки уходили в задне-боковой сектор.
- Для начального картографирования с места ослаблены пороги `slam_toolbox`:
  - `minimum_time_interval: 0.05`;
  - `minimum_travel_distance: 0.02`;
  - `minimum_travel_heading: 0.05`.
- Для режима growing-map отключён `static_layer` в `global_costmap`:
  - `/map` продолжает публиковаться SLAM;
  - Nav2 global costmap работает как rolling window по текущим `/scan` obstacles;
  - это убрало ошибки `Received map message is malformed` на самых первых узких/пустых сообщениях карты.
- Проверено live в Isaac:

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch launch/slam_nav2_launch.py
```

  - `/scan` публикуется с углами примерно `[-3.14159, 3.14159]`;
  - `/map`, `/map_metadata`, `map -> base_link` доступны;
  - `/navigate_to_pose` доступен;
  - `slam_toolbox`, `controller_server`, `planner_server`, `behavior_server`, `bt_navigator` переведены в `active`.
- После короткого вращения на месте:

```bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.30}}"
```

  - карта выросла примерно до `6.1 x 9.3 m`;
  - `slam_toolbox/graph_visualization` показал 5 узлов;
  - `/map` содержал свободные, неизвестные и занятые клетки.
- Текущая карта сохранена через `nav2_map_server map_saver_cli`:
  - `data/maps/slam_2026-04-26.yaml`;
  - `data/maps/slam_2026-04-26.pgm`.

## Ближайший следующий шаг

Запускать semantic navigation уже поверх SLAM-профиля:

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch launch/slam_nav2_launch.py

.ruclip_venv/bin/python semantic_nav_to_pose.py "дорога" \
  --top-k 1 \
  --goal-frame map \
  --max-goal-distance 2.0 \
  --feedback-log-period-sec 2.0
```

Для полноценного самостоятельного исследования всей сцены следующим отдельным шагом нужен frontier/exploration node; текущий профиль уже строит карту по мере вращения и движения робота.

## 2026-05-04 — Semantic navigation over SLAM map frame

- Прочитаны текущие Markdown-файлы проекта: `CURRENT_PROGRESS.md`, `README_CONTEXT.md`, `Agents.md`.
- Подтверждено текущее live-состояние:
  - Isaac Sim публикует `/chassis/odom`, `/tf`, `/cmd_vel`, `/front_3d_lidar/lidar_points`, RGB topics и `/clock`;
  - Qdrant container `semantic_nav_qdrant` работает;
  - до запуска bringup `/navigate_to_pose` отсутствовал.
- Запущен SLAM/Nav2 профиль:

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch launch/slam_nav2_launch.py
```

  - `/navigate_to_pose` доступен;
  - `/scan` публикуется с углами примерно `[-3.14159, 3.14159]`;
  - TF `map -> base_link` появился;
  - Nav2 lifecycle nodes перешли в `active`.
- Выполнен успешный semantic navigation smoke-test уже в `map` frame:

```bash
.ruclip_venv/bin/python semantic_nav_to_pose.py "дорога" \
  --top-k 1 \
  --goal-frame map \
  --max-goal-distance 2.0 \
  --feedback-log-period-sec 2.0
```

  - Qdrant top-1: `keyframe_000033`, score `0.1912`;
  - source pose был в `odom`, goal отправлен в `map`;
  - goal был ограничен примерно с `16.863m` до `2.000m`;
  - Nav2 принял goal и сообщил `Navigation succeeded for keyframe_000033`.
- Попытка увеличить smoke-test до `--max-goal-distance 4.0` сначала выявила ограничение временного профиля:
  - planner error: `Goal Coordinates ... was outside bounds`;
  - причина: `global_costmap` rolling window `10 x 10 m` оказался слишком мал для более длинного плана около границы окна.
- Обновлён `config/nav2_slam_params.yaml`:
  - `global_costmap.width: 20`;
  - `global_costmap.height: 20`.
- После перезапуска SLAM/Nav2 `--max-goal-distance 4.0` успешно завершился:

```bash
.ruclip_venv/bin/python semantic_nav_to_pose.py "дорога" \
  --top-k 1 \
  --goal-frame map \
  --max-goal-distance 4.0 \
  --feedback-log-period-sec 3.0
```

  - Qdrant top-1: `keyframe_000033`, score `0.1912`;
  - goal был ограничен примерно с `17.063m` до `4.000m`;
  - Nav2 принял goal;
  - во время движения было `2` recovery;
  - distance remaining уменьшился до `0.413`;
  - Nav2 сообщил `Navigation succeeded for keyframe_000033`.
- Сохранена текущая выросшая SLAM-карта:
  - `data/maps/slam_2026-05-04_map_goal_4m.yaml`;
  - `data/maps/slam_2026-05-04_map_goal_4m.pgm`;
  - размер карты при сохранении: `470 x 387` cells, `0.05 m/pix`.

## Ближайший следующий шаг

Продолжить постепенное увеличение `--max-goal-distance` для `semantic_nav_to_pose.py` в `map` frame и параллельно улучшать exploration/mapping. Для перехода от smoke-test к полноценному MVP нужен exploration/frontier node или заранее сохранённая карта + localization profile, чтобы semantic goals не зависели от случайно выросшего rolling-map состояния.

## 2026-05-05 — Frontier exploration layer

- Прочитаны текущие `CURRENT_PROGRESS.md`, `README_CONTEXT.md`, `Agents.md`, `semantic_nav_to_pose.py`, SLAM/Nav2 launch и configs.
- Добавлен standalone ROS2 Jazzy script/node `frontier_explorer.py`.
  - Подписывается на `/map` (`nav_msgs/msg/OccupancyGrid`) с transient-local QoS.
  - Ищет frontier cells: свободная клетка рядом с неизвестной.
  - Кластеризует frontier cells 8-связностью.
  - Выбирает goal cell около центроида кластера.
  - Ранжирует candidates по размеру frontier-кластера и расстоянию до робота.
  - Отправляет goal в Nav2 action `/navigate_to_pose`.
  - Поддерживает `--dry-run`, `--once`, `--max-goal-distance`, `--min-frontier-size`, `--max-goals`, timeout/replan параметры.
  - Для live SLAM startup добавлен fallback `--allow-latest-tf-compose`: если прямой lookup `map -> base_link` не имеет timestamp overlap, pose робота берётся композицией latest `map -> odom` и `odom -> base_link`.
- Обновлён `launch/slam_nav2_launch.py`:
  - добавлен optional launch arg `run_explorer:=false`;
  - при `run_explorer:=true` запускает локальный `frontier_explorer.py`;
  - добавлены launch args `explorer_max_goal_distance`, `explorer_min_frontier_size`, `explorer_goal_timeout_sec`.
- Проверки:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m py_compile frontier_explorer.py launch/slam_nav2_launch.py
ros2 launch launch/slam_nav2_launch.py --show-args
```

  - синтаксис проходит;
  - новые launch args отображаются;
  - default-поведение SLAM/Nav2 launch не меняется, потому что `run_explorer=false`.
- Live dry-run с запущенным Isaac Sim + SLAM/Nav2:

```bash
python3 frontier_explorer.py \
  --dry-run \
  --once \
  --map-timeout-sec 10 \
  --initial-tf-timeout-sec 10 \
  --tf-timeout-sec 3 \
  --max-goal-distance 4.0
```

  - найдено `802` frontier cells;
  - найдено `3` usable clusters;
  - лучший goal: примерно `x=2.150`, `y=-1.434`;
  - расстояние до goal: примерно `2.05 m`;
  - размер лучшего кластера: `681`.
- Выполнен первый настоящий frontier exploration goal:

```bash
python3 frontier_explorer.py \
  --once \
  --map-timeout-sec 10 \
  --initial-tf-timeout-sec 10 \
  --tf-timeout-sec 3 \
  --max-goal-distance 4.0 \
  --goal-timeout-sec 90 \
  --feedback-log-period-sec 3
```

  - Nav2 принял goal `frame=map`, `x=2.150`, `y=-1.434`;
  - distance remaining уменьшался примерно `1.885 -> 0.385`;
  - recoveries: `0`;
  - Nav2 сообщил `Goal succeeded`;
  - `frontier_explorer.py` сообщил `Frontier goal reached`.
- Сохранена карта после первого frontier шага:
  - `data/maps/slam_2026-05-05_frontier_once.yaml`;
  - `data/maps/slam_2026-05-05_frontier_once.pgm`;
  - размер карты при сохранении: `282 x 273` cells, `0.05 m/pix`.
- Временный SLAM/Nav2 launch остановлен; после остановки live ROS graph снова содержит только Isaac Sim topics без `/map` и `/navigate_to_pose`.

## Ближайший следующий шаг

Запустить несколько sequential frontier goals через `frontier_explorer.py --max-goals N` или `ros2 launch launch/slam_nav2_launch.py run_explorer:=true`, собрать более полную карту, затем повторить semantic navigation к RuCLIP/Qdrant keyframes уже на выросшей карте. После этого стоит добавить связку exploration + keyframe recorder, чтобы visual memory пополнялась во время автономного обхода.

## 2026-05-05 — Sequential frontier exploration and semantic smoke-test

- Подтверждено live-состояние перед стартом:
  - Isaac Sim публикует sensor topics, `/chassis/odom`, `/tf`, `/clock`, `/cmd_vel`;
  - Qdrant container `semantic_nav_qdrant` работает;
  - до запуска bringup `/map` и `/navigate_to_pose` отсутствовали.
- Запущен SLAM/Nav2 профиль:

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch launch/slam_nav2_launch.py
```

  - `slam_toolbox`, Nav2 lifecycle nodes и `/navigate_to_pose` активировались.
- Выполнена успешная серия из `3` sequential frontier goals:

```bash
python3 frontier_explorer.py \
  --max-goals 3 \
  --map-timeout-sec 15 \
  --initial-tf-timeout-sec 30 \
  --tf-timeout-sec 3 \
  --max-goal-distance 4.0 \
  --min-frontier-size 8 \
  --goal-timeout-sec 110 \
  --feedback-log-period-sec 4
```

  - Goal #1: `x=-1.228`, `y=0.325`, reached, recoveries `0`.
  - Goal #2: `x=-0.577`, `y=-1.628`, reached, recoveries `0`.
  - Goal #3: `x=-0.527`, `y=-3.267`, reached, recoveries `0`.
  - Frontier map grew during the run:
    - before goal #1: `110` frontier cells, `2` clusters;
    - before goal #2: `967` frontier cells, `3` clusters;
    - before goal #3: `1557` frontier cells, `6` clusters.
- Saved the map after the successful `3`-goal exploration run:

```bash
ros2 run nav2_map_server map_saver_cli \
  -t /map \
  -f data/maps/slam_2026-05-05_frontier_3goals \
  --fmt pgm
```

  - `data/maps/slam_2026-05-05_frontier_3goals.yaml`;
  - `data/maps/slam_2026-05-05_frontier_3goals.pgm`;
  - saved map size: `314 x 387` cells, `0.05 m/pix`.
- Repeated semantic navigation over the grown SLAM map:

```bash
.ruclip_venv/bin/python semantic_nav_to_pose.py "дорога" \
  --top-k 1 \
  --goal-frame map \
  --max-goal-distance 4.0 \
  --feedback-log-period-sec 4.0
```

  - Qdrant top-1 remained `keyframe_000033`, score `0.1912`;
  - source pose was in `odom`, goal was transformed to `map`;
  - goal was limited from about `18.045 m` to `4.000 m`;
  - Nav2 accepted goal `x=2.908`, `y=-1.622`;
  - Nav2 initially had a planner recovery and then reduced distance remaining down to about `0.389 m`;
  - action did not return success before the external shell `timeout 180`;
  - after client termination, Nav2 kept executing the action until SLAM/Nav2 launch was stopped;
  - launch shutdown cancelled the still-running Nav2 goal.
- Improvements added after this semantic smoke-test:
  - `semantic_nav_to_pose.py` now supports `--result-timeout-sec` and requests goal cancel if no Nav2 result arrives in time;
  - `semantic_nav_to_pose.py` now orients a smoke-test-limited goal toward the intermediate target instead of inheriting the far keyframe orientation;
  - `semantic_nav_to_pose.py` handles shutdown more cleanly when the ROS context is already shutting down;
  - `config/nav2_slam_params.yaml` `general_goal_checker.xy_goal_tolerance` increased from `0.35` to `0.45`, because MVP semantic navigation targets approximate observation poses.
- Verification after code changes:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m py_compile semantic_nav_to_pose.py frontier_explorer.py launch/slam_nav2_launch.py
.ruclip_venv/bin/python semantic_nav_to_pose.py --help
```

  - syntax checks pass;
  - `--result-timeout-sec` appears in CLI help.
- A second live retry after these code changes was attempted, but Isaac/ROS time jumped backwards during frontier navigation:
  - logs showed `Detected jump back in time. Clearing TF buffer`;
  - repeated `TF_OLD_DATA` warnings appeared for `base_link` and `odom`;
  - `frontier_explorer.py` timed out/cancelled the first stuck candidate, tried the next candidate, then Nav2 failed with `error_code=102`;
  - after the clock/TF jump, `frontier_explorer.py` found `0` usable clusters;
  - SLAM/Nav2 launch was stopped.
- Final live graph after stopping bringup again contained only Isaac Sim topics; `/map` and `/navigate_to_pose` were gone.

## Ближайший следующий шаг

Reset/restart Isaac Sim cleanly before the next live run, then rerun:

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch launch/slam_nav2_launch.py
python3 frontier_explorer.py --max-goals 3 --max-goal-distance 4.0
.ruclip_venv/bin/python semantic_nav_to_pose.py "дорога" \
  --top-k 1 \
  --goal-frame map \
  --max-goal-distance 4.0 \
  --result-timeout-sec 120 \
  --feedback-log-period-sec 4.0
```

The key thing to verify next is whether the updated semantic client now either succeeds within the wider goal tolerance or cancels cleanly without leaving Nav2 executing after timeout.

## 2026-05-05 — Web control panel MVP

- Decided to add a standalone browser UI layer for operating the current MVP:
  - live robot camera view;
  - text input for Russian semantic navigation command;
  - execution log for semantic navigation subprocess;
  - stop button for the active navigation client.
- Added `web_control_panel.py`.
  - Uses only Python stdlib HTTP server plus ROS2/rclpy/OpenCV dependencies already present in the project environment.
  - Subscribes to `/front_stereo_camera/left/image_raw` with best-effort QoS.
  - Serves MJPEG camera stream at `/camera.mjpg`.
  - Serves control UI at `/`.
  - Provides `/api/status`, `/api/logs`, `/api/events`, `/api/navigate`, `/api/stop`.
  - Starts existing `.ruclip_venv/bin/python semantic_nav_to_pose.py ...` as a subprocess for text commands.
  - Streams subprocess output into the web log.
- Improved `semantic_nav_to_pose.py` for web-driven operation:
  - stores the active Nav2 goal handle;
  - on `KeyboardInterrupt`/web stop, requests Nav2 goal cancel before shutdown;
  - this avoids leaving Nav2 executing if the UI stops the semantic client.
- Verification:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m py_compile web_control_panel.py semantic_nav_to_pose.py
python3 web_control_panel.py --help
python3 web_control_panel.py --host 127.0.0.1 --port 8765
curl -sS http://127.0.0.1:8765/api/status
curl -sS http://127.0.0.1:8765/ | sed -n '1,30p'
timeout 5 curl -sS http://127.0.0.1:8765/camera.mjpg | head -c 64 | xxd
```

  - syntax checks pass;
  - web UI HTML is served;
  - `/api/status` reported live camera frames:
    - `has_frame=true`;
    - `image_frame=front_stereo_camera_left_optical`;
    - frame age around `0.16 sec` during the check;
  - `/camera.mjpg` returned a multipart JPEG stream.
- The web panel is currently running locally at:

```text
http://127.0.0.1:8765/
```

## Ближайший следующий шаг

With Isaac Sim, Qdrant, and SLAM/Nav2 running, open the web panel and send a query such as `дорога`. Verify:

1. the browser log shows RuCLIP/Qdrant/Nav2 output;
2. the robot moves through `/navigate_to_pose`;
3. `Stop active goal` cancels the semantic client and Nav2 goal cleanly;
4. if this works, add web buttons for starting/stopping SLAM/Nav2 and frontier exploration.

## 2026-05-05 — Web panel port change

- Changed the default `web_control_panel.py` port from `8088` to `8765`, because `8088` is often occupied in the user's working environment.
- Updated `CURRENT_PROGRESS.md` web panel examples to use:

```text
http://127.0.0.1:8765/
```

- Restarted the currently running panel on the new default port without passing `--port`.
- Verification:
  - `python3 -m py_compile web_control_panel.py semantic_nav_to_pose.py` passes;
  - `http://127.0.0.1:8765/api/status` returns live camera status;
  - `127.0.0.1:8088` is no longer serving the panel.

## 2026-05-05 — Web panel log layout fix

- Fixed the web panel layout issue where long logs expanded the page and pushed the camera view downward.
- Updated `web_control_panel.py` CSS:
  - `html`, `body`, `main`, and `aside` now constrain desktop layout to viewport height;
  - grid/flex containers use `min-height: 0` where needed;
  - the log area scrolls internally instead of growing the whole page;
  - mobile layout keeps normal page scrolling.
- Restarted the running panel on the default port `8765`.
- Verification:
  - `python3 -m py_compile web_control_panel.py` passes;
  - `http://127.0.0.1:8765/` serves the updated CSS;
  - `http://127.0.0.1:8765/api/status` still returns live camera status.

## 2026-05-05 — Web panel SLAM/Nav2 readiness guard

- User tried web query `yellow`; web panel launched `semantic_nav_to_pose.py`, RuCLIP/Qdrant returned `keyframe_000007`, but the robot did not move.
- Root cause from log:
  - `map` frame did not exist;
  - live ROS graph had Isaac Sim topics only, but no `/map` and no `/navigate_to_pose`;
  - TF lookup failed: `map -> base_link` missing.
- Updated `web_control_panel.py`:
  - tracks readiness of `/map`;
  - tracks readiness of `/navigate_to_pose` through action status topic `/navigate_to_pose/_action/status`;
  - tracks TF readiness for `map -> base_link`;
  - `/api/status` now includes a `system` object with `map_topic`, `navigate_action`, `goal_frame`, `ready`, and `summary`;
  - UI shows `SLAM/Nav2` and `TF map` status cards;
  - `Go` is disabled until SLAM/Nav2 is ready;
  - `/api/navigate` now rejects requests with HTTP `409` before launching RuCLIP if SLAM/Nav2 is not ready;
  - noisy polling logs for `GET /api/status`, `/api/events`, and `/camera.mjpg` are filtered from the execution log.
- Fixed a preflight deadlock in `/api/navigate` caused by nested state locking.
- Verification with current Isaac-only graph:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m py_compile web_control_panel.py semantic_nav_to_pose.py
curl -sS http://127.0.0.1:8765/api/status
curl -sS -X POST http://127.0.0.1:8765/api/navigate \
  -H 'Content-Type: application/json' \
  -d '{"query":"yellow"}' \
  -w '\n%{http_code}\n'
```

  - `/api/status` reports:
    - `map_topic=false`;
    - `navigate_action=false`;
    - `goal_frame=false`;
    - `ready=false`;
    - `summary="Missing: /map, /navigate_to_pose, map->base_link"`.
  - `/api/navigate` returns `409` with:

```text
SLAM/Nav2 is not ready. Start `ros2 launch launch/slam_nav2_launch.py` first. Missing: /map, /navigate_to_pose, map->base_link
```

- Web panel is running again on:

```text
http://127.0.0.1:8765/
```

## Ближайший следующий шаг

Start SLAM/Nav2 before sending a web command:

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch launch/slam_nav2_launch.py
```

Wait until the web panel `SLAM/Nav2` card says `ready`, then send a query from the browser. If the readiness card stays missing after bringup, inspect `/map`, `/navigate_to_pose/_action/status`, and `tf2_echo map base_link`.

## 2026-05-11 — Web panel starts SLAM/Nav2

- User asked to fix the web panel so SLAM/Nav2 can be started from the page instead of manually from a terminal.
- Updated `web_control_panel.py`:
  - added backend endpoints:
    - `POST /api/slam/start`;
    - `POST /api/slam/stop`;
  - `POST /api/slam/start` launches:

```bash
ros2 launch /home/arman/test/diplom/launch/slam_nav2_launch.py log_level:=info
```

  - `POST /api/slam/stop` sends SIGINT to the launch process group;
  - if active semantic navigation is running, stop SLAM/Nav2 first stops the active semantic client;
  - SLAM/Nav2 stdout/stderr are streamed into the web log with source `slam`;
  - `/api/status` now includes `slam.running`, `slam.runtime_sec`, `slam.last_exit_code`;
  - UI now has buttons:
    - `Start SLAM/Nav2`;
    - `Stop SLAM/Nav2`;
  - UI now shows `SLAM process` and `SLAM runtime` cards.
- Verification:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m py_compile web_control_panel.py semantic_nav_to_pose.py
python3 web_control_panel.py --help
curl -sS http://127.0.0.1:8765/api/status
curl -sS -X POST http://127.0.0.1:8765/api/slam/start
ros2 action list -t
```

  - syntax check passes;
  - web UI serves the new buttons;
  - `POST /api/slam/start` returned `{"ok": true, "started": true}`;
  - after startup `/api/status` reported:
    - `slam.running=true`;
    - `system.ready=true`;
    - `summary="SLAM/Nav2 ready"`;
  - ROS action list included `/navigate_to_pose [nav2_msgs/action/NavigateToPose]`.
- Current state after this change:
  - web panel is running at `http://127.0.0.1:8765/`;
  - SLAM/Nav2 was started through the web backend and is currently ready for a semantic query.

## Ближайший следующий шаг

From the browser:

1. Open `http://127.0.0.1:8765/`.
2. Confirm `SLAM/Nav2` shows `ready`.
3. Enter a query, for example `дорога` or `yellow`.
4. Press `Go`.
5. Verify the log shows RuCLIP/Qdrant/Nav2 output and the robot receives `/navigate_to_pose`.

## 2026-05-11 — Web panel starts and checks Qdrant

- User reported web navigation failure:
  - RuCLIP encoded query successfully;
  - Qdrant search failed with `Connection refused` to `127.0.0.1:6333`;
  - semantic client exited with code `1`.
- Root cause:
  - Qdrant container `semantic_nav_qdrant` was not running;
  - `curl http://127.0.0.1:6333/` failed with connection refused.
- Updated `web_control_panel.py`:
  - added Qdrant readiness check against `--qdrant-url` default `http://127.0.0.1:6333`;
  - `/api/status` now includes `qdrant.ready`, `qdrant.summary`, `qdrant.url`;
  - `Go` is disabled unless both Qdrant and SLAM/Nav2 are ready;
  - added backend endpoints:
    - `POST /api/qdrant/start`;
    - `POST /api/qdrant/stop`;
  - `POST /api/qdrant/start` runs:

```bash
docker compose -f /home/arman/test/diplom/compose.qdrant.yml up -d
```

  - `POST /api/qdrant/stop` runs:

```bash
docker compose -f /home/arman/test/diplom/compose.qdrant.yml down
```

  - UI now has buttons:
    - `Start Qdrant`;
    - `Stop Qdrant`;
  - UI now shows `Qdrant` and `Qdrant URL` status cards;
  - Qdrant command output is streamed into the web log with source `qdrant`.
- Verification:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m py_compile web_control_panel.py semantic_nav_to_pose.py
curl -sS http://127.0.0.1:8765/api/status
curl -sS -X POST http://127.0.0.1:8765/api/qdrant/start
curl -sS http://127.0.0.1:6333/
curl -sS -X POST http://127.0.0.1:8765/api/slam/start
ros2 action list -t
```

  - Qdrant start returned `{"ok": true, "started": true}`;
  - Docker showed `semantic_nav_qdrant` running with localhost ports `6333-6334`;
  - Qdrant root endpoint returned version `1.17.1`;
  - before SLAM/Nav2 readiness, `/api/navigate` correctly returned `409` for missing `/map`, `/navigate_to_pose`, `map->base_link`;
  - after starting SLAM/Nav2 through the web panel, `/api/status` reported:
    - `qdrant.ready=true`;
    - `slam.running=true`;
    - `system.ready=true`;
    - `summary="SLAM/Nav2 ready"`;
  - ROS action list included `/navigate_to_pose [nav2_msgs/action/NavigateToPose]`.
- Current live state:
  - web panel running at `http://127.0.0.1:8765/`;
  - Qdrant is running;
  - SLAM/Nav2 is running and ready;
  - semantic query can now be launched from the web panel with `Go`.

## Ближайший следующий шаг

Use the browser panel for an end-to-end semantic navigation test:

1. Confirm `Qdrant` is `ready`.
2. Confirm `SLAM/Nav2` is `ready`.
3. Enter `дорога` or another query.
4. Press `Go`.
5. If navigation succeeds, add optional web controls for frontier exploration and map saving.

## 2026-05-11 — Web exploration mission button

- Added `exploration_mission.py`:
  - saves the robot start pose from TF `map -> base_link`;
  - starts `ros_keyframe_recorder.py` before movement;
  - runs `frontier_explorer.py` for a bounded exploration mission;
  - sends a Nav2 return goal back to the saved start pose;
  - stops recording after the return attempt;
  - embeds the newly recorded keyframes with `ruclip_embed_keyframes.py`;
  - reloads Qdrant with `qdrant_load_keyframes.py --recreate`, so the visual memory DB is rebuilt from the new exploration run.
- Updated `web_control_panel.py`:
  - added `POST /api/explore/start`;
  - added `POST /api/explore/stop`;
  - added web UI buttons `Explore` and `Stop Explore`;
  - added `Explore` status and runtime cards;
  - semantic `Go`, Qdrant controls, and SLAM/Nav2 controls are disabled while exploration is active;
  - exploration logs stream into the panel with source `explore`.
- Default mission behavior:
  - `--explore-max-goals 3`;
  - `--explore-max-goal-distance 4.0`;
  - `--explore-goal-timeout-sec 110.0`;
  - `--explore-return-timeout-sec 120.0`;
  - run artifacts go into `data/exploration_runs/run_YYYYMMDD_HHMMSS/`.
- Verification:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m py_compile web_control_panel.py exploration_mission.py semantic_nav_to_pose.py frontier_explorer.py ros_keyframe_recorder.py ruclip_embed_keyframes.py qdrant_load_keyframes.py
```

## Ближайший следующий шаг

Restart/open the web panel at `http://127.0.0.1:8765/`, confirm `Qdrant` and `SLAM/Nav2` are `ready`, then press `Explore` to run a short autonomous exploration and rebuild the semantic visual memory DB.

## 2026-05-11 — Exploration web-panel stress test and fixes

- User reported that during exploration camera frames became slower and then web buttons stopped responding.
- Reproduced the panel-side failure:
  - `web_control_panel.py` process was alive;
  - `curl http://127.0.0.1:8765/api/status` timed out;
  - `ss` showed stale `CLOSE-WAIT` connections on port `8765`;
  - the default HTTP listen backlog was only `5`;
  - long-lived `/camera.mjpg` and `/api/events` connections had no client socket timeout.
- Fixed `web_control_panel.py`:
  - added client socket timeout for HTTP handlers;
  - expanded `ThreadingHTTPServer.request_queue_size` to `128`;
  - catch socket timeout/OSError in MJPEG and SSE stream handlers;
  - reduced web camera encode load:
    - `--stream-width` default `640`;
    - `--jpeg-quality` default `70`;
    - `--max-stream-fps` default `3.0`.
- Reproduced and fixed exploration behavior issues:
  - first run failed because `frontier_explorer.py` started with an empty TF buffer and timed out waiting for `map -> base_link`;
  - web exploration now passes `--initial-tf-timeout-sec 45.0`;
  - `exploration_mission.py` now uses `use_sim_time` by default;
  - `ros_keyframe_recorder.py` no longer prints an `rcl_shutdown already called` traceback on normal SIGINT;
  - `exploration_mission.py` throttles return-goal feedback logs using `--feedback-log-period-sec`;
  - `frontier_explorer.py` now enforces `--max-goals` before sending each candidate goal, not only after a full candidate round.
- Shortened the future web Explore mission defaults:
  - `--explore-max-goals 1`;
  - `--explore-goal-timeout-sec 60.0`;
  - this should make the button feel like a bounded "drive for a bit, record, return, rebuild DB" action instead of a long multi-goal expedition.
- Manual test result:
  - `POST /api/explore/start` worked;
  - `/api/status` stayed responsive during exploration;
  - camera `frame_age_sec` stayed fresh, typically below `0.25s`;
  - keyframes were recorded into `data/exploration_runs/run_20260511_223329/`;
  - 58 keyframes were embedded;
  - Qdrant collection `semantic_visual_memory` was recreated with 58 points;
  - mission exit code was `6` because Nav2 failed to fully return to the saved start pose before `--return-timeout-sec 120.0`.
- Current live state after restart:
  - web panel running at `http://127.0.0.1:8765/`;
  - Qdrant ready;
  - SLAM/Nav2 ready;
  - Explore is idle;
  - Qdrant DB currently has 58 points from the latest exploration run.

## Ближайший следующий шаг

Run one more Explore from the web page with the new shorter defaults. If return still times out, tune return behavior separately: either increase `--explore-return-timeout-sec`, choose closer frontier goals, or return to an odom/local checkpoint instead of the original map pose.

## 2026-05-11 — Shutdown and launch instructions

- Stopped all project-side runtime processes:
  - web control panel on `127.0.0.1:8765`;
  - `slam_nav2_launch.py` ROS2 launch processes;
  - exploration/recorder/frontier/navigation helper processes;
  - Qdrant Docker container via `docker compose -f compose.qdrant.yml down`.
- Verified:
  - `http://127.0.0.1:8765/api/status` is not reachable;
  - `http://127.0.0.1:6333/` is not reachable;
  - no matching runtime process remains.

### Launch From Clean State

Use one terminal from the project root:

```bash
cd /home/arman/test/diplom
source /opt/ros/jazzy/setup.bash
docker compose -f compose.qdrant.yml up -d
python3 web_control_panel.py --host 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8765/
```

From the web page:

1. Wait until `Qdrant` is `ready`.
2. Click `Start SLAM/Nav2`.
3. Wait until `SLAM/Nav2` is `ready`.
4. Use `Explore` to run a short exploration and rebuild Qdrant from new keyframes.
5. Use the semantic command input and `Go` for navigation queries.

### Optional Manual Checks

```bash
curl http://127.0.0.1:6333/
curl http://127.0.0.1:8765/api/status
ros2 action list -t | grep navigate_to_pose
```

### Stop Everything Again

```bash
pkill -f web_control_panel.py || true
pkill -f exploration_mission.py || true
pkill -f frontier_explorer.py || true
pkill -f ros_keyframe_recorder.py || true
pkill -f semantic_nav_to_pose.py || true
pkill -f slam_nav2_launch.py || true
docker compose -f /home/arman/test/diplom/compose.qdrant.yml down
```
