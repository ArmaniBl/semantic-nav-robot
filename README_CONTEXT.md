# README_REQUIREMENTS.md

## 1. Название проекта

**Разработка интеллектуального компонента навигации мобильного робота с использованием технологий компьютерного зрения**

---

## 2. Цель проекта

Цель проекта — разработать компонент семантической навигации для мобильного робота, который способен:

1. автономно исследовать среду;
2. сохранять ключевые кадры с привязкой к pose робота;
3. кодировать изображения в семантическое векторное пространство с помощью RuCLIP;
4. хранить визуальную память в Qdrant;
5. принимать русскоязычные текстовые запросы оператора;
6. искать релевантные фрагменты визуальной памяти;
7. отправлять робота к найденной позе наблюдения через Nav2;
8. выполнять визуальную верификацию после прибытия;
9. обновлять статус записи памяти.

---

## 3. Ключевая идея

Робот исследует неизвестную или частично известную среду в NVIDIA Isaac Sim 6.0.

Во время движения он получает:

- RGB-изображения;
- odometry / pose;
- TF;
- при необходимости depth / stereo / IMU.

Система выбирает keyframes, вычисляет image embeddings через RuCLIP и сохраняет их вместе с spatial metadata.

Когда оператор задаёт запрос на русском языке, например:

```text
найди красный гидрант
````

система:

1. кодирует текстовый запрос через RuCLIP;
2. ищет ближайшие image embeddings в Qdrant;
3. фильтрует устаревшие записи;
4. извлекает pose найденного keyframe;
5. формирует navigation goal;
6. отправляет goal в Nav2 через `NavigateToPose`;
7. после прибытия делает новый снимок;
8. повторно сравнивает снимок с запросом;
9. обновляет статус памяти.

---

## 4. Важное ограничение MVP

На уровне MVP система выполняет навигацию к **позе наблюдения**, а не к точной 3D-позиции объекта.

```text
Найденный keyframe pose = место, откуда объект был виден.
```

Это не означает, что система точно вычислила координату объекта.

Для настоящей 3D-локализации объекта потребуются дополнительные компоненты:

* depth camera или stereo camera;
* object detection;
* segmentation;
* camera projection;
* ray casting;
* transform из camera frame в map frame;
* проверка reachable goal pose.

Эти компоненты могут быть добавлены позже, но не входят в базовый MVP, если явно не реализованы.

---

## 5. Основной стек

Текущий основной стек:

| Компонент          | Версия / вариант     | Назначение                        |
| ------------------ | -------------------- | --------------------------------- |
| OS                 | Ubuntu 24.04 LTS     | Основная ОС для ROS2 Jazzy        |
| ROS2               | Jazzy Jalisco        | Основной middleware               |
| Python             | 3.10 для jazzy       | ROS2-ноды и AI-инференс           |
| Симулятор          | NVIDIA Isaac Sim 6.0 | Фотореалистичная симуляция        |
| Навигация          | Nav2 jazzy           | Планирование и движение к цели    |
| ROS client         | rclpy                | Python ROS2-ноды                  |
| Embedding model    | RuCLIP               | Русскоязычный image-text matching |
| Vector DB          | Qdrant               | Хранение embeddings и metadata    |
| DL backend         | PyTorch 2.x          | Инференс RuCLIP                   |
| CV                 | OpenCV, NumPy        | Предобработка изображений         |
| Evaluation         | evo                  | APE/RPE trajectory metrics        |

---

## 6. Совместимость ROS2 Jazzy и Isaac Sim 6.0

Основной режим разработки:

```text
ROS2 jazzy  + Ubuntu 22.04 + Isaac Sim 6.0
```

Правила:

1. Пока проект находится на jazzy, все ROS2-пакеты, launch-файлы и зависимости должны быть совместимы с jazzy.

---

## 7. Основной data flow

```text
                    ┌────────────────────────┐
                    │     Isaac Sim 6.0       │
                    │  RGB / Depth / Odom / TF│
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │     ROS2 Bridge         │
                    └───────────┬────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────┐
        │ ROS2 topics                                │
        │                                            │
        │ /camera/image_raw                          │
        │ /camera/camera_info                        │
        │ /odom                                      │
        │ /tf                                        │
        │ /tf_static                                 │
        └────────────────────┬───────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │ keyframe_selection_node                    │
        │                                            │
        │ - получает image + pose                    │
        │ - применяет критерии отбора                │
        │ - сохраняет keyframe image                 │
        └────────────────────┬───────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │ ruclip_embedding_node                      │
        │                                            │
        │ - загружает RuCLIP                         │
        │ - считает image embeddings                 │
        │ - нормализует embeddings                   │
        └────────────────────┬───────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │ qdrant_memory_node                         │
        │                                            │
        │ - пишет vectors                            │
        │ - пишет metadata payload                   │
        │ - обновляет status                         │
        └────────────────────┬───────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │ semantic_query_node                        │
        │                                            │
        │ - принимает русский текстовый запрос       │
        │ - считает text embedding через RuCLIP      │
        │ - ищет top-k в Qdrant                      │
        │ - фильтрует stale records                  │
        └────────────────────┬───────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │ nav_goal_node                              │
        │                                            │
        │ - выбирает candidate observation pose      │
        │ - проверяет frame_id                       │
        │ - отправляет NavigateToPose goal           │
        └────────────────────┬───────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │ Nav2                                       │
        │                                            │
        │ /navigate_to_pose                          │
        │ nav2_msgs/action/NavigateToPose            │
        └────────────────────┬───────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │ verification_node                          │
        │                                            │
        │ - делает снимок после прибытия             │
        │ - считает RuCLIP similarity                │
        │ - обновляет active/stale/uncertain         │
        └────────────────────────────────────────────┘
```

---

## 8. Предлагаемые ROS2-пакеты проекта

Рекомендуемая структура workspace:

```text
semantic_nav_ws/
  src/
    semantic_nav_interfaces/
    semantic_visual_memory/
    semantic_nav_bringup/
    semantic_nav_experiments/
```

### 8.1 `semantic_nav_interfaces`

Назначение:

* custom services;
* custom messages;
* возможно custom actions, если стандартных Nav2 action/service недостаточно.

Примеры интерфейсов:

```text
SemanticQuery.srv
UpdateMemoryStatus.srv
MemoryRecord.msg
```

### 8.2 `semantic_visual_memory`

Назначение:

* keyframe selection;
* RuCLIP image embedding;
* Qdrant storage;
* semantic text query;
* memory status update.

### 8.3 `semantic_nav_bringup`

Назначение:

* launch-файлы;
* конфиги;
* параметры;
* запуск связки Isaac Sim / ROS2 / Nav2 / memory nodes.

### 8.4 `semantic_nav_experiments`

Назначение:

* robustness experiments;
* Recall@K scripts;
* latency measurement;
* evo trajectory evaluation;
* plotting scripts.

---

## 9. Возможные custom interfaces

### 9.1 `SemanticQuery.srv`

```text
string query
uint32 top_k
---
bool success
string message
geometry_msgs/PoseStamped[] candidate_poses
float32[] scores
string[] memory_ids
```

Назначение:

* принять текстовый запрос;
* вернуть top-k найденных наблюдений;
* вернуть pose, score и memory_id.

---

### 9.2 `UpdateMemoryStatus.srv`

```text
string memory_id
string status
string reason
---
bool success
string message
```

Разрешённые значения `status`:

```text
active
stale
uncertain
```

---

### 9.3 `MemoryRecord.msg`

```text
string memory_id
builtin_interfaces/Time stamp
string image_path
string pose_frame
geometry_msgs/Pose pose
string status
string scene_id
string robot_id
string embedding_model
float32 score
```

---

## 10. Основные ROS2 topics

Ожидаемые входные topics:

| Topic                 | Type                         | Назначение       |
| --------------------- | ---------------------------- | ---------------- |
| `/camera/image_raw`   | `sensor_msgs/msg/Image`      | RGB-поток        |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | Параметры камеры |
| `/odom`               | `nav_msgs/msg/Odometry`      | Одометрия        |
| `/tf`                 | `tf2_msgs/msg/TFMessage`     | Dynamic TF       |
| `/tf_static`          | `tf2_msgs/msg/TFMessage`     | Static TF        |

Возможные внутренние topics:

| Topic                               | Type                           | Назначение                        |
| ----------------------------------- | ------------------------------ | --------------------------------- |
| `/semantic_memory/keyframes`        | custom или стандартный message | События сохранения keyframe       |
| `/semantic_memory/status`           | custom message                 | Статус памяти                     |
| `/semantic_nav/query_result`        | custom message                 | Результат semantic search         |
| `/semantic_nav/verification_result` | custom message                 | Результат проверки после прибытия |

Основной action для навигации:

| Action              | Type                              |
| ------------------- | --------------------------------- |
| `/navigate_to_pose` | `nav2_msgs/action/NavigateToPose` |

---

## 11. TF frames

Базовая TF-цепочка:

```text
map
 └── odom
      └── base_link
           └── camera_link
                └── camera_optical_frame
```

Правила:

1. Nav2 goal должен быть в frame, который понимает Nav2, обычно `map`.
2. Если keyframe pose сохранён в `odom`, перед отправкой goal нужен transform в `map`.
3. Все memory records должны хранить `pose_frame`.
4. Нельзя silently assume, что `odom == map`.
5. При ошибке TF lookup система должна возвращать диагностическое сообщение, а не отправлять некорректный goal.

---

## 12. RuCLIP

### 12.1 Назначение

RuCLIP используется для:

* кодирования изображений;
* кодирования русскоязычных текстовых запросов;
* вычисления semantic similarity между изображением и текстом;
* поиска в визуальной памяти.

### 12.2 Основная модель

Основной вариант:

```text
ai-forever/ru-clip
```

Альтернативы после проверки:

```text
ai-forever/ruclip-vit-base-patch32-384
ai-forever/ruclip-vit-base-patch16-224
```

### 12.3 Правила инференса

Инференс должен выполняться с:

```python
torch.no_grad()
```

или:

```python
torch.inference_mode()
```

Embeddings должны нормализоваться перед cosine similarity:

```python
embedding = embedding / embedding.norm(dim=-1, keepdim=True)
```

При ограничении VRAM 8–10 GB:

```text
batch_size = 1
```

или минимальный batch size, подтверждённый экспериментально.

### 12.4 Ограничения

RuCLIP не является object detector.

RuCLIP не возвращает bounding boxes.

RuCLIP не гарантирует физическое наличие объекта в сцене.

RuCLIP score должен интерпретироваться как semantic similarity, а не как абсолютная вероятность наличия объекта.

---

## 13. Qdrant

### 13.1 Назначение

Qdrant хранит:

* image embeddings;
* metadata keyframes;
* pose;
* статус записи;
* scene/robot/model information.

### 13.2 Payload schema

Минимальный payload:

```json
{
  "memory_id": "uuid",
  "timestamp": 0.0,
  "frame_id": "camera_frame_000001",
  "image_path": "data/keyframes/frame_000001.png",
  "pose_frame": "map",
  "pose": {
    "position": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0
    },
    "orientation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "w": 1.0
    }
  },
  "status": "active",
  "scene_id": "scene_01",
  "robot_id": "robot_01",
  "embedding_model": "ai-forever/ru-clip",
  "similarity_metric": "cosine"
}
```

### 13.3 Status values

```text
active
stale
uncertain
```

### 13.4 Search filtering

По умолчанию semantic search должен исключать:

```text
status == stale
```

Допустимые фильтры:

* `scene_id`;
* `robot_id`;
* `status`;
* `timestamp`;
* `embedding_model`;
* `pose_frame`.

---

## 14. Keyframe selection

Keyframe selection должен учитывать не только время, но и движение робота.

Минимальные параметры:

```yaml
keyframe_selection:
  min_time_delta_sec: 1.0
  min_translation_delta_m: 0.25
  min_rotation_delta_rad: 0.25
  max_blur_threshold: null
  enable_embedding_novelty_check: false
  min_embedding_distance: 0.05
```

Рекомендация:

* на первом этапе использовать time + translation + rotation;
* novelty по embedding добавить позже как экспериментальный режим.

---

## 15. Verification после прибытия

После успешного завершения Nav2 action:

1. получить новый RGB-кадр;
2. вычислить RuCLIP image embedding;
3. сравнить с text embedding исходного запроса;
4. обновить memory status.

Начальные эвристические пороги:

```yaml
verification:
  confirm_threshold: 0.25
  stale_threshold: 0.15
```

Интерпретация:

```text
score >= confirm_threshold:
  object likely present -> active

score < stale_threshold:
  object likely absent -> stale

stale_threshold <= score < confirm_threshold:
  uncertain -> uncertain
```

Важно:

Эти пороги не являются универсальными. Они должны быть откалиброваны на validation scenes.

---

## 16. Nav2 integration

Основной программный интерфейс:

```text
/navigate_to_pose
nav2_msgs/action/NavigateToPose
```

Navigation goal формируется из найденного observation pose.

Перед отправкой goal необходимо проверить:

1. `frame_id`;
2. доступность TF transform;
3. валидность quaternion;
4. достижимость pose с точки зрения costmap;
5. отсутствие stale-статуса у memory record.

Если goal недостижим, система должна:

1. попробовать следующий candidate из top-k;
2. если candidates закончились — вернуть ошибку;
3. не отправлять goal в неизвестный или некорректный frame.

---

## 17. Simulation requirements

Isaac Sim 6.0 используется для:

* фотореалистичной симуляции;
* публикации sensor data в ROS2;
* controlled experiments;
* изменения условий сцены;
* тестирования robustness.

Минимальные условия сцены:

* мобильный робот;
* RGB camera;
* статические объекты;
* несколько целевых объектов для запросов;
* Nav2-compatible environment;
* корректная TF-цепочка;
* odometry source.

Опциональные условия:

* depth camera;
* stereo camera;
* IMU;
* динамические препятствия;
* controlled lighting changes;
* occlusion scenarios;
* blur/noise через post-processing pipeline или проверенный Isaac Sim API.

Если точный API Isaac Sim Replicator для конкретного эффекта не проверен, использовать формулировку:

```text
[REQUIRES VERIFICATION] Требуется проверить API Isaac Sim 6.0 для данного вида domain randomization.
```

---

## 18. Метрики

### 18.1 Semantic retrieval

| Метрика              | Описание                                                           |
| -------------------- | ------------------------------------------------------------------ |
| Precision@1          | Доля случаев, когда top-1 результат соответствует целевому объекту |
| Recall@K             | Доля случаев, когда целевой объект есть в top-k                    |
| Mean Reciprocal Rank | Средняя обратная позиция первого правильного результата            |

### 18.2 Navigation

| Метрика                 | Описание                              |
| ----------------------- | ------------------------------------- |
| Navigation success rate | Доля успешных достижений goal         |
| Time to goal            | Время от отправки goal до результата  |
| Path length             | Длина пройденного пути                |
| Recovery count          | Количество recovery behaviors Nav2    |
| Goal failure rate       | Доля недостижимых или ошибочных goals |

### 18.3 Trajectory

| Метрика   | Инструмент |
| --------- | ---------- |
| APE / ATE | `evo_ape`  |
| RPE       | `evo_rpe`  |

Пример команд:

```bash
evo_ape tum gt.txt est.txt -va --plot
evo_rpe tum gt.txt est.txt -va --plot
```

### 18.4 Verification

| Метрика               | Описание                                                    |
| --------------------- | ----------------------------------------------------------- |
| Verification accuracy | Доля правильных active/stale решений                        |
| False stale rate      | Доля объектов, ошибочно помеченных stale                    |
| False active rate     | Доля отсутствующих объектов, ошибочно подтверждённых active |

### 18.5 Robustness

Проверять зависимость:

```text
Recall@K vs noise_level
Recall@K vs blur_level
Recall@K vs occlusion_level
Recall@K vs lighting_condition
Verification accuracy vs perturbation_level
```

Фейковые результаты запрещены.

---

## 19. Logging requirements

Каждый major event должен логироваться:

* keyframe saved;
* embedding computed;
* Qdrant point inserted;
* query received;
* top-k search result;
* goal sent;
* Nav2 feedback;
* Nav2 result;
* verification score;
* memory status update;
* TF failure;
* Qdrant failure;
* model inference failure.

Логи должны содержать:

* timestamp;
* node name;
* memory_id;
* frame_id;
* query;
* score;
* status;
* error message.

---

## 20. Error handling

Система должна корректно обрабатывать:

* отсутствие изображения;
* отсутствие pose;
* TF lookup failure;
* Qdrant unavailable;
* RuCLIP model load failure;
* CUDA out of memory;
* invalid query;
* empty search result;
* stale-only search result;
* Nav2 goal rejected;
* Nav2 aborted;
* verification timeout.

Нельзя завершать node без понятного diagnostic message.

---

## 21. GPU / performance constraints

Ориентировочное ограничение:

```text
VRAM: 8–10 GB
```

Правила:

* `batch_size=1` по умолчанию;
* `torch.no_grad()` / `torch.inference_mode()`;
* lazy model loading;
* CPU fallback;
* опциональный отдельный process для RuCLIP inference;
* не блокировать ROS2 executor долгим инференсом без async/multithreading решения;
* не сохранять каждый кадр без отбора keyframes.

---

## 22. Конфигурация параметров

Все важные параметры должны быть вынесены в YAML:

```yaml
semantic_memory:
  embedding_model: "ai-forever/ru-clip"
  similarity_metric: "cosine"
  device: "cuda"
  batch_size: 1

keyframe_selection:
  min_time_delta_sec: 1.0
  min_translation_delta_m: 0.25
  min_rotation_delta_rad: 0.25

qdrant:
  host: "localhost"
  port: 6333
  collection_name: "semantic_visual_memory"

navigation:
  action_name: "/navigate_to_pose"
  goal_frame: "map"
  try_top_k_candidates: 3

verification:
  confirm_threshold: 0.25
  stale_threshold: 0.15

simulation:
  use_sim_time: true
```

---

## 23. Минимальные критерии готовности MVP

MVP считается готовым, если:

1. Isaac Sim публикует RGB image и odometry/TF в ROS2.
2. ROS2 node получает image + pose.
3. Keyframe selection сохраняет кадры.
4. RuCLIP создаёт image embeddings.
5. Qdrant хранит embeddings и metadata.
6. Русский текстовый запрос кодируется через RuCLIP.
7. Semantic search возвращает top-k keyframes.
8. Система формирует `NavigateToPose` goal.
9. Nav2 принимает goal.
10. После прибытия выполняется verification.
11. Memory record обновляется как `active`, `stale` или `uncertain`.
12. Есть скрипт оценки Recall@K.
13. Есть логирование latency.
14. Есть инструкция запуска.

---

## 24. Риски

### 24.1 RuCLIP similarity не равна object detection

RuCLIP может найти изображение, семантически похожее на запрос, но не гарантирует наличие объекта.

Митигация:

* validation set;
* calibration thresholds;
* top-k search;
* verification after arrival;
* optional object detector later.

---

### 24.2 Pose keyframe не равна pose объекта

Робот едет туда, откуда объект был виден, а не в точную точку объекта.

Митигация:

* явно писать observation pose;
* выбирать safe approach pose;
* добавить depth/segmentation в future work.

---

### 24.3 TF inconsistency

Ошибки между `map`, `odom`, `base_link` могут сделать memory poses бесполезными.

Митигация:

* хранить `pose_frame`;
* проверять TF перед goal;
* логировать transform failures;
* использовать единый frame для Qdrant payload.

---

### 24.4 Isaac Sim / ROS2 compatibility

Isaac Sim 6.0 поддерживает ROS2 workflows, но конкретные настройки окружения должны быть проверены.

Митигация:

* фиксировать Docker/venv;
* документировать environment variables;
* тестировать bridge отдельно;
* не менять ROS2-дистрибуцию без migration plan.

---

### 24.5 GPU memory

RuCLIP + Isaac Sim + ROS2 могут создавать высокую нагрузку на GPU.

Митигация:

* batch size 1;
* inference_mode;
* CPU fallback;
* раздельные процессы;
* уменьшение частоты keyframes.

---

## 25. Future work

Возможные расширения:

1. Depth-based object localization.
2. Object detector / segmentation model.
3. 3D semantic map.
4. Multi-query memory reasoning.
5. Dynamic object handling.
6. Lifelong memory update.
7. Multi-robot shared semantic memory.
8. FAISS baseline для offline comparison.
9. ONNX/TensorRT оптимизация RuCLIP после проверки совместимости.

---

## 26. Правило документации

Любое новое решение должно быть сопровождено:

1. назначением;
2. местом в архитектуре;
3. ROS2 interfaces;
4. параметрами;
5. инструкцией запуска;
6. способом проверки;
7. известными ограничениями.

Нельзя добавлять код без объяснения, как он интегрируется в общий pipeline.

````

---

## Короткие замечания по выбранному стеку

С учётом **Isaac Sim 6.0** текущий целевой профиль проекта:

```text
current:
Ubuntu 24.04 + ROS2 Jazzy + Python 3.12 + external ROS nodes
````

Конкретный Isaac Sim / ROS2 Bridge workflow, переменные окружения и доступность custom interfaces нужно проверять отдельно, особенно если Isaac Sim использует внутренние библиотеки или запускается из собственного Python environment.

Чтобы включить рос2 окружение: source /opt/ros/jazzy/setup.bash