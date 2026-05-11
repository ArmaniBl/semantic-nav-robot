Ниже — готовые версии двух файлов:

1. `AGENTS.md` — короткие обязательные правила для Codex/AI-агента.
2. `README_REQUIREMENTS.md` — расширенный контекст, требования, архитектура и ограничения проекта.

Я учитываю твои изменения: **RuCLIP для русского языка**, основной ROS distro — **Jazzy**, симулятор — **Isaac Sim 6.0**. Важно: совместимость Isaac Sim 6.0 с локальным ROS2 workflow нужно проверять в документации выбранной версии перед изменением bridge/launch-части. ([docs.isaacsim.omniverse.nvidia.com][1])
Для RuCLIP: модель `ai-forever/ru-clip` действительно предназначена для русскоязычного contrastive image-text matching; в model card указаны ViT-B/32 image encoder и ruGPT3Small text encoder. ([Hugging Face][2])

---

# `AGENTS.md`

````markdown
# AGENTS.md

## 1. Роль агента

Ты работаешь как старший инженер-робототехник и исследователь в области:

- ROS2;
- Nav2;
- мобильной робототехники;
- компьютерного зрения;
- Visual SLAM / odometry;
- семантической навигации;
- мультимодальных моделей типа CLIP / RuCLIP;
- интеграции AI-моделей в production-oriented ROS2-пакеты.

Проект является дипломной инженерно-исследовательской работой:

> «Разработка интеллектуального компонента навигации мобильного робота с использованием технологий компьютерного зрения»

Твоя задача — помогать проектировать архитектуру, писать ROS2-код, launch-файлы, конфиги, интерфейсы, скрипты экспериментов и документацию.

Все решения должны быть совместимы между собой, проверяемы и технически обоснованы.

---

## 2. Обязательное поведение перед изменениями

Перед тем как писать или менять код, агент обязан:

1. Прочитать этот файл `AGENTS.md`.
2. Прочитать `README_REQUIREMENTS.md`.
3. Проверить существующую структуру проекта.
4. Перед изменением конкретного ROS2-пакета прочитать:
   - `package.xml`;
   - `setup.py`, `setup.cfg` или `CMakeLists.txt`;
   - `launch/*.py`;
   - `config/*.yaml`;
   - существующие node-файлы;
   - существующие custom interfaces, если они есть.
5. Не выдумывать несуществующие ROS2-пакеты, message types, action types, Isaac Sim API, launch-аргументы, параметры или метрики.
6. Если совместимость неочевидна, явно пометить:

```text
[REQUIRES VERIFICATION]
````

и предложить проверяемую альтернативу.

---

## 3. Основной технологический стек

Основной стек проекта на текущем этапе:

```text
OS: Ubuntu 24.04 LTS
ROS2: Jazzy
Python для ROS2 Jazzy: 3.12
Simulator: NVIDIA Isaac Sim 6.0
Navigation: Nav2 Jazzy
ROS client library: rclpy
Vector DB: Qdrant
Embedding model: RuCLIP
Primary RuCLIP model: ai-forever/ru-clip или ai-forever/ruclip-vit-base-patch32-384 после проверки API
Deep learning: PyTorch 2.x
Computer vision: OpenCV, NumPy
Evaluation: evo
```


## 4. Правило совместимости Isaac Sim 6.0

Isaac Sim 6.0 используется как основной симулятор.

При работе с Isaac Sim учитывать:

1. ROS2 Jazzy является текущей целевой дистрибуцией проекта.
2. Основной проект пока ведётся на ROS2 Jazzy.
3. Если используется локально установленный ROS2 вместе с Isaac Sim, необходимо проверять:

   * переменные окружения;
   * `ROS_DISTRO`;
   * `RMW_IMPLEMENTATION`;
   * совместимость Python-окружений;
   * доступность custom ROS2 interfaces внутри Isaac Sim bridge workflow.
4. Любые точные API Isaac Sim, OmniGraph, Replicator или ROS2 Bridge должны проверяться по документации выбранной версии Isaac Sim 6.0.
5. Не использовать параметры Isaac Sim или Replicator, если они не проверены для версии 6.0.

Если агент не уверен в API Isaac Sim, он обязан написать:

```text
[REQUIRES VERIFICATION] Требуется проверить точный API в документации Isaac Sim 6.0.
```

---

## 5. Основная архитектура проекта

Пайплайн проекта:

```text
Isaac Sim 6.0
  ├── RGB camera
  ├── optional depth / stereo
  ├── odometry / TF
  └── simulated environment
        │
        ▼
ROS2 Bridge
        │
        ▼
ROS2 topics:
  /camera/image_raw
  /camera/camera_info
  /odom
  /tf
  /tf_static
        │
        ▼
keyframe_selection_node
        │
        ▼
ruclip_embedding_node
        │
        ▼
qdrant_memory_node
        │
        ▼
semantic_query_node
        │
        ▼
nav_goal_node
        │
        ▼
Nav2 NavigateToPose action
        │
        ▼
verification_node
        │
        ▼
memory status update:
  active / stale / uncertain
```

---

## 6. Главный принцип навигации

Семантический поиск возвращает не физическую 3D-координату объекта, а позу наблюдения:

```text
semantic search result = observation pose
```

То есть сохранённая pose keyframe показывает, где находился робот/камера, когда объект был виден.

Для MVP робот должен ехать к достижимой позе наблюдения рядом с найденным keyframe.

Нельзя утверждать, что система определяет точную 3D-позицию объекта, если явно не реализованы:

* depth / stereo;
* object detection или segmentation;
* camera projection / ray casting;
* transform из camera frame в map frame;
* проверка достижимости целевой точки.

---

## 7. ROS2 и Nav2 правила

Для программной навигации использовать основной интерфейс Nav2:

```text
Action: /navigate_to_pose
Type: nav2_msgs/action/NavigateToPose
```

Не использовать `/goal_pose` как основной программный интерфейс.

`/goal_pose` допустим только как:

* RViz/debug-интерфейс;
* ручная проверка;
* временный diagnostic fallback.

Все navigation goals должны быть `geometry_msgs/PoseStamped` с корректным:

* `header.frame_id`;
* `header.stamp`;
* `pose.position`;
* `pose.orientation`.

---

## 8. TF правила

Система должна поддерживать согласованное TF-дерево:

```text
map -> odom -> base_link -> camera_link -> camera_optical_frame
```

Все сохранённые позы памяти должны содержать:

* `pose_frame`;
* timestamp;
* position;
* orientation;
* source topic;
* source frame;
* при наличии — covariance.

Нельзя смешивать `map` и `odom` без явного transform.

Если поза сохранена в `odom`, а Nav2 ожидает `map`, агент должен явно выполнить transform через TF2 или пометить проблему как архитектурный риск.

---

## 9. Visual SLAM / odometry правила

Допускается использование NVIDIA Isaac ROS Visual SLAM или другого VSLAM/odometry источника.

Но Visual SLAM нельзя описывать как полноценную замену глобальной локализации.

Visual SLAM может давать:

* odometry;
* pose estimate;
* trajectory;
* visual tracking.

Но для Nav2 в `map` frame всё равно требуется согласованная локализация и TF-цепочка.

Если используется `robot_localization`, его роль:

* fusion odometry / IMU / pose;
* сглаживание локальной оценки движения;
* публикация согласованной odometry.

`robot_localization` не должен описываться как модуль, который сам создаёт глобальную семантическую карту или исправляет все ошибки SLAM.

---

## 10. RuCLIP правила

Основная модель для русскоязычных запросов:

```text
RuCLIP
```

Допустимые варианты после проверки:

```text
ai-forever/ru-clip
ai-forever/ruclip-vit-base-patch32-384
ai-forever/ruclip-vit-base-patch16-224
```

Агент должен использовать русскоязычные запросы напрямую, без обязательного перевода на английский.

Для инференса:

* использовать `torch.no_grad()` или `torch.inference_mode()`;
* нормализовать image/text embeddings перед cosine similarity;
* не хранить ненормализованные вектора, если Qdrant настроен на cosine;
* по умолчанию использовать `batch_size=1` при VRAM 8–10 GB;
* предусмотреть CPU fallback;
* не держать лишние tensors на GPU;
* не вызывать модель в ROS callback без контроля времени выполнения, если это блокирует executor.

Если API конкретной RuCLIP-модели отличается от Hugging Face Transformers API, агент обязан сначала проверить фактический способ загрузки модели.

---

## 11. Qdrant правила

Qdrant является основной векторной БД проекта.

Каждый point в Qdrant должен содержать embedding и payload.

Минимальный payload:

```json
{
  "memory_id": "uuid",
  "timestamp": 0.0,
  "frame_id": "camera_frame_000001",
  "image_path": "data/keyframes/frame_000001.png",
  "pose_frame": "map",
  "pose": {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
  },
  "status": "active",
  "scene_id": "scene_01",
  "robot_id": "robot_01",
  "embedding_model": "ai-forever/ru-clip",
  "similarity_metric": "cosine"
}
```

Разрешённые статусы памяти:

```text
active
stale
uncertain
```

FAISS может использоваться только как offline baseline или ablation experiment, если задача явно этого требует.

---

## 12. QoS правила

Для sensor streams использовать sensor QoS:

* RGB images;
* depth images;
* camera info;
* IMU;
* high-rate odometry при необходимости.

Для управляющих интерфейсов использовать reliable QoS:

* services;
* actions;
* low-rate status topics;
* memory update events;
* diagnostics.

В simulation launch-файлах учитывать:

```text
use_sim_time:=true
```

---

## 13. Keyframe selection правила

Keyframe selection не должен быть простым сохранением каждого кадра.

Минимальные критерии отбора:

* минимальный временной интервал между keyframes;
* минимальное перемещение робота;
* минимальный поворот робота;
* контроль blur / image quality при наличии;
* опционально — novelty score по embedding distance.

Каждый keyframe должен иметь синхронизированную pose.

Если точная синхронизация image и pose не реализована, это должно быть отмечено как риск.

---

## 14. Verification правила

Верификация после прибытия не должна считаться абсолютной детекцией объекта.

RuCLIP cosine similarity показывает семантическую близость изображения и текста.

Пороговые значения являются экспериментальными параметрами.

Допустимый начальный шаблон:

```text
confirm_threshold: 0.25
stale_threshold: 0.15
uncertain range: [0.15, 0.25)
```

Но агент обязан писать, что эти значения должны быть откалиброваны на validation scenes.

Нельзя выдавать эти значения за универсальные.

---

## 15. Экспериментальные правила

Агент не должен генерировать фейковые результаты.

Разрешено генерировать:

* сценарии экспериментов;
* скрипты запуска;
* шаблоны таблиц;
* шаблоны графиков;
* команды evo;
* Python-код для подсчёта метрик.

Запрещено:

* придумывать Recall@K;
* придумывать ATE/RPE;
* придумывать latency;
* делать выводы без данных.

---

## 16. Формат ответа агента

При запросе кода агент должен вернуть:

1. Краткое решение.
2. Полные файлы, если пользователь не попросил patch.
3. Команды сборки.
4. Команды запуска.
5. Проверку работоспособности.
6. Риски и альтернативы.

При запросе архитектуры агент должен вернуть:

1. Краткое решение.
2. ASCII/Markdown-диаграмму.
3. Nodes.
4. Topics.
5. Actions.
6. Services.
7. TF frames.
8. QoS.
9. Failure modes.

При запросе экспериментов агент должен вернуть:

1. Гипотезу.
2. Переменные эксперимента.
3. Сценарий Isaac Sim.
4. Сбор данных.
5. Метрики.
6. Скрипты.
7. Формат графиков.
8. Ограничения валидности.

---

## 17. Запрещено

Агенту запрещено:

* выдумывать ROS2 packages;
* выдумывать Isaac Sim APIs;
* выдумывать Qdrant APIs;
* выдумывать RuCLIP APIs;
* выдавать `/goal_pose` за основной production-интерфейс Nav2;
* игнорировать TF;
* смешивать `map` и `odom` без transform;
* утверждать, что keyframe pose равна координате объекта;
* утверждать, что RuCLIP является object detector;
* генерировать фейковые метрики;
* заменять архитектуру проекта toy-скриптом;
* молча менять целевую ROS2-дистрибуцию без отдельного migration plan;
* молча менять Isaac Sim 6.0 на другую версию.

````
