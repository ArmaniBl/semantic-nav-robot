# Project Context

## Назначение

Проект реализует управляемую навигацию мобильного робота по сохраненным
визуальным наблюдениям.

Оператор записывает базу keyframes, выбирает collection, вводит текстовый
запрос и отправляет робота к найденной позе наблюдения.

## Основной Поток Данных

```text
Isaac Sim / ROS topics
  -> RGB image + odometry + TF
  -> keyframe recorder
  -> metadata + images
  -> Qdrant collection
  -> text query
  -> candidate observation pose
  -> Nav2 NavigateToPose
  -> live camera verification
```

## Важное Ограничение

Система едет к позе наблюдения, а не к точной 3D-координате объекта.

```text
Найденный keyframe pose = место, откуда нужная область была видна.
```

Если требуется точная 3D-локализация объекта, нужны дополнительные компоненты:

- depth или stereo;
- алгоритм выделения нужной области изображения;
- camera projection;
- ray casting;
- transform из camera frame в рабочий frame;
- проверка reachable goal pose.

Эти компоненты не входят в текущий runtime.

## Текущий Runtime

Основной режим:

```text
web_control_panel.py
  -> launch/nav2_odom_launch.py
  -> ros_keyframe_recorder.py
  -> Qdrant
  -> semantic_nav_to_pose.py
```

Рабочий goal frame:

```text
odom
```

## `Go` Mission

Один запуск `Go`:

1. принимает текстовый запрос;
2. выбирает candidates из Qdrant;
3. проверяет, что pose candidate сохранена в нужном frame;
4. отправляет goal в `/navigate_to_pose`;
5. во время движения проверяет live camera;
6. если совпадение найдено, goal отменяется и робот останавливается;
7. если робот прибыл к candidate, запускается поворот на месте;
8. поворот идет до 360 градусов по TF yaw;
9. если совпадение найдено во время поворота, поворот сразу останавливается;
10. если совпадения нет, пробуется следующий candidate;
11. максимум candidates за один запуск задается `--max-candidate-attempts`.

## Порог Совпадения

Порог зависит от количества слов в запросе:

```text
1 слово  -> 0.25
2 слова  -> 0.26
больше   -> порог растет медленнее
максимум -> 0.8
```

Это значение пишется в логах:

```text
Dynamic visual match threshold for query='...' words=N threshold=X
```

## Запись Данных

Recorder сохраняет:

- image path;
- image topic;
- image frame;
- pose frame;
- pose source frame;
- pose position;
- pose orientation;
- run id;
- status.

После записи данные загружаются в Qdrant collection:

```text
semantic_visual_memory_run_YYYYMMDD_HHMMSS
```

## Безопасность Движения

Runtime использует несколько защит:

- Nav2 planner/controller;
- проверку расстояния до цели по action feedback;
- остановку при близком препятствии по `/scan`;
- публикацию zero `/cmd_vel` при отмене goal;
- проверку непрерывности keyframe trajectory перед использованием collection.

## Логи

Основные файлы:

```text
logs/web/process.log
logs/web/events_YYYYMMDD_HHMMSS.log
```

По логам проверяются:

- выбранная collection;
- отправленная goal pose;
- Nav2 feedback;
- distance remaining;
- calculated threshold;
- live camera score;
- arrival spin start/finish;
- mission result.

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
