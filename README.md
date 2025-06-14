# ML Сервис для Скоринга

Этот сервис предоставляет контейнеризированное решение для ML-скоринга, которое обрабатывает входные данные и генерирует предсказания с использованием предобученной модели CatBoost.

## Возможности

- Обработка входных данных в формате CSV.
- Применение шагов предобработки данных.
- Генерация предсказаний с помощью предобученной модели CatBoost.
- Вывод результатов в формате CSV.
- Генерация JSON-файла с топ-5 важными признаками модели.
- Создание PNG-файла с графиком плотности распределения предсказанных скоров.

## Требования

Этот проект доступен на GitHub. Для его полноценной работы вам потребуется:

-   **Docker Desktop** (или Docker Engine) установленный и запущенный на вашей системе.
-   Файлы данных:
    -   `test.csv`: Входные данные для скоринга в формате CSV.
    -   `catboost_model.cbm`: Предобученная модель CatBoost.
    *Обратите внимание: Эти файлы (а также train.csv и sample_submition.csv, если они вам нужны для локального тестирования/обучения) не включены в репозиторий из-за их большого размера. Вы можете скачать их по ссылке: [Google Drive Folder](https://drive.google.com/drive/folders/1IYkbv7U6WUqZetn2FFJ5SizYQuNZ9QEJ?usp=sharing)*

## Структура проекта

```
.
├── app.py              # Основной скрипт приложения, координирующий весь процесс.
├── preprocess.py       # Модуль, содержащий функции для предобработки входных данных.
├── score.py           # Модуль для загрузки модели, получения важности признаков и выполнения предсказаний.
├── requirements.txt    # Список Python-зависимостей проекта.
├── Dockerfile         # Инструкции для сборки Docker-образа.
├── input/            # Директория для входных файлов (например, test.csv). Будет монтироваться в контейнер.
└── output/           # Директория для выходных файлов (предсказания, важность признаков, графики). Будет монтироваться из контейнера.
```

## Запуск сервиса

Выполните следующие шаги в терминале (рекомендуется PowerShell для Windows) в корневой директории проекта `ml_ops1`.

### Шаг 1: Подготовка файлов и директорий

1.  Убедитесь, что все необходимые файлы (`app.py`, `preprocess.py`, `score.py`, `requirements.txt`, `Dockerfile`, `catboost_model.cbm`) находятся в корневой директории проекта.
2.  Создайте директории для входных и выходных данных, если их еще нет:
    ```bash
    mkdir input
    mkdir output
    ```
3.  Скопируйте ваш файл `test.csv` (который вы хотите скорить) в только что созданную папку `input`:
    ```bash
    copy test.csv input/
    ```
    (Для Linux/macOS используйте `cp test.csv input/`)

### Шаг 2: Сборка Docker-образа

Этот шаг создаст исполняемый образ вашего ML-сервиса.

1.  Откройте терминал (например, PowerShell) в корневой директории проекта.
2.  Выполните команду для сборки образа:
    ```bash
    docker build -t ml-scoring-service .
    ```
    -   `-t ml-scoring-service`: Присваивает имя `ml-scoring-service` вашему образу.
    -   `.`: Указывает, что `Dockerfile` находится в текущей директории.

    *Пояснение:* Этот процесс может занять некоторое время, так как Docker загружает базовый образ Python (версии 3.11) и устанавливает все зависимости из `requirements.txt`. Также будет скопирована большая модель `catboost_model.cbm`.

    *Проверка:* После успешной сборки вы увидите сообщение `Successfully built <image_id>` и `Successfully tagged ml-scoring-service:latest`.

### Шаг 3: Запуск Docker-контейнера

Этот шаг запустит ваш ML-сервис внутри изолированного Docker-контейнера.

1.  В том же терминале выполните команду для запуска контейнера:
    ```bash
    docker run -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output ml-scoring-service
    ```
    -   `-v ${PWD}/input:/app/input`: Монтирует вашу локальную папку `input` (где лежит `test.csv`) в папку `/app/input` внутри контейнера. Это позволяет контейнеру читать входные данные.
    -   `-v ${PWD}/output:/app/output`: Монтирует вашу локальную папку `output` в папку `/app/output` внутри контейнера. Это позволяет контейнеру записывать результаты в вашу локальную файловую систему.
    -   `ml-scoring-service`: Указывает Docker, какой образ использовать для запуска контейнера.

    *Пояснение:* Сервис выполнит предобработку `test.csv`, сделает предсказания и сохранит результаты в `/app/output` внутри контейнера, которые затем появятся в вашей локальной папке `output`.

    *Для пользователей Linux/macOS:* Используйте `$(pwd)` вместо `${PWD}`:
    ```bash
    docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output ml-scoring-service
    ```

## Тестирование работоспособности

После выполнения Шага 3, сервис должен был отработать и завершиться. Все результаты будут находиться в вашей локальной папке `output`.

1.  **Проверьте содержимое выходной директории:**
    ```bash
    dir output/
    ```
    (Для Linux/macOS используйте `ls output/`)
    Вы должны увидеть три файла:
    -   `submission.csv`
    -   `feature_importance.json`
    -   `prediction_distribution.png`

2.  **Проверьте файл с предсказаниями (`submission.csv`):**
    Откройте его в текстовом редакторе или просмотрите первые строки:
    ```bash
    Get-Content output/submission.csv -Head 5
    ```
    (Для Linux/macOS используйте `head output/submission.csv`)
    Убедитесь, что файл содержит столбец `id` (генерируемый) и `target` с числовыми предсказаниями.

3.  **Проверьте файл с важностью признаков (`feature_importance.json`):**
    ```bash
    Get-Content output/feature_importance.json
    ```
    (Для Linux/macOS используйте `cat output/feature_importance.json`)
    Вы должны увидеть JSON-объект с названиями признаков и их значениями важности.

4.  **Проверьте график распределения предсказаний (`prediction_distribution.png`):**
    Откройте этот файл с помощью программы для просмотра изображений. График должен показывать распределение вероятностей (скоров) от 0 до 1.

## Устранение неполадок (Troubleshooting)

*   **Ошибка `docker: invalid reference format.`:** Это происходит, если вы используете `$(pwd)` в Windows PowerShell. Убедитесь, что вы используете `${PWD}`.
*   **Ошибка `FileNotFoundError: Input file input/test.csv not found`:** Убедитесь, что файл `test.csv` действительно находится в папке `input` и что вы правильно монтировали тома (`-v`).
*   **Контейнер запускается, но быстро завершается без ошибок:** Это нормальное поведение, так как скрипт `app.py` выполняет свою задачу и выходит. Проверьте содержимое папки `output` на наличие сгенерированных файлов. Если их нет, проверьте логи контейнера.
*   **Просмотр логов контейнера:** Если сервис не отработал как ожидалось, вы можете посмотреть логи последнего запущенного контейнера:
    ```bash
    docker logs $(docker ps -lq)
    ```