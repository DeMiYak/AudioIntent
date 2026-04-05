# Извлечение выражений установления и прекращения контакта в русском фильме

Проект решает задачу извлечения реплик установления и прекращения контакта в русскоязычных фильмах и приводит результат к формату оценки из `evaluation.ipynb`.

---

## Постановка задачи

### Что является входом

В проекте используются три группы входных данных.

1. **Gold-разметка** в `data/raw/gold/data_val.xlsx`:
   - лист **`Диалоги`** — источник примеров для обучения ML-модели;
   - лист **`Вал - Статус свободен`** — validation-окна с gold-парами;
   - лист **`Статус свободен - персонажи`** — список персонажей validation-фильма.

2. **Validation-данные** для фильма **«Статус: свободен»**:
   - видео и/или аудио фильма;
   - папка `audio_profiles` с голосовыми профилями персонажей.

3. **Test-данные** для фильма **«Питер FM»**:
   - видео и/или аудио фильма (только медиафайл, без gold-разметки).

Поддерживаются форматы: `mkv`, `ac3`, `dts`, `wav`, `mp3`, `flac`, `m4a`, `aac`, `mp4`.

### Что является целевым выходом

Целевой результат для оценки — две колонки:

- `opening`
- `closing`

Значения этих колонок должны иметь вид:

- `Спикер - фраза`
- несколько пар внутри одной ячейки разделяются через `;`

Пример:

```text
opening = "Никита - привет; Алина - здравствуй"
closing = "Алина - пока"
```

- `Временная метка - спикер - фраза - тип`
- дополнительный вывод, ячейки разделяются на "Время начала", "Время конца", "Спикер - фраза" в столбцах-типах (opening-closing)

 Файл `src/evaluation.ipynb` ожидает `Спикер - фраза`.

### Что оценивается

На validation-выборке качество считается по листу **`Вал - Статус свободен`**, где для каждого окна уже заданы:
- временные границы фрагмента;
- gold-значения в колонках `opening` и `closing`.

Весь pipeline должен приводить сырые аудио- и видео-данные к файлу `extracted_pairs.xlsx`, совместимому с ноутбуком оценки.

---

## Описание решения

### Общая идея

Решение построено как поэтапный pipeline:

1. из gold-разметки (лист `Диалоги`) подготавливаются обучающие примеры;
2. на их основе строится rule-based baseline **и** обучается ML-классификатор;
3. для каждого validation-окна вырезается соответствующий фрагмент аудио;
4. выполняются ASR и diarization;
5. из этих результатов собираются реплики (`utterances`);
6. diarization-speakers сопоставляются с персонажами через голосовые профили;
7. по тексту реплик извлекаются `opening` / `closing` (rule-based, ML, или оба);
8. найденные пары агрегируются в Excel-формат `Спикер - фраза`.

### Стек

- **Python** — основной язык проекта.
- **pandas / openpyxl** — чтение Excel, подготовка таблиц, экспорт в `xlsx`.
- **faster-whisper** — ASR (запускается на Colab, артефакты используются локально).
- **pyannote.audio** — speaker diarization (запускается на Colab).
- **Resemblyzer** — голосовые эмбеддинги для speaker identification.
- **scikit-learn** — TF-IDF + LogisticRegression для ML-классификатора; метрики.
- **joblib** — сериализация ML-модели.
- **ffmpeg** — извлечение и нормализация аудио.
- **Jupyter Notebook** — прогон validation pipeline и оценка метрик.

Зависимости для локального запуска: `requirements.txt`.
Зависимости для Google Colab ноутбуков уже представлены внутри самих ноутбуков. Достаточно просто один раз запустить все ячейки

---

## Шаги реализации

### Шаг 1. Зафиксировать формат задачи и структуру проекта

**Статус:** выполнен.

**Файлы этапа:**
- `README.md`
- `requirements.txt`
- `configs/model.yaml`, `configs/paths.yaml`

Зафиксирована структура репозитория и целевой формат `opening` / `closing`.

---

### Шаг 2. Подготовить gold-разметку из листа `Диалоги`

**Статус:** выполнен.

**Файлы этапа:**
- `src/preprocess_gold.py`
- `data/raw/gold/data_val.xlsx`
- `data/processed/gold_dialogues.jsonl` — 1170 реплик из 18 фильмов
- `data/processed/gold_stats.json`

```bash
python -m src.preprocess_gold \
  --input data/raw/gold/data_val.xlsx \
  --output data/processed/gold_dialogues.jsonl \
  --stats-output data/processed/gold_stats.json
```

Распределение меток: 582 none / 413 contact_open / 178 contact_close.

---

### Шаг 3. Построить rule-based baseline по gold-разметке

**Статус:** выполнен, стабилизирован.

**Файлы этапа:**
- `src/rule_based_intent.py` — TF-IDF лексикон (229 open / 119 close) + MANUAL_PATTERNS

Ключевые механизмы: `collect_candidates_for_text`, `score_candidate`, `expand_match_to_phrase`, `acceptance_threshold`.
MANUAL_PATTERNS покрывают типовые русские приветствия и прощания (здравствуй(те), алло, ну пока, до свидания, до встречи, увидимся и др.).

```bash
python -m src.rule_based_intent \
  --fit-input data/processed/gold_dialogues.jsonl \
  --predict-input data/processed/gold_dialogues.jsonl \
  --lexicon-output data/processed/rule_lexicon.json \
  --predictions-output artifacts/rule_based_predictions.jsonl \
  --metrics-output artifacts/rule_based_metrics.json
```

---

### Шаг 4. Подготовить слой работы с validation Excel

**Статус:** выполнен.

**Файлы этапа:**
- `src/validation_io.py`
- `src/export_validation_gold.py`

```bash
python -m src.export_validation_gold \
  --input data/raw/gold/data_val.xlsx \
  --output artifacts/validation_status_svoboden/gold.xlsx
```

---

### Шаг 5–6. ASR и diarization

**Статус:** выполнены на Colab; артефакты хранятся локально.

**Файлы этапа:**
Сначала работаем с ASR notebook:
- `notebooks/google_colab_asr_pipeline.ipynb`
Затем с Diarization notebook:
- `notebooks/google_colab_diarization_pipeline_venv.ipynb`

**Артефакты:**
- `artifacts/validation_status_svoboden_asr_diarization_colab/windows/val_NNN_YYYYY/`
  - `audio.wav`, `transcript.json`, `diarization.json`

---

### Шаг 7–8. Сборка реплик и speaker identification

**Статус:** реализованы в `src/utterance_builder.py` и `src/speaker_id.py`.

- `--diarization-segment-mode regular` — использовать `regular_segments` из diarization.json (исправляет коллапс спикеров в 6 validation-окнах).
- Resemblyzer, порог сходства 0.65, мин. длительность аудио на спикера 1.5 с.
- Голосовые профили персонажей: `data/raw/validation/audio_profiles/`.

---

### Шаг 9. Формирование пар `opening` / `closing` — validation pipeline

**Статус:** выполнен и воспроизводим.

**Файлы этапа:**
- `src/pair_formatter.py`
- `src/pipeline.py`
- `notebooks/validation_postprocess_and_evaluation_local.ipynb`
- `artifacts/eval_comparison.json` — сравнение метрик по всем версиям

**Текущие метрики (validation_status_svoboden_local_postprocess_v11, combined mode):**
| Набор    | P     | R     | F1    | matched | exact |
|----------|-------|-------|-------|---------|-------|
| all      | 0.231 | 0.250 | 0.240 | 34      | 8     |
| opening  | 0.278 | 0.250 | 0.263 | 27      | 7     |
| closing  | 0.125 | 0.250 | 0.167 | 7       | 1     |

Метрики ограничены качеством speaker attribution (Resemblyzer). Лучший результат достигается при `--intent-mode combined`, `--similarity-threshold 0.48`, `--ml-confidence-threshold 0.35`. Дальнейший тюнинг на validation нецелесообразен — переходим к тестовому фильму.

**Полный validation pipeline:**
```bash
python -m src.pipeline \
  --gold-excel data/raw/gold/data_val.xlsx \
  --fit-input data/processed/gold_dialogues.jsonl \
  --validation-dir data/raw/validation \
  --media-input data/raw/validation/status_svoboden.mkv \
  --samples-dir data/raw/validation/audio_profiles \
  --output-dir artifacts/validation_status_svoboden_local_postprocess_vN \
  --diarization-segment-mode regular \
  --skip-asr --skip-diarization \
  --transcript-input-dir artifacts/validation_status_svoboden_asr_diarization_colab/windows \
  --diarization-input-dir artifacts/validation_status_svoboden_asr_diarization_colab/windows \
  --hf-token YOUR_HF_TOKEN
```

---

### Шаг 10. ML-классификатор намерений

**Статус:** реализован, обучен, интегрирован в pipeline (combined mode).

**Файлы этапа:**
- `src/ml_intent.py` — TF-IDF (char n-gram 2-5) + sklearn-классификатор; признаки: текст реплики, относительная позиция в диалоге, маркеры контакта в соседних репликах (контекстные признаки)
- `src/train_intent_model.py` — CLI для обучения; поддерживает `--classifier lr|svm|nb|sgd|ridge`
- `src/compare_classifiers.py` — сравнение всех классификаторов на validation-выборке
- `src/predict_intent_model.py` — CLI для инференса на JSONL-репликах
- `src/evaluate.py` — оценка предсказаний против gold на уровне реплик

**Обучение (Ridge — лучший по F1 на validation):**
```bash
python -m src.train_intent_model \
  --fit-input data/processed/gold_dialogues.jsonl \
  --model-output data/models/intent_classifier.joblib \
  --stats-output data/models/train_stats.json \
  --classifier ridge
```

**Сравнение классификаторов:**
```bash
python -m src.compare_classifiers \
  --fit-input data/processed/gold_dialogues.jsonl \
  --gold-excel data/raw/gold/data_val.xlsx \
  --transcript-dir artifacts/validation_status_svoboden_asr_diarization_colab/windows \
  --samples-dir data/raw/validation/audio_profiles \
  --output artifacts/classifier_comparison.json
```

**Инференс на репликах:**
```bash
python -m src.predict_intent_model \
  --model data/models/intent_classifier.joblib \
  --input utterances.jsonl \
  --output predictions.jsonl
```

**Оценка предсказаний против gold:**
```bash
python -m src.evaluate \
  --gold data/processed/gold_dialogues.jsonl \
  --predictions artifacts/ml_predictions.jsonl \
  --output artifacts/eval_ml_metrics.json
```

**Интеграция в pipeline** через `--intent-mode`:
```bash
python -m src.pipeline \
  ... \
  --intent-mode ml \
  --ml-model data/models/intent_classifier.joblib

# или combined (rule-based + ML, rule-based имеет приоритет):
python -m src.pipeline \
  ... \
  --intent-mode combined \
  --ml-model data/models/intent_classifier.joblib
```

---

### Шаг 11. Тестовый фильм «Питер FM»

**Статус:** выполнен полностью.

**Входные данные:** только видео/аудио `data/raw/test/Peter_FM_2006.mkv` (без gold).

**Голосовые профили** (14 персонажей) хранятся локально в `data/raw/test/audio_profile/` (не в git).
Для воспроизведения: запустить команды из `Peter_FM_audio_prompt.txt` через `src/extract_audio_profile.py`.
Персонажи: Маша Емельянова, Максим Васильев, Лерыч, Костя, Немец-контрактор, Марина, Татьяна Петровна, Директор радио (Феликс), Дима-бедуин, Управдом, Майор Горобец, Генерал Пётр Ефимыч, Петя (Друг Макса 1), Мужик на скамейке.

**Шаги:**
1. **[выполнен]** Извлечь голосовые профили персонажей с помощью команд из `Peter_FM_audio_prompt.txt`.
2. **[выполнен]** Разбить фильм на чанки по 10 минут:
   ```bash
   python -m src.chunk_film \
     --input data/raw/test/Peter_FM_2006.mkv \
     --output-dir artifacts/test_piter_fm_asr_diarization_colab/windows \
     --chunk-duration 600
   ```
3. **[выполнен]** Запустить ASR на Colab (`notebooks/google_colab_asr_pipeline_test_film.ipynb`).
4. **[выполнен]** Запустить diarization на Colab (`notebooks/google_colab_diarization_pipeline_venv_test_film.ipynb`).
5. **[выполнен]** Запустить постпроцессинг локально:
   ```bash
   python -m src.pipeline \
     --scan-windows \
     --transcript-input-dir artifacts/test_piter_fm_asr_diarization_colab/windows \
     --diarization-input-dir artifacts/test_piter_fm_asr_diarization_colab/windows \
     --samples-dir data/raw/test/audio_profile \
     --output-dir artifacts/test_piter_fm \
     --extracted-pairs-output artifacts/test_piter_fm/extracted_pairs.xlsx \
     --diarization-segment-mode regular \
     --intent-mode combined \
     --ml-model data/models/intent_classifier.joblib \
     --ml-confidence-threshold 0.35 \
     --similarity-threshold 0.48 \
     --skip-asr --skip-diarization \
     --fit-input data/processed/gold_dialogues.jsonl
   ```
6. **[выполнен]** Экспортировать плоскую таблицу событий:
   ```bash
   python -m src.export_detailed_pairs \
     --output-dir artifacts/test_piter_fm \
     --source-dir artifacts/test_piter_fm_asr_diarization_colab/windows \
     --film-name "Питер FM" \
     --exclude-chunk 9 \
     --excel artifacts/test_piter_fm/detailed_pairs.xlsx
   ```
   Результат — `artifacts/test_piter_fm/detailed_pairs.xlsx`: колонки `ID | Фильм | Время начала | Время окончания | Тип | Аннотация | opening | closing`.

---

## Настройка Google Drive для Colab

Ноутбуки ASR и diarization монтируют Google Drive и ожидают следующую структуру папок:

```text
MyDrive/
└── AudioIntent/
    ├── notebooks/                             # клонировать из репозитория
    │   ├── google_colab_asr_pipeline_test_film.ipynb
    │   └── google_colab_diarization_pipeline_venv_test_film.ipynb
    └── data/
        ├── raw/
        │   └── test/
        │       ├── Peter_FM_2006.mkv          # видеофайл фильма
        │       └── audio_profile/             # голосовые профили (14 папок с WAV)
        ├── processed/
        │   └── gold_dialogues.jsonl           # подготовить локально (шаг 2)
        └── artifacts/
            └── test_piter_fm_asr_diarization_colab/  # создаётся автоматически при ASR
                └── windows/
                    ├── chunk_000/
                    │   ├── audio.wav
                    │   ├── chunk_info.json
                    │   └── transcript.json    # записывается ASR-ноутбуком
                    └── ...
```

**Что нужно загрузить вручную перед запуском:**
- `Peter_FM_2006.mkv` — видеофайл (~466 МБ)
- `audio_profile/` — голосовые профили (~10 МБ, 14 папок)
- `gold_dialogues.jsonl` — подготовить локально командой из шага 2 и загрузить (~1 МБ)

**Что создаётся автоматически:**
- `artifacts/test_piter_fm_asr_diarization_colab/windows/` — чанки + транскрипты создаются ASR-ноутбуком (~170 МБ)
- `diarization.json` в каждый чанк записывает diarization-ноутбук (~5 МБ суммарно)

**Необходимый объём свободного места на Google Drive:** не менее 1 ГБ.

**HuggingFace token** для ноутбуков нужно получить токен на [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens); модель `pyannote/speaker-diarization-community-1` требует принятия условий использования на странице модели. В Google Colab нужно нажать на ключ в левой панели (`Secrets`) и добавить токен, а затем разрешить доступ (`Notebook Access`).

---

## Структура проекта

```text
.
├── artifacts/                            # генерируется при запуске, в git только .gitkeep
│   ├── eval_comparison.json              # сравнение метрик по всем версиям
│   ├── validation_status_svoboden_asr_diarization_colab/  # ASR + diarization (Colab)
│   ├── validation_status_svoboden_local_postprocess_vN/   # постпроцессинг (локально, версия vN)
│   │   ├── extracted_pairs.xlsx
│   │   ├── gold.xlsx
│   │   ├── eval_metrics.json
│   │   └── windows/
│   ├── test_piter_fm_asr_diarization_colab/               # ASR + diarization тест (Colab)
│   └── test_piter_fm/                                     # постпроцессинг тест (локально)
│       ├── extracted_pairs.xlsx
│       └── detailed_pairs.xlsx
├── configs/
├── data/
│   ├── raw/
│   │   ├── gold/                         # data_val.xlsx
│   │   ├── validation/                   # audio + audio_profiles
│   │   └── test/                         # Питер FM видео/аудио + audio_profile/
│   ├── interim/
│   ├── models/                           # intent_classifier.joblib
│   └── processed/
│       ├── gold_dialogues.jsonl          # 1170 реплик для обучения
│       └── gold_stats.json
├── notebooks/
│   ├── evaluation.ipynb                          # оценка метрик на validation-выборке
│   ├── validation_postprocess_and_evaluation_local.ipynb
│   ├── google_colab_asr_pipeline.ipynb           # ASR на Colab (validation)
│   ├── google_colab_asr_pipeline_test_film.ipynb # ASR на Colab (тестовый фильм)
│   ├── google_colab_diarization_pipeline_venv.ipynb          # diarization (validation)
│   └── google_colab_diarization_pipeline_venv_test_film.ipynb # diarization (тест)
└── src/
    ├── pipeline.py                       # главный orchestrator
    ├── rule_based_intent.py              # rule-based извлечение
    ├── ml_intent.py                      # ML-классификатор (TF-IDF + LR)
    ├── train_intent_model.py             # CLI обучения ML-модели
    ├── predict_intent_model.py           # CLI инференса ML-модели
    ├── evaluate.py                       # CLI оценки предсказаний
    ├── pair_formatter.py                 # агрегация в opening/closing
    ├── speaker_id.py                     # Resemblyzer speaker attribution
    ├── utterance_builder.py              # сборка реплик из ASR+diarization
    ├── validation_io.py                  # чтение validation Excel
    ├── asr.py                            # faster-whisper ASR
    ├── diarization.py                    # pyannote diarization
    ├── chunk_film.py                     # разбивка фильма на чанки для тестового пайплайна
    ├── extract_audio_profile.py          # извлечение голосовых профилей из видео (ffmpeg)
    ├── export_detailed_pairs.py          # экспорт плоской таблицы событий в Excel
    └── compare_classifiers.py            # сравнение ML-классификаторов на validation-выборке
```
