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

Именно такой формат ожидает `src/evaluation.ipynb`.

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

**Артефакты:**
- `artifacts/validation_status_svoboden_asr_diarization_colab/windows/val_NNN_YYYYY/`
  - `audio.wav`, `transcript.json`, `diarization.json`

Для повторного запуска:
```bash
python -m src.pipeline --only-asr   # только ASR
python -m src.pipeline --only-diarization  # только diarization
```

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

**Текущие метрики (validation_status_svoboden_local_postprocess_v4):**
| Набор    | P     | R     | F1    | matched |
|----------|-------|-------|-------|---------|
| all      | 0.321 | 0.188 | 0.237 | 18      |
| opening  | 0.318 | 0.175 | 0.226 | 14      |
| closing  | 0.333 | 0.250 | 0.286 | 4       |

Метрики ограничены качеством speaker attribution (Resemblyzer); дальнейший тюнинг правил нецелесообразен — переходим к ML.

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

**Статус:** реализован, готов к обучению и интеграции в pipeline.

**Файлы этапа:**
- `src/ml_intent.py` — TF-IDF (char n-gram 2-5) + LogisticRegression(class_weight='balanced')
- `src/train_intent_model.py` — CLI для обучения модели
- `src/predict_intent_model.py` — CLI для инференса на JSONL-репликах
- `src/evaluate.py` — оценка предсказаний против gold на уровне реплик

**Обучение:**
```bash
python -m src.train_intent_model \
  --fit-input data/processed/gold_dialogues.jsonl \
  --model-output data/models/intent_classifier.joblib \
  --stats-output data/models/train_stats.json
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

**Статус:** планируется после стабилизации ML-модели.

**Входные данные:** только видео/аудио `data/raw/test/piter_fm.*` (без gold).

**Шаги:**
1. Запустить ASR + diarization на Colab (так же, как для validation-фильма).
2. Подготовить голосовые профили персонажей (выделить образцы голоса из аудио).
3. Запустить `pipeline.py` с `--intent-mode ml` или `--intent-mode combined`.
4. Сохранить `extracted_pairs.xlsx` как итоговый результат.

---

## Структура проекта

```text
.
├── artifacts/
│   ├── eval_comparison.json              # сравнение метрик по всем версиям
│   └── validation_status_svoboden_local_postprocess_vN/
│       ├── extracted_pairs.xlsx
│       ├── gold.xlsx
│       ├── eval_metrics.json
│       └── windows/
├── configs/
├── data/
│   ├── raw/
│   │   ├── gold/                         # data_val.xlsx
│   │   ├── validation/                   # audio + audio_profiles
│   │   └── test/                         # Питер FM видео/аудио
│   ├── interim/
│   ├── models/                           # intent_classifier.joblib
│   └── processed/
│       ├── gold_dialogues.jsonl          # 1170 реплик для обучения
│       └── gold_stats.json
├── notebooks/
│   └── validation_postprocess_and_evaluation_local.ipynb
├── reports/
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
    └── legacy/                           # исходные стаб-файлы (архив)
```
