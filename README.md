# Извлечение реплик установления и прекращения контакта в русскоязычном фильме

Проект решает задачу автоматического извлечения реплик, которые устанавливают или завершают контакт между персонажами: приветствия, прощания, знакомства, начала и завершения коротких взаимодействий. Итоговый результат приводится к Excel-таблице с колонками `opening` и `closing`, где значения имеют формат `Спикер - фраза`.

Документация описывает текущий рабочий снимок кода из архива `v11.zip` и не предполагает наличие модулей, которых нет в `src/`.

---

## 1. Входные и выходные данные

### Вход

1. Gold-разметка: `data/raw/gold/data_val.xlsx`.
   - Лист `Диалоги` используется для обучения rule-based и ML-компонентов.
   - Лист `Вал - Статус свободен` используется для validation-оценки.
   - Лист с персонажами validation-фильма используется как справочная информация.

2. Validation-фильм `Статус: свободен`:
   - медиафайл фильма;
   - папка голосовых профилей персонажей: `data/raw/validation/audio_profiles/`;
   - артефакты ASR и diarization из Colab: `artifacts/validation_status_svoboden_asr_diarization_colab/windows/`.

3. Test-фильм `Питер FM`:
   - медиафайл фильма;
   - папка голосовых профилей: `data/raw/test/audio_profile/`;
   - артефакты ASR и diarization из Colab.

### Выход

Основной файл для оценки:

```text
extracted_pairs.xlsx
```

Колонки:

```text
opening | closing
```

Формат значения:

```text
Спикер - фраза
```

Если в одном окне найдено несколько событий одного типа, они объединяются в одной ячейке через `;`.

Дополнительный файл для просмотра результата на тестовом фильме:

```text
detailed_pairs.xlsx
```

Он создаётся модулем `src.export_detailed_pairs` и содержит построчную таблицу найденных событий.

---

## 2. Краткое описание pipeline

Общий порядок обработки:

1. Подготовить обучающие примеры из gold-разметки.
2. Построить rule-based лексикон.
3. Обучить ML-классификатор намерений.
4. Получить ASR-транскрипт фильма через faster-whisper.
5. Получить diarization через pyannote.
6. Собрать реплики из ASR + diarization.
7. Сопоставить diarization-спикеров с персонажами по голосовым профилям через Resemblyzer.
8. Извлечь `contact_open` / `contact_close` rule-based, ML или combined-режимом.
9. Агрегировать найденные события в Excel-таблицу.

---

## 3. Используемые файлы проекта

```text
src/preprocess_gold.py          # подготовка gold_dialogues.jsonl из Excel
src/rule_based_intent.py        # rule-based извлечение opening/closing
src/ml_intent.py                # TF-IDF + sklearn ML-классификатор
src/train_intent_model.py       # обучение ML-модели
src/predict_intent_model.py     # инференс ML-модели на JSONL-репликах
src/compare_classifiers.py      # сравнение LR/SVM/NB/SGD/Ridge
src/pipeline.py                 # основной validation/test pipeline
src/utterance_builder.py        # сборка реплик из ASR + diarization
src/speaker_id.py               # Resemblyzer speaker identification
src/pair_formatter.py           # агрегация predictions в opening/closing
src/export_detailed_pairs.py    # экспорт плоской таблицы событий
src/export_validation_gold.py   # экспорт gold.xlsx для validation
src/evaluate.py                 # оценка JSONL-предсказаний на уровне реплик
src/asr.py                      # faster-whisper ASR
src/diarization.py              # pyannote diarization
src/chunk_film.py               # разбиение фильма на чанки
src/extract_audio_profile.py    # извлечение голосовых профилей через ffmpeg
src/validation_io.py            # чтение validation-листа Excel
```

Ноутбуки:

```text
notebooks/google_colab_asr_pipeline.ipynb
notebooks/google_colab_diarization_pipeline_venv.ipynb
notebooks/google_colab_asr_pipeline_test_film.ipynb
notebooks/google_colab_diarization_pipeline_venv_test_film.ipynb
notebooks/validation_postprocess_and_evaluation_local.ipynb
notebooks/evaluation.ipynb
```

---

## 4. Основные компоненты

### 4.1. Gold-разметка

`src.preprocess_gold` извлекает из листа `Диалоги` реплики с тегами `<opening>` и `<closing>` и сохраняет их в JSONL.

Рабочий объём обучающих данных:

```text
1170 реплик из 18 фильмов
413 contact_open
175 contact_close
582 none
```

### 4.2. Rule-based компонент

Файл: `src/rule_based_intent.py`.

Компонент использует:

- лексикон, извлечённый из gold-разметки;
- сильные и слабые маркеры opening/closing;
- `MANUAL_PATTERNS` — регулярные выражения для устойчивых или важных конструкций.

В текущем коде ручные паттерны не разделяются на уровни надёжности. В combined-режиме все правила из `OPEN_MANUAL_RULES` и `CLOSE_MANUAL_RULES` освобождаются от ML-фильтра.

Примеры ручных паттернов:

```text
здравствуй / здравствуйте
алло
до свидания
до встречи
увидимся
пока-пока
давайте познакомимся
желаете что-нибудь
я пошёл
буду через полчаса
```

### 4.3. ML-компонент

Файлы:

```text
src/ml_intent.py
src/train_intent_model.py
src/predict_intent_model.py
```

Модель:

```text
TF-IDF по символьным n-граммам 2–5 + sklearn-классификатор
```

Поддерживаемые классификаторы:

```text
lr | svm | nb | sgd | ridge
```

Для финального рабочего запуска используется Ridge-модель, обученная через:

```bash
python -m src.train_intent_model \
  --fit-input data/processed/gold_dialogues.jsonl \
  --model-output data/models/intent_classifier.joblib \
  --stats-output data/models/train_stats.json \
  --classifier ridge
```

Признаки ML-модели:

- текст реплики;
- символьные n-граммы;
- относительная позиция реплики в диалоге;
- длина реплики;
- флаги начала/конца диалога;
- маркеры opening/closing в предыдущей и следующей реплике.

### 4.4. Combined-режим

Функция: `_run_intent_extraction()` в `src/pipeline.py`.

Логика текущего combined-режима:

1. Rule-based и ML запускаются на одних и тех же репликах.
2. ML-предсказания ниже `--ml-confidence-threshold` отбрасываются.
3. Rule-based предсказание сохраняется, если:
   - ML предсказал тот же `intent_type` для той же `utterance_id`, или
   - сработало ручное правило из `OPEN_MANUAL_RULES` / `CLOSE_MANUAL_RULES`.
4. ML-предсказания добавляются, если такая пара `(utterance_id, intent_type)` ещё не была добавлена rule-based компонентом.
5. Если одна реплика получила одновременно `contact_open` и `contact_close`, то rule-based предсказание удаляется только в случае, когда ML уверенно дал противоположный тип.

То есть combined-режим в текущем коде — это не простое объединение rule-based и ML, а объединение с ML-фильтрацией rule-based результатов и исключением для ручных правил.

### 4.5. Speaker identification

Файл: `src/speaker_id.py`.

Используется Resemblyzer:

```text
VoiceEncoder → embedding → cosine similarity
```

Порог принятия персонажа задаётся параметром:

```text
--similarity-threshold
```

Для validation-оценки использовался порог `0.48`.

Если лучший кандидат ниже порога, speaker получает техническую метку `unknown_speaker`, которая при экспорте заменяется на значение `--unknown-speaker-name`, обычно `unknown`.

---

## 5. Пошаговое воспроизведение через bash

Команды ниже рассчитаны на запуск из корня репозитория.

### Шаг 1. Подготовить gold JSONL

```bash
python -m src.preprocess_gold \
  --input data/raw/gold/data_val.xlsx \
  --sheet-name "Диалоги" \
  --output data/processed/gold_dialogues.jsonl \
  --stats-output data/processed/gold_stats.json \
  --skipped-output data/processed/gold_skipped_lines.json
```

### Шаг 2. Построить rule-based лексикон и проверить baseline

```bash
python -m src.rule_based_intent \
  --fit-input data/processed/gold_dialogues.jsonl \
  --predict-input data/processed/gold_dialogues.jsonl \
  --lexicon-output data/processed/rule_lexicon.json \
  --predictions-output artifacts/rule_based_predictions.jsonl \
  --metrics-output artifacts/rule_based_metrics.json \
  --min-freq 1
```

### Шаг 3. Обучить ML-модель

```bash
python -m src.train_intent_model \
  --fit-input data/processed/gold_dialogues.jsonl \
  --model-output data/models/intent_classifier.joblib \
  --stats-output data/models/train_stats.json \
  --classifier ridge
```

### Шаг 4. Опционально сравнить классификаторы

```bash
python -m src.compare_classifiers \
  --fit-input data/processed/gold_dialogues.jsonl \
  --gold-excel data/raw/gold/data_val.xlsx \
  --transcript-dir artifacts/validation_status_svoboden_asr_diarization_colab/windows \
  --samples-dir data/raw/validation/audio_profiles \
  --output artifacts/classifier_comparison.json \
  --classifiers lr svm nb sgd ridge \
  --cv-folds 5
```

### Шаг 5. Получить ASR и diarization

ASR и diarization выполняются в Colab-ноутбуках:

```text
notebooks/google_colab_asr_pipeline.ipynb
notebooks/google_colab_diarization_pipeline_venv.ipynb
```

После выполнения ноутбуков локально должна существовать папка:

```text
artifacts/validation_status_svoboden_asr_diarization_colab/windows/
```

В каждом окне должны быть как минимум:

```text
transcript.json
diarization.json
audio.wav
```

### Шаг 6. Запустить validation pipeline

```bash
python -m src.pipeline \
  --gold-excel data/raw/gold/data_val.xlsx \
  --validation-sheet "Вал - Статус свободен" \
  --fit-input data/processed/gold_dialogues.jsonl \
  --validation-dir data/raw/validation \
  --media-input data/raw/validation/status_svoboden.mkv \
  --samples-dir data/raw/validation/audio_profiles \
  --output-dir artifacts/validation_status_svoboden_local_postprocess_v11 \
  --extracted-pairs-output artifacts/validation_status_svoboden_local_postprocess_v11/extracted_pairs.xlsx \
  --gold-output artifacts/validation_status_svoboden_local_postprocess_v11/gold.xlsx \
  --transcript-input-dir artifacts/validation_status_svoboden_asr_diarization_colab/windows \
  --diarization-input-dir artifacts/validation_status_svoboden_asr_diarization_colab/windows \
  --skip-asr \
  --skip-diarization \
  --device cpu \
  --min-freq 1 \
  --min-sample-duration-sec 0.5 \
  --min-utterance-duration-sec 0.7 \
  --min-total-duration-sec 0.8 \
  --max-total-duration-sec 45.0 \
  --max-pause-within-utterance-sec 0.8 \
  --max-total-utterance-duration-sec 20.0 \
  --max-nonoverlap-assign-distance-sec 1.0 \
  --similarity-threshold 0.48 \
  --top-k-candidates 3 \
  --unknown-speaker-label unknown_speaker \
  --unknown-speaker-name unknown \
  --diarization-segment-mode regular \
  --intent-mode combined \
  --ml-model data/models/intent_classifier.joblib \
  --ml-confidence-threshold 0.35
```

Основные результаты команды:

```text
artifacts/validation_status_svoboden_local_postprocess_v11/extracted_pairs.xlsx
artifacts/validation_status_svoboden_local_postprocess_v11/gold.xlsx
artifacts/validation_status_svoboden_local_postprocess_v11/run_summary.json
artifacts/validation_status_svoboden_local_postprocess_v11/windows/*/predictions.jsonl
```

### Шаг 7. Оценить validation-результат

Для Excel-оценки используется ноутбук:

```text
notebooks/validation_postprocess_and_evaluation_local.ipynb
```

или `notebooks/evaluation.ipynb`, если нужно сравнить `gold.xlsx` и `extracted_pairs.xlsx` в формате evaluation.

### Шаг 8. Разбить тестовый фильм на чанки

```bash
python -m src.chunk_film \
  --input data/raw/test/Peter_FM_2006.mkv \
  --output-dir artifacts/test_piter_fm_asr_diarization_colab/windows \
  --chunk-duration 600 \
  --sample-rate 16000 \
  --force
```

### Шаг 9. Запустить ASR и diarization для тестового фильма

В Colab используются ноутбуки:

```text
notebooks/google_colab_asr_pipeline_test_film.ipynb
notebooks/google_colab_diarization_pipeline_venv_test_film.ipynb
```

Ожидаемая папка после их выполнения:

```text
artifacts/test_piter_fm_asr_diarization_colab/windows/
```

### Шаг 10. Запустить постпроцессинг тестового фильма

```bash
python -m src.pipeline \
  --scan-windows \
  --fit-input data/processed/gold_dialogues.jsonl \
  --samples-dir data/raw/test/audio_profile \
  --output-dir artifacts/test_piter_fm \
  --extracted-pairs-output artifacts/test_piter_fm/extracted_pairs.xlsx \
  --transcript-input-dir artifacts/test_piter_fm_asr_diarization_colab/windows \
  --diarization-input-dir artifacts/test_piter_fm_asr_diarization_colab/windows \
  --skip-asr \
  --skip-diarization \
  --device cpu \
  --min-freq 1 \
  --min-sample-duration-sec 0.5 \
  --min-utterance-duration-sec 0.7 \
  --min-total-duration-sec 0.8 \
  --max-total-duration-sec 45.0 \
  --max-pause-within-utterance-sec 0.8 \
  --max-total-utterance-duration-sec 20.0 \
  --max-nonoverlap-assign-distance-sec 1.0 \
  --similarity-threshold 0.48 \
  --top-k-candidates 3 \
  --unknown-speaker-label unknown_speaker \
  --unknown-speaker-name unknown \
  --diarization-segment-mode regular \
  --intent-mode combined \
  --ml-model data/models/intent_classifier.joblib \
  --ml-confidence-threshold 0.35
```

### Шаг 11. Экспортировать `detailed_pairs.xlsx`

```bash
python -m src.export_detailed_pairs \
  --output-dir artifacts/test_piter_fm \
  --source-dir artifacts/test_piter_fm_asr_diarization_colab/windows \
  --film-name "Питер FM" \
  --exclude-chunk 9 \
  --excel artifacts/test_piter_fm/detailed_pairs.xlsx
```

Итоговый файл:

```text
artifacts/test_piter_fm/detailed_pairs.xlsx
```

---

## 6. Validation-результат рабочей конфигурации

Validation-фильм: `Статус: свободен`, 28 окон, 48 gold-событий.

| Режим | Pred | Jaccard | Precision | Recall | F1 | matched | exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based | 28 | 0.169 | 0.393 | 0.229 | 0.290 | 18 | 6 |
| ML only | 42 | 0.098 | 0.190 | 0.167 | 0.178 | 22 | 6 |
| Combined | 52 | 0.136 | 0.231 | 0.250 | 0.240 | 34 | 8 |

Рабочим выбран combined-режим, потому что он даёт наибольшее число найденных совпадений (`matched=34`) и больший recall, хотя rule-based отдельно имеет более высокий scalar F1.

---

## 7. Структура ожидаемых артефактов

```text
artifacts/
├── validation_status_svoboden_asr_diarization_colab/
│   └── windows/
│       └── val_.../
│           ├── audio.wav
│           ├── transcript.json
│           └── diarization.json
├── validation_status_svoboden_local_postprocess_v11/
│   ├── extracted_pairs.xlsx
│   ├── gold.xlsx
│   ├── run_summary.json
│   ├── rule_lexicon.json
│   ├── character_profiles.json
│   └── windows/
│       └── val_.../
│           ├── utterances.jsonl
│           ├── utterances_named.jsonl
│           ├── speaker_assignments.json
│           ├── predictions.jsonl
│           └── summary.json
├── test_piter_fm_asr_diarization_colab/
│   └── windows/
│       └── chunk_.../
│           ├── audio.wav
│           ├── transcript.json
│           ├── diarization.json
│           └── chunk_info.json
└── test_piter_fm/
    ├── extracted_pairs.xlsx
    ├── detailed_pairs.xlsx
    ├── run_summary.json
    └── windows/
```

---

## 8. Важные замечания по воспроизведению

- В `src/pipeline.py` значения по умолчанию не равны рабочей конфигурации. Для воспроизведения результата нужно явно передавать `--intent-mode combined`, `--ml-confidence-threshold 0.35`, `--similarity-threshold 0.48` и `--diarization-segment-mode regular`.
- При `--skip-asr --skip-diarization` Hugging Face token не нужен, потому что diarization не запускается заново.
- `--diarization-segment-mode regular` важен: он берёт `regular_segments` из `diarization.json`, если они есть.
- `src.evaluate.py` оценивает JSONL-предсказания на уровне обучающих реплик, а Excel-оценка validation-файлов выполняется через ноутбуки.
- Файл `detailed_pairs.xlsx` не создаётся основным pipeline автоматически; его создаёт отдельная команда `src.export_detailed_pairs`.
