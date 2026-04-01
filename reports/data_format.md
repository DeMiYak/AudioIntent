# Data Format Specification

Этот документ описывает реальные артефакты текущего pipeline для validation и test.

Цель pipeline — извлечение пар вида `speaker - phrase` отдельно для:

* `opening`
* `closing`

Итоговый Excel должен быть совместим с `notebooks/evaluation.ipynb`.

---

## 1. Validation windows

Источник validation-окон:

* `data/raw/gold/data_val.xlsx`
* лист `Вал - Статус свободен`

Каждое окно содержит:

* `row_id` — ID строки из Excel, используется в итоговом Excel для evaluation
* `window_id` — внутренний уникальный идентификатор окна для артефактов pipeline
* `film`
* `start_sec`
* `end_sec`
* `duration_sec`
* `start_time`
* `end_time`
* `context`
* `dialogue_type`
* `annotation`
* `gold_opening`
* `gold_closing`

### Важно

* `row_id` не обязан быть уникальным.
* `window_id` обязан быть уникальным.
* `window_id` используется для имён папок и связи между этапами pipeline.

---

## 2. Gold Excel for evaluation

Файл:

* `gold.xlsx`

Назначение:

* gold-таблица для проверки через `evaluation.ipynb`

Колонки:

* `ID`
* `Фильм`
* `Время начала`
* `Время окончания`
* `opening`
* `closing`

Формат значений:

* одна пара: `Спикер - фраза`
* несколько пар: `Спикер - фраза; Спикер - фраза`

Пример:

* `opening = Никита - алло; Алина - привет`
* `closing = Алина - пока; Никита - давай`

---

## 3. ASR output: `transcript.json`

Файл:

* `windows/<window_id>/transcript.json`

Назначение:

* результат ASR для одного validation/test окна

Структура верхнего уровня:

* `source: str`
* `model_name: str`
* `device: str`
* `compute_type: str`
* `batched_mode: bool`
* `batch_size: int | null`
* `language: str | null`
* `language_probability: float | null`
* `duration_sec: float | null`
* `segments: list[dict]`

### Поля `segment`

Каждый сегмент ASR содержит:

* `id: int | str`
* `start: float`
* `end: float`
* `text: str`
* `words: list[dict]`

### Поля `word`

Каждое слово содержит:

* `word: str`
* `start: float | null`
* `end: float | null`

### Замечания

* `words` могут быть пустыми или неполными.
* downstream должен уметь работать и без word-level timestamps.

---

## 4. Diarization output: `diarization.json`

Файл:

* `windows/<window_id>/diarization.json`

Назначение:

* результат speaker diarization для одного окна

Структура:

* `segments: list[dict]`

Каждый сегмент содержит:

* `start_time: float`
* `end_time: float`
* `speaker_label: str`

Пример структуры:

* `segments[0].start_time = 0.12`
* `segments[0].end_time = 1.98`
* `segments[0].speaker_label = "SPEAKER_00"`

---

## 5. Utterances: `utterances.jsonl`

Файл:

* `windows/<window_id>/utterances.jsonl`

Назначение:

* реплики, собранные из ASR и diarization

Одна строка = один JSON-объект.

Поля:

* `utterance_id: str`
* `speaker_label: str`
* `start_time: float`
* `end_time: float`
* `text: str`

### Замечания

* это промежуточный формат после стыковки ASR и diarization
* на этом этапе имя персонажа ещё может быть неизвестно

---

## 6. Named utterances: `utterances_named.jsonl`

Файл:

* `windows/<window_id>/utterances_named.jsonl`

Назначение:

* utterances после speaker identification

Поля:

* `utterance_id: str`
* `speaker_label: str`
* `speaker_name: str`
* `start_time: float`
* `end_time: float`
* `text: str`

### Правило

Если speaker identification не смог надёжно определить персонажа, используется:

* `speaker_name = "unknown"`

---

## 7. Speaker assignments: `speaker_assignments.json`

Файл:

* `windows/<window_id>/speaker_assignments.json`

Назначение:

* отображение diarization labels в имена персонажей

Типичная структура:

* список или словарь сопоставлений вида:

  * `speaker_label`
  * `speaker_name`
  * служебные поля качества или сходства

Минимальный контракт:

* downstream должен уметь восстановить соответствие `speaker_label -> speaker_name`

---

## 8. Rule-based predictions: `predictions.jsonl`

Файл:

* `windows/<window_id>/predictions.jsonl`

Назначение:

* найденные выражения opening/closing в репликах

Одна строка = одно найденное выражение.

Поля:

* `expression: str`
* `intent_type: str`
* `speaker_name: str`
* `speaker_label: str | null`
* `start_time: float | null`
* `end_time: float | null`
* `char_start: int | null`
* `char_end: int | null`
* `source_utterance: str | null`

Допустимые значения `intent_type`:

* `contact_open`
* `contact_close`

### Важно

* одна реплика может содержать несколько выражений
* каждое выражение записывается как отдельное предсказание

---

## 9. Window summary: `summary.json`

Файл:

* `windows/<window_id>/summary.json`

Назначение:

* краткая техническая сводка по окну

Обычно содержит:

* идентификатор окна
* число ASR segments
* число diarization segments
* число utterances
* число named utterances
* число predictions
* пути к ключевым артефактам
* статус обработки или ошибки

Это технический файл для отладки, а не формат evaluation.

---

## 10. Final Excel: `extracted_pairs.xlsx`

Файл:

* `extracted_pairs.xlsx`

Назначение:

* итоговый prediction-файл для проверки в `evaluation.ipynb`

Колонки:

* `ID`
* `Фильм`
* `Время начала`
* `Время окончания`
* `opening`
* `closing`

Формат колонок `opening` и `closing`:

* одна пара: `Спикер - фраза`
* несколько пар: `Спикер - фраза; Спикер - фраза`
* если предсказаний нет — пустая строка или пустое значение

Пример:

* `opening = Никита - алло; Алина - привет`
* `closing = Алина - пока`

---

## 11. Pair formatting rules

При преобразовании `predictions.jsonl` в `opening` и `closing` используются правила:

1. `contact_open` попадает в колонку `opening`
2. `contact_close` попадает в колонку `closing`
3. каждая запись форматируется как `speaker_name - expression`
4. несколько записей объединяются через `;`
5. порядок внутри окна желательно сохранять по времени:

   * сначала `start_time`
   * затем `char_start`
6. если `speaker_name` неизвестен, допускается строка `unknown - expression`

---

## 12. Source-of-truth by stage

Текущий pipeline устроен так:

1. Excel (`data_val.xlsx`) задаёт validation-окна и gold
2. ASR создаёт `transcript.json`
3. diarization создаёт `diarization.json`
4. utterance builder создаёт `utterances.jsonl`
5. speaker identification создаёт `utterances_named.jsonl`
6. rule-based extraction создаёт `predictions.jsonl`
7. pair formatter создаёт `extracted_pairs.xlsx`

---

## 13. Compatibility requirement

Итоговый pipeline считается корректным, если:

* `gold.xlsx` и `extracted_pairs.xlsx` имеют одинаковый табличный контракт
* `evaluation.ipynb` может читать оба файла без дополнительных преобразований
* колонки `opening` и `closing` содержат строки формата:

  * `Спикер - фраза`
  * несколько пар через `;`

---

## 14. Practical note

Новая постановка задачи оценивает не таймкоды, а только пары:

* `speaker - phrase`

Поэтому временные метки остаются важны для внутренних этапов pipeline, но не являются финальным output для evaluation.
