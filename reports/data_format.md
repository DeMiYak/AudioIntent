# Data Format Specification

Этот документ описывает реальные артефакты текущего pipeline для validation и test.
Цель pipeline — извлечение пар вида:

- `speaker - phrase`

отдельно для:

- `opening`
- `closing`

Итоговый Excel должен быть совместим с `notebooks/evaluation.ipynb`.

---

## 1. Validation windows

Источник validation-окон:
- `data/raw/gold/data_val.xlsx`
- лист: `Вал - Статус свободен`

Каждое окно содержит:

- `row_id` — ID строки из Excel, используется в итоговом Excel для evaluation
- `window_id` — внутренний уникальный идентификатор окна для артефактов pipeline
- `film`
- `start_sec`
- `end_sec`
- `duration_sec`
- `start_time`
- `end_time`
- `context`
- `dialogue_type`
- `annotation`
- `gold_opening`
- `gold_closing`

### Важно
- `row_id` не обязан быть уникальным.
- `window_id` обязан быть уникальным.
- `window_id` используется для имён папок и связи между этапами pipeline.

---

## 2. Gold Excel for evaluation

Файл:
- `gold.xlsx`

Назначение:
- gold-таблица для проверки через `evaluation.ipynb`

Колонки:

- `ID`
- `Фильм`
- `Время начала`
- `Время окончания`
- `opening`
- `closing`

Формат значений:
- одна пара: `Спикер - фраза`
- несколько пар: `Спикер - фраза; Спикер - фраза`

Пример:

```text
opening = Никита - алло; Алина - привет
closing = Алина - пока; Никита - давай