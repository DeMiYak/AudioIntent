# Data Format Specification

## 1. SpeakerSegment
Результат диаризации.

Поля:
- `start_time: float`
- `end_time: float`
- `speaker_label: str`

## 2. Utterance
Реплика, собранная из ASR и diarization.

Поля:
- `utterance_id: str`
- `start_time: float`
- `end_time: float`
- `speaker_label: str`
- `speaker_name: str | null`
- `text: str`
- `words: list[dict]`

Каждый элемент `words`:
- `word: str`
- `start_time: float`
- `end_time: float`

## 3. IntentPrediction
Результат модуля извлечения интенций.

Поля:
- `utterance_id: str`
- `expression: str`
- `intent_type: str`
- `char_start: int`
- `char_end: int`
- `token_start: int`
- `token_end: int`
- `confidence: float | null`

Допустимые значения `intent_type`:
- `contact_open`
- `contact_close`

## 4. FinalPrediction
Итоговый результат пайплайна.

Поля:
- `prediction_id: str`
- `start_time: float`
- `end_time: float`
- `speaker_label: str`
- `speaker_name: str`
- `expression: str`
- `intent_type: str`
- `source_utterance: str`
- `confidence: float | null`

## Правила
1. В одной реплике может быть несколько выражений.
2. Каждое выражение превращается в отдельный `FinalPrediction`.
3. Если имя персонажа неизвестно, используется `speaker_name = "unknown"`.
4. Временные метки выражения определяются по словам, входящим в span.
5. Границы выражений в обучающих данных определяются по gold-разметке.