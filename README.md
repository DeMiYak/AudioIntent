# Contact Establishment / Termination Pipeline

Проект для автоматического извлечения выражений установления и прекращения контакта в русскоязычных видеофильмах.

## Текст задачи

### Входные данные
Входные данные предоставлены на втором изображении. К ним относится: 
1. Валидационный фильм: 
a. Видеофайл фильма 
b. Звуковая дорожка 
c. Список персонажей и сэмпл голосов 
d. Список кортежей 
2. Тестовый фильм: 
a. Видеофайл фильма (около 1.5 часов) 
b. Звуковая дорожка 
3. Золотая разметка: 
a. Набор текстовых диалогов, в которых размечены выражения установления контакта и его прекращения. (используется для анализа типичных выражений) 

### Критерии успеха
Условия: 
1. Код был задокументирован и воспроизводим 
2. Был отчёт, в котором описано, что делалось, почему и к чему привело, включая анализ ошибок на тестовом фильме. 
3. Работа идёт с текстом и аудио. 
4. Видео можно использовать только для получения конкретных фичей (например, сделать дополнительный слой разметки о смене сцены). 

### Возможный путь реализации MVP
1. Транскрибация (распознавание речи). Библиотеки: 
- Vosk, 
- DeepSpeech, 
- SpeechRecognition, 
- speechbrain, 
- Kaldi 
2. Диаризация (разделение по говорящим. Получаем метки спикеров: speaker_0, speaker_1,...). Библиотеки: 
- pyannote.audio, 
- speechbrain 
3. Идентификация спикера: Сопоставление меток с именами персонажей из списка. Для каждого эталонного голоса вычисляем эмбеддинг, затем сравниваем с эмбеддингами обнаруженных спикеров через косинусное расстояние Библиотеки: 
- resemblyzer, 
- speechbrain, 
- faiss 
4. Классификация интенций: Определение, является ли выражение контактоустанавливающим или контактозавершающим 
- Вариант А (правила): поиск ключевых фраз из золотой разметки. 
- Вариант Б (ML): обучение лёгкого классификатора на золотых диалогах/выражениях 
5. Формирование выходных данных. Для каждого найденного выражения сохраняем: 
- временную метку, 
- текст, 
- говорящего, 
- тип интенции

## Цель
Для заданного фильма получить набор кортежей:
(start_time, end_time, speaker_name, expression, intent_type)

## Основные модули
- ASR
- diarization
- speaker identification
- intent extraction
- evaluation

## Структура проекта
- `src/` — исходный код
- `data/` — данные
- `configs/` — конфиги
- `models/` — обученные модели
- `reports/` — отчёт и анализ
- `artifacts/` — результаты запусков

## Ближайший план
1. Подготовка gold-разметки
2. Rule-based baseline
3. ASR + diarization
4. ML intent extraction
5. Финальный pipeline

## Запуск кода
.\venv311\Scripts\activate


## Этап 1

Собрать репозиторий и зафиксировать формат результата.

## Этап 2. Подготовка золотой разметки

### Что делается на этом этапе

На данном этапе исходная золотая разметка приводится к единому машинно-обрабатываемому формату.  
Входом служит Excel-файл, содержащий список диалогов, где:

- каждая строка таблицы соответствует одному диалогу;
- у диалога есть идентификатор `id`;
- текст диалога хранится в колонке `marked_text`;
- контактоустанавливающие выражения заключены в теги `<opening>...</opening>`;
- контактозавершающие выражения заключены в теги `<closing>...</closing>`.

Пример разметки:

```text
Андрей Васильевич Пряников: <opening>Родионова</opening>. Хорошо, что тебя встретил.
Виктория Родионова: <opening>Здравия желаю</opening>, товарищ подполковник.
Андрей Васильевич Пряников: Значит так, тебя там в отделе новенький дожидается.
```

### Как запускать

```bash
python -m src.preprocess_gold \
  --input data/raw/gold/gold_dialogues.xlsx \
  --output data/processed/gold_dialogues.jsonl \
  --stats-output data/processed/gold_stats.json \
  --skipped-output data/processed/gold_skipped_lines.json \
  --sheet-name 0 \
  --id-col id \
  --text-col marked_text
```

## Этап 3. Rule-based baseline

### Что делается на этом этапе

На данном этапе строится первый рабочий baseline для извлечения контактоустанавливающих и контактозавершающих выражений.

В качестве источника правил используется золотая разметка, подготовленная на предыдущем этапе.  
Из неё автоматически извлекаются выражения, размеченные как:

- `contact_open`
- `contact_close`

После этого формируется словарь baseline, который затем применяется к репликам в формате `gold_dialogues.jsonl` или к любым другим данным, приведённым к той же структуре.

### Почему выбран именно такой baseline

Rule-based baseline нужен по нескольким причинам:

1. Он задаёт первую прозрачную и интерпретируемую точку отсчёта.  
   До построения ML-модели важно иметь простое решение, которое можно легко проверить и объяснить.

2. Он позволяет быстро понять, насколько типовые контактоустанавливающие и контактозавершающие формулы вообще покрываются простым словарным поиском.

3. Он полезен как технический baseline для будущего сравнения с ML-моделью.  
   На более позднем этапе можно будет показать, в каких случаях rule-based подход работает хорошо, а в каких — начинает ошибаться.

### Как устроен baseline

Rule-based baseline строится в два шага:

1. Из золотой разметки извлекается словарь выражений по двум классам:
   - `contact_open`
   - `contact_close`

2. Для каждой реплики выполняется поиск этих выражений по тексту с помощью регулярных выражений.

Дополнительно в baseline реализованы:

- нормализация текста для сопоставления (`lowercase`, `ё -> е`);
- приоритет более длинных выражений;
- разрешение конфликтующих пересечений между найденными span’ами;
- вычисление как символьных, так и токенных границ найденного выражения.

### Что считается входом и выходом

#### Вход

Основной входной файл:

`data/processed/gold_dialogues.jsonl`

Каждая строка содержит:
- `dialogue_id`
- `utterance_id`
- `speaker_name`
- `text`
- `annotations`

#### Выход

На выходе baseline сохраняет:

1. `data/processed/rule_lexicon.json` — словарь выражений, извлечённых из gold-разметки;
2. `artifacts/rule_based_predictions.jsonl` — предсказания baseline;
3. `artifacts/rule_based_metrics.json` — метрики качества.

Пример одной записи предсказания:

```json
{
  "dialogue_id": "dlg_001",
  "utterance_id": "dlg_001_utt_002",
  "speaker_name": "Виктория Родионова",
  "source_text": "Здравия желаю, товарищ подполковник.",
  "expression": "Здравия желаю",
  "intent_type": "contact_open",
  "char_start": 0,
  "char_end": 14,
  "token_start": 0,
  "token_end": 1,
  "confidence": 1.0,
  "rule_expression": "здравия желаю",
  "rule_frequency": 1
}
```


---

### Как запускать

```bash
python -m src.rule_based_intent \
  --fit-input data/processed/gold_dialogues.jsonl \
  --predict-input data/processed/gold_dialogues.jsonl \
  --lexicon-output data/processed/rule_lexicon.json \
  --predictions-output artifacts/rule_based_predictions.jsonl \
  --metrics-output artifacts/rule_based_metrics.json \
  --min-freq 1
```

## Этап 4. Подъём ASR + diarization на коротком фрагменте фильма

Важно иметь ffmpeg утилиту для pyannote.audio

### Что делается на этом этапе

На данном этапе поднимается базовый аудио-пайплайн на ограниченном фрагменте валидационного фильма длительностью 5–10 минут.

Этап включает две основные подзадачи:

1. **ASR (Automatic Speech Recognition)**  
   Из выбранного фрагмента фильма извлекается аудио, после чего выполняется автоматическая транскрибация речи с временными метками.

2. **Speaker diarization**  
   Для того же аудиофрагмента определяется, какие интервалы принадлежат каким говорящим. На выходе получаются анонимные speaker labels вида:
   - `SPEAKER_00`
   - `SPEAKER_01`
   - `SPEAKER_02`
   и т.д.

### Почему этап ограничен 5–10 минутами

Полный фильм слишком велик для самого первого запуска и затрудняет отладку.

Поэтому на данном этапе используется короткий фрагмент валидационного фильма. Это позволяет:

- быстро проверить, что выбранный стек вообще работает на реальных данных;
- не тратить лишнее время на полный прогон ещё до стабилизации кода;
- увидеть структуру промежуточных артефактов;
- локализовать ошибки на ранней стадии.

Таким образом, задача этапа — не обработать весь фильм целиком, а убедиться, что базовая ASR/diarization-цепочка работает корректно и воспроизводимо.

### Почему выбрана именно такая реализация

Этап 4 строится вокруг следующей логики:

1. **Сначала аудио приводится к единому формату**  
   Из видео или исходной звуковой дорожки извлекается WAV-файл в формате:
   - mono
   - 16 kHz

   Это делается для того, чтобы и ASR, и diarization работали на одном и том же стабильном входе, а промежуточные результаты можно было переиспользовать дальше.

2. **ASR и diarization выполняются раздельно**  
   Несмотря на то, что некоторые инструменты умеют совмещать эти шаги, для проекта удобнее хранить их как отдельные независимые модули. Это упрощает:
   - отладку,
   - сохранение промежуточных артефактов,
   - анализ ошибок,
   - повторный запуск отдельных частей пайплайна.

3. **Результаты сохраняются в `data/interim/`**  
   Это позволяет использовать их на следующих этапах без повторного пересчёта.

### Что считается входом

На этапе 4 можно использовать либо видеофайл, либо готовую звуковую дорожку.

Типовой вход:
- `data/raw/validation/film.mp4`
или
- `data/raw/validation/audio.wav`

### Что считается результатом

Результатом этапа являются два промежуточных артефакта:

1. **Подготовленный WAV-файл**
   - например, `data/interim/validation_sample.wav`

2. **ASR-результат**
   - например, `data/interim/asr_validation_sample.json`

3. **Diarization-результат**
   - например, `data/interim/diarization_validation_sample.json`

#### Пример структуры ASR-результата

```json
{
  "audio_path": "data/interim/validation_sample.wav",
  "model_name": "medium",
  "device": "cuda",
  "compute_type": "float16",
  "language": "ru",
  "alignment_used": true,
  "segments": [
    {
      "segment_id": "seg_0001",
      "start_time": 0.52,
      "end_time": 3.14,
      "text": "Здравствуйте, товарищ подполковник.",
      "words": [
        {
          "word": "Здравствуйте",
          "start_time": 0.52,
          "end_time": 1.31,
          "score": 0.98
        }
      ]
    }
  ]
}
```

### Как запускать

#### Подготовка аудио + ASR
```bash
python -m src.asr \
  --media-input data/raw/validation/film.mp4 \
  --prepared-audio-output data/interim/validation_sample.wav \
  --transcript-output data/interim/asr_validation_sample.json \
  --start-sec 0 \
  --duration-sec 600 \
  --model-name medium \
  --device auto \
  --compute-type auto \
  --batch-size 8
```

#### Speaker Diarization
```bash
python -m src.diarization \
  --audio-input data/interim/validation_sample.wav \
  --output data/interim/diarization_validation_sample.json \
  --device auto
```

## Этап 5. Сборка utterances из ASR и diarization

### Что делается на этом этапе

На данном этапе результаты автоматической транскрибации речи и speaker diarization объединяются в единую структуру реплик, далее называемую `utterances`.

Если на этапе 4 были получены:

- ASR-сегменты с текстом и временными метками;
- diarization-сегменты с метками говорящих;

то на этапе 5 эти два источника информации синхронизируются и преобразуются в последовательность реплик следующего вида:

```json
{
  "utterance_id": "utt_0001",
  "start_time": 12.52,
  "end_time": 15.30,
  "speaker_label": "SPEAKER_00",
  "speaker_name": null,
  "text": "Здравствуйте, товарищ подполковник.",
  "words": [
    {
      "word": "Здравствуйте",
      "start_time": 12.52,
      "end_time": 13.10
    }
  ]
}
```


---

### Как запускать

```bash
python -m src.utterance_builder \
  --asr-input data/interim/asr_validation_sample.json \
  --diarization-input data/interim/diarization_validation_sample.json \
  --utterances-output data/processed/utterances_validation_sample.jsonl \
  --stats-output data/processed/utterances_validation_sample_stats.json \
  --max-pause-sec 1.0 \
  --unknown-speaker-label unknown_speaker \
  --max-nonoverlap-assign-distance-sec 1.0
```

## Этап 6. Speaker identification

### Что делается на этом этапе

На данном этапе анонимные `speaker_label`, полученные после diarization, сопоставляются с именами персонажей из входных данных.

Если после этапа 5 каждая реплика (`utterance`) уже содержит:
- текст,
- временной интервал,
- `speaker_label`,

то на этапе 6 в эту структуру добавляется:
- `speaker_name`,
- при необходимости — численная мера уверенности сопоставления.

Таким образом, реплика из вида:

```json
{
  "utterance_id": "utt_0001",
  "start_time": 12.52,
  "end_time": 15.30,
  "speaker_label": "SPEAKER_00",
  "speaker_name": null,
  "text": "Здравствуйте, товарищ подполковник."
}
```


---

### Как запускать

```bash
python -m src.speaker_id \
  --audio-input data/interim/validation_sample.wav \
  --utterances-input data/processed/utterances_validation_sample.jsonl \
  --samples-dir data/raw/samples \
  --utterances-output data/processed/utterances_validation_sample_with_speakers.jsonl \
  --mapping-output artifacts/speaker_mapping_validation_sample.json \
  --stats-output artifacts/speaker_id_validation_sample_stats.json \
  --similarity-threshold 0.65 \
  --min-sample-duration-sec 0.5 \
  --min-utterance-duration-sec 0.7 \
  --min-total-duration-sec 1.5 \
  --max-total-duration-sec 45.0
```
## Шаг 7

Собрать end-to-end baseline pipeline.

## Шаг 8

Обучить ML-модель intent extraction в Kaggle.

## Шаг 9

Сравнить baseline и ML.

## Шаг 10

Интегрировать ML в pipeline.

## Шаг 11

Прогнать на тестовом фильме.

## Шаг 12

Сделать анализ ошибок и отчёт.