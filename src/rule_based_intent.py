from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Группы маркеров высокой точности.
# Сильные маркеры могут существовать как компактные фразы; слабые маркерам обычно нужен контекст.
OPEN_STRONG = {
    "алло",
    "алло!",
    "здравствуй",
    "здравствуйте",
    "добрый день",
    "добрый вечер",
    "привет",
    "извините",
    "можно",
    "можно?",
}

OPEN_WEAK = {
    "а",
    "да",
    "ну",
    "слушай",
    "смотри",
    "это",
    "что",
    "кто",
}

CLOSE_STRONG = {
    "до свидания",
    "пока",
    "прощай",
    "спасибо",
    "чао",
    "счастливо",
}

CLOSE_WEAK = {
    "все",
    "ладно",
    "ну все",
    "давай",
    "сейчас",
}

OPEN_MANUAL_RULES = {
    "where_have_i_seen_you",
    "would_you_like_anything",
    "can_i_join_you_for_a_drink",
    "is_that_you_nikita",
    # Общие русские приветствия (лексикон может их пропустить)
    "zdravstvuy_generic",
    "allo_generic",
    "privetstvuyu",
    "dobro_pozhalovat",
    "lets_get_acquainted",
}

CLOSE_MANUAL_RULES = {
    "withdraw_from_competition",
    "do_not_disturb",
    "i_am_leaving",
    "call_me_later",
    "be_there_soon",
    "wait_at_time",
    "bye_chao",
    # Общие русские прощания (лексикон может их пропустить)
    "nu_poka",
    "nu_vse",
    "do_vstrechi",
    "uvidimsya",
    "vsego_dobrogo",
    "poka_poka",
    "do_svidaniya_generic",
    "davay_poka",
    "schastlivo_generic",
    "udachi_generic",
}

# ---------------------------------------------------------------------------
# Трёхуровневая система надёжности manual rules.
#
# Tier A — однозначные, не требуют подтверждения ML.
# Tier B — контекстно-зависимые, рекомендуется ML-подтверждение.
# Tier C — слабые / опасные паттерны, обязательно ML-подтверждение.
# ---------------------------------------------------------------------------

MANUAL_RULES_TIER_A: frozenset[str] = frozenset({
    # Специфичные для фильма (однозначный смысл)
    "where_have_i_seen_you",
    "would_you_like_anything",
    "can_i_join_you_for_a_drink",
    "is_that_you_nikita",
    "withdraw_from_competition",
    "do_not_disturb",
    "i_am_leaving",
    "call_me_later",
    "be_there_soon",
    "wait_at_time",
    # Однозначные приветствия
    "lets_get_acquainted",
    "privetstvuyu",
    "dobro_pozhalovat",
    # Однозначные прощания
    "do_svidaniya_generic",
    "vsego_dobrogo",
    "uvidimsya",
    "poka_poka",
    "bye_chao",
    "do_vstrechi",
})

MANUAL_RULES_TIER_B: frozenset[str] = frozenset({
    # Контекстно-зависимые — «алло» может быть и приветствием, и просто возгласом
    "allo_generic",
    # «Здравствуй» чёткое слово, но без контекста иногда неоднозначно
    "zdravstvuy_generic",
    # «Ну, пока» и «счастливо» — прощания, но встречаются в нейтральных контекстах
    "nu_poka",
    "schastlivo_generic",
})

MANUAL_RULES_TIER_C: frozenset[str] = frozenset({
    # Опасные слабые паттерны — высокий риск false positives
    "nu_vse",       # «ну, всё» слишком часто вне контекста прощания
    "davay_poka",   # «давай, пока» — правдоподобное прощание, но «давай» шумный маркер
    "udachi_generic",  # «удачи» — частое пожелание вне прощания
})

# Эти маркеры часто появляются в лексиконе из-за шумных / слишком узких gold-спанов.
# Считать их крайне неоднозначными при использовании как самостоятельных opening-правил.
AMBIGUOUS_OPEN_RULES = {
    "а",
    "да",
    "ну",
    "все",
    "так",
    "что",
    "нас",
    "таня",
}

AMBIGUOUS_CLOSE_RULES: set[str] = set()

# Ручные паттерны для валидационных случаев, где лексикон обычно слишком узок.
MANUAL_PATTERNS: list[dict[str, Any]] = [
    {
        "intent_type": "contact_open",
        "name": "lets_get_acquainted",
        "pattern": re.compile(
            r"(?<!\w)(?:давайте|давай)\s+познакомимся|(?:давайте|давай)\s+знакомиться|познакомимся(?=[?.!,\s]|$)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_open",
        "name": "where_have_i_seen_you",
        "pattern": re.compile(
            r"(?<!\w)где\s+я\s+тебя\s+видел(?=[?.!,]|$)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_open",
        "name": "would_you_like_anything",
        "pattern": re.compile(
            r"(?<!\w)желаете\s+что\s*[-—–]?\s*нибудь(?=[?.!,]|$)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_open",
        "name": "can_i_join_you_for_a_drink",
        "pattern": re.compile(
            r"(?<!\w)можно\s+с\s+вами\s+выпить(?=[?.!,]|$)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_open",
        "name": "is_that_you_nikita",
        "pattern": re.compile(
            r"(?<!\w)это\s+вы\s*[—-]?\s*никита(?=[?.!,]|$)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "withdraw_from_competition",
        "pattern": re.compile(
            r"(?<!\w)мы\s+снимаемся\s+с\s+соревнований(?=[?.!,]|$)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "do_not_disturb",
        "pattern": re.compile(
            r"(?<!\w)не\s+буду\s+вам\s+мешать(?=[?.!,]|$)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "i_am_leaving",
        "pattern": re.compile(
            r"(?<!\w)я\s+пош[её]л(?=[?.!,]|$)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "call_me_later",
        "pattern": re.compile(
            r"(?<!\w)надумаешь\s*[-—–?]\s*[зс]вони(?=[?.!,]|$)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "be_there_soon",
        "pattern": re.compile(
            r"(?<!\w)буду\s+через\s+полчаса(?=[?.!,]|$)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "wait_at_time",
        "pattern": re.compile(
            r"(?<!\w)жду\s+тебя\s+в\s+(?:\d+|десять|10)(?=[?.!,]|$)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "bye_chao",
        "pattern": re.compile(
            r"(?<!\w)чао(?:\s*,\s*[а-яёa-z-]+)?(?=[?.!,]|$)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    # ----------------------------------------------------------------
    # Generic Russian farewells — добавлены для повышения recall closing
    # ----------------------------------------------------------------
    {
        "intent_type": "contact_close",
        "name": "nu_poka",
        "pattern": re.compile(
            r"(?<!\w)ну[,\s]+пока(?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "nu_vse",
        "pattern": re.compile(
            r"(?<!\w)ну[,\s]+вс[её](?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "do_vstrechi",
        "pattern": re.compile(
            r"(?<!\w)до\s+встречи(?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "uvidimsya",
        "pattern": re.compile(
            r"(?<!\w)увидимся(?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "vsego_dobrogo",
        "pattern": re.compile(
            r"(?<!\w)всего\s+доброго(?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "poka_poka",
        "pattern": re.compile(
            r"(?<!\w)пока[-\s]*пока(?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "do_svidaniya_generic",
        "pattern": re.compile(
            r"(?<!\w)до\s+свидания(?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "davay_poka",
        "pattern": re.compile(
            r"(?<!\w)давай[,\s]+пока(?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "schastlivo_generic",
        "pattern": re.compile(
            r"(?<!\w)счастливо(?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_close",
        "name": "udachi_generic",
        "pattern": re.compile(
            r"(?<!\w)удачи(?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    # ----------------------------------------------------------------
    # Generic Russian greetings — для повышения recall opening
    # ----------------------------------------------------------------
    {
        "intent_type": "contact_open",
        "name": "zdravstvuy_generic",
        "pattern": re.compile(
            r"(?<!\w)здравствуй(?:те)?(?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_open",
        "name": "allo_generic",
        "pattern": re.compile(
            r"(?<!\w)алл[оa](?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_open",
        "name": "privetstvuyu",
        "pattern": re.compile(
            r"(?<!\w)приветствую(?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
    {
        "intent_type": "contact_open",
        "name": "dobro_pozhalovat",
        "pattern": re.compile(
            r"(?<!\w)добро\s+пожаловать(?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    },
]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Загружает JSONL-файл в список словарей.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records



def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """
    Сохраняет список словарей в JSONL.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



def save_json(data: dict[str, Any] | list[dict[str, Any]], path: str | Path) -> None:
    """
    Сохраняет объект в JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def normalize_for_matching(text: str) -> str:
    """
    Нормализация текста для сопоставления:
    - lowercase
    - ё -> е

    Важно: длина строки не меняется, поэтому char-индексы остаются валидными.
    """
    return text.lower().replace("ё", "е")



def tokenize_text(text: str) -> list[str]:
    """
    Простая токенизация:
    - слова
    - знаки препинания отдельно
    """
    return TOKEN_PATTERN.findall(text)



def char_to_token_spans(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """
    Для каждого токена вычисляет span (char_start, char_end) в исходном тексте.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0

    for token in tokens:
        match = re.search(re.escape(token), text[cursor:])
        if not match:
            raise ValueError(
                f"Не удалось найти токен '{token}' в тексте: {text} (cursor={cursor})"
            )
        start = cursor + match.start()
        end = start + len(token)
        spans.append((start, end))
        cursor = end

    return spans



def char_span_to_token_span(
    text: str,
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    """
    Переводит символьный span в span по индексам токенов.
    """
    tokens = tokenize_text(text)
    token_spans = char_to_token_spans(text, tokens)

    covered_tokens: list[int] = []

    for token_idx, (tok_start, tok_end) in enumerate(token_spans):
        overlap = not (tok_end <= char_start or tok_start >= char_end)
        if overlap:
            covered_tokens.append(token_idx)

    if not covered_tokens:
        return -1, -1

    return covered_tokens[0], covered_tokens[-1]



def build_expression_regex(expression_text: str) -> re.Pattern[str]:
    """
    Строит regex для поиска выражения в тексте.

    Особенности:
    - матчинг идёт по нормализованному тексту
    - пробелы допускают вариативность через \\s+
    - дефисы допускают вариативность через \\s*[-—–]?\\s*
    - по краям добавляются word-boundary-подобные ограничения, если это нужно
    """
    expr = normalize_for_matching(expression_text).strip()
    if not expr:
        raise ValueError("Пустое выражение нельзя превратить в regex.")

    parts: list[str] = []
    idx = 0
    while idx < len(expr):
        ch = expr[idx]
        if ch.isspace():
            while idx < len(expr) and expr[idx].isspace():
                idx += 1
            parts.append(r"\s+")
            continue
        if ch in "-—–":
            while idx < len(expr) and expr[idx] in "-—– ":
                idx += 1
            parts.append(r"\s*[-—–]?\s*")
            continue
        parts.append(re.escape(ch))
        idx += 1

    body = "".join(parts)

    non_space_chars = [ch for ch in expr if not ch.isspace()]
    first_char = non_space_chars[0] if non_space_chars else ""
    last_char = non_space_chars[-1] if non_space_chars else ""

    if first_char and (first_char.isalnum() or first_char == "_"):
        body = rf"(?<!\w){body}"

    if last_char and (last_char.isalnum() or last_char == "_"):
        body = rf"{body}(?!\w)"

    return re.compile(body, re.UNICODE)



def split_into_sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0

    for match in re.finditer(r"[.!?]+", text):
        end = match.end()
        spans.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1

    if start < len(text):
        spans.append((start, len(text)))

    return spans or [(0, len(text))]



def _trim_span(text: str, start: int, end: int) -> tuple[int, int, str]:
    while start < end and text[start] in " \t\n,;:-—–":
        start += 1
    while end > start and text[end - 1] in " \t\n,;:-—–.!?":
        end -= 1
    return start, end, text[start:end]



def _split_clause_bounds(sentence_text: str, local_start: int, local_end: int) -> tuple[int, int]:
    left = local_start
    right = local_end

    separators = set(",;:")

    while left > 0:
        prev = sentence_text[left - 1]
        if prev in separators:
            break
        left -= 1

    while right < len(sentence_text):
        curr = sentence_text[right]
        if curr in separators:
            break
        right += 1

    return left, right



def expand_match_to_phrase(
    text: str,
    char_start: int,
    char_end: int,
    rule_expression: str,
) -> tuple[int, int, str]:
    """
    Расширяет матч до более осмысленной фразы.

    Приоритет:
    1. manual pattern span, если он пересекается с матчем
    2. компактный span для strong-маркеров
    3. клауза внутри предложения
    4. всё предложение, если оно короткое
    5. fallback на исходный матч
    """
    normalized_rule = normalize_for_matching(rule_expression)

    for pattern_info in MANUAL_PATTERNS:
        for match in pattern_info["pattern"].finditer(text):
            m_start, m_end = match.span()
            if not (m_end <= char_start or m_start >= char_end):
                return _trim_span(text, m_start, m_end)

    if normalized_rule in OPEN_STRONG or normalized_rule in CLOSE_STRONG:
        return _trim_span(text, char_start, char_end)

    sentence_spans = split_into_sentence_spans(text)
    sent_start, sent_end = 0, len(text)
    for s_start, s_end in sentence_spans:
        if s_start <= char_start < s_end:
            sent_start, sent_end = s_start, s_end
            break

    sentence_text = text[sent_start:sent_end]
    local_start = max(0, char_start - sent_start)
    local_end = max(local_start, min(len(sentence_text), char_end - sent_start))

    clause_local_start, clause_local_end = _split_clause_bounds(
        sentence_text, local_start, local_end
    )
    clause_start = sent_start + clause_local_start
    clause_end = sent_start + clause_local_end
    clause_start, clause_end, clause_text = _trim_span(text, clause_start, clause_end)

    # Предпочесть клаузу, если она не слишком длинная и заметно больше маркера.
    clause_tokens = tokenize_text(clause_text)
    match_tokens = tokenize_text(text[char_start:char_end])
    if 2 <= len(clause_tokens) <= 12 and len(clause_tokens) > len(match_tokens):
        return clause_start, clause_end, clause_text

    sentence_start, sentence_end, sentence_clean = _trim_span(text, sent_start, sent_end)
    if 2 <= len(tokenize_text(sentence_clean)) <= 10:
        return sentence_start, sentence_end, sentence_clean

    return _trim_span(text, char_start, char_end)



def extract_lexicon_from_gold(
    gold_records: list[dict[str, Any]],
    min_freq: int = 1,
) -> dict[str, Any]:
    """
    Из gold-разметки строит словарь выражений для rule-based baseline.
    """
    counter_by_intent: dict[str, Counter[str]] = {
        "contact_open": Counter(),
        "contact_close": Counter(),
    }

    for record in gold_records:
        annotations = record.get("annotations", [])
        for ann in annotations:
            expression_text = str(ann["expression_text"]).strip()
            intent_type = ann["intent_type"]

            if not expression_text:
                continue
            if intent_type not in counter_by_intent:
                continue

            normalized_expr = normalize_for_matching(expression_text)
            counter_by_intent[intent_type][normalized_expr] += 1

    lexicon: dict[str, Any] = {
        "contact_open": [],
        "contact_close": [],
    }

    for intent_type, counter in counter_by_intent.items():
        items = [
            {"expression_text": expr, "frequency": freq}
            for expr, freq in counter.items()
            if freq >= min_freq
        ]
        items.sort(
            key=lambda x: (-len(x["expression_text"]), -x["frequency"], x["expression_text"])
        )
        lexicon[intent_type] = items

    return lexicon



def compile_lexicon(lexicon: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Компилирует словарь выражений в regex-паттерны для быстрого поиска.
    """
    compiled: dict[str, list[dict[str, Any]]] = {
        "contact_open": [],
        "contact_close": [],
    }

    for intent_type, items in lexicon.items():
        compiled_items: list[dict[str, Any]] = []

        for item in items:
            expr = item["expression_text"]
            freq = int(item["frequency"])
            pattern = build_expression_regex(expr)

            compiled_items.append(
                {
                    "expression_text": expr,
                    "frequency": freq,
                    "pattern": pattern,
                }
            )

        compiled[intent_type] = compiled_items

    return compiled



def collect_candidates_for_text(
    dialogue_id: str,
    utterance_id: str,
    speaker_name: str,
    text: str,
    compiled_lexicon: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Ищет все возможные rule-based совпадения в реплике.
    """
    normalized_text = normalize_for_matching(text)
    candidates: list[dict[str, Any]] = []

    for intent_type, items in compiled_lexicon.items():
        for item in items:
            expr = item["expression_text"]
            freq = item["frequency"]
            pattern = item["pattern"]

            for match in pattern.finditer(normalized_text):
                char_start, char_end = match.span()
                matched_text = text[char_start:char_end]
                token_start, token_end = char_span_to_token_span(text, char_start, char_end)

                candidates.append(
                    {
                        "dialogue_id": dialogue_id,
                        "utterance_id": utterance_id,
                        "speaker_name": speaker_name,
                        "source_text": text,
                        "matched_text": matched_text,
                        "expression": matched_text,
                        "intent_type": intent_type,
                        "char_start": char_start,
                        "char_end": char_end,
                        "token_start": token_start,
                        "token_end": token_end,
                        "confidence": 1.0,
                        "rule_expression": expr,
                        "rule_frequency": freq,
                    }
                )

    for pattern_info in MANUAL_PATTERNS:
        intent_type = pattern_info["intent_type"]
        for match in pattern_info["pattern"].finditer(text):
            char_start, char_end = match.span()
            token_start, token_end = char_span_to_token_span(text, char_start, char_end)
            matched_text = text[char_start:char_end]
            candidates.append(
                {
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                    "speaker_name": speaker_name,
                    "source_text": text,
                    "matched_text": matched_text,
                    "expression": matched_text,
                    "intent_type": intent_type,
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_start": token_start,
                    "token_end": token_end,
                    "confidence": 1.0,
                    "rule_expression": pattern_info["name"],
                    "rule_frequency": 999,
                }
            )

    return candidates



def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """
    Проверяет пересечение двух span'ов.
    """
    return not (a_end <= b_start or a_start >= b_end)



def score_candidate(candidate: dict[str, Any]) -> float:
    """
    Скоринг кандидата с учётом:
    - силы маркера / manual pattern
    - длины фразы
    - позиции внутри окна
    - штрафов за шумные однословные opening-правила
    """
    rule_expression = normalize_for_matching(str(candidate.get("rule_expression", "")))
    intent_type = str(candidate.get("intent_type", ""))
    phrase = normalize_for_matching(str(candidate.get("expression", "")))
    phrase_token_len = len([tok for tok in tokenize_text(phrase) if re.search(r"\w", tok)])

    score = 0.0

    if intent_type == "contact_open":
        if rule_expression in OPEN_STRONG:
            score += 3.0
        if rule_expression in OPEN_WEAK:
            score -= 2.0
        if rule_expression in OPEN_MANUAL_RULES:
            score += 2.5
        if rule_expression in AMBIGUOUS_OPEN_RULES:
            score -= 2.5
    elif intent_type == "contact_close":
        if rule_expression in CLOSE_STRONG:
            score += 3.0
        if rule_expression in CLOSE_WEAK:
            score -= 1.0
        if rule_expression in CLOSE_MANUAL_RULES:
            score += 2.5
        if rule_expression in AMBIGUOUS_CLOSE_RULES:
            score -= 2.0

    if phrase_token_len == 1 and rule_expression not in OPEN_STRONG and rule_expression not in CLOSE_STRONG:
        score -= 1.5
    elif phrase_token_len == 2:
        score += 0.25
    elif phrase_token_len >= 3:
        score += 1.0

    if intent_type == "contact_open":
        if any(marker in phrase for marker in ["здрав", "добрый", "привет", "алло", "можно", "желаете"]):
            score += 0.75
        if "?" in str(candidate.get("source_text", ""))[int(candidate.get("char_start", 0)):int(candidate.get("char_end", 0)) + 1]:
            score += 0.25
    elif intent_type == "contact_close":
        if any(marker in phrase for marker in ["чао", "спасибо", "жду", "буду", "не буду", "снимаемся", "звони", "до свидания", "пока"]):
            score += 0.75

    start_time = candidate.get("start_time")
    source_start_sec = candidate.get("source_start_sec")
    source_end_sec = candidate.get("source_end_sec")
    if start_time is not None and source_start_sec is not None and source_end_sec is not None:
        try:
            duration = float(source_end_sec) - float(source_start_sec)
            rel = (float(start_time) - float(source_start_sec)) / duration if duration > 0 else None
        except (TypeError, ValueError):
            rel = None
        if rel is not None:
            candidate["relative_position"] = round(rel, 4)
            if intent_type == "contact_open":
                if rel <= 0.30:
                    score += 1.5
                elif rel <= 0.55:
                    score += 0.5
                else:
                    score -= 1.75
            elif intent_type == "contact_close":
                if rel >= 0.70:
                    score += 1.5
                elif rel >= 0.45:
                    score += 0.5
                else:
                    score -= 0.5

    candidate["score"] = round(score, 4)
    return score


def acceptance_threshold(candidate: dict[str, Any]) -> float:
    rule_expression = normalize_for_matching(str(candidate.get("rule_expression", "")))
    intent_type = str(candidate.get("intent_type", ""))
    phrase = normalize_for_matching(str(candidate.get("expression", "")))
    phrase_token_len = len([tok for tok in tokenize_text(phrase) if re.search(r"\w", tok)])

    # Вычисляем относительную позицию, если есть временна́я информация.
    rel = candidate.get("relative_position")

    if intent_type == "contact_close" and rule_expression in CLOSE_MANUAL_RULES:
        # Дополнительно снижаем порог для кандидатов ближе к концу окна.
        if rel is not None and rel >= 0.60:
            return 0.0
        return 0.5
    if intent_type == "contact_open" and rule_expression in OPEN_MANUAL_RULES:
        # Снижаем порог для кандидатов ближе к началу окна.
        if rel is not None and rel <= 0.40:
            return 0.5
        return 1.0
    if intent_type == "contact_close" and (rule_expression in CLOSE_STRONG or phrase_token_len >= 2):
        return 0.75
    if intent_type == "contact_open" and phrase_token_len >= 3:
        return 1.0
    return 1.25



def resolve_overlaps(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Удаляет конфликтующие пересекающиеся совпадения.

    Важный нюанс: opening и closing могут частично пересекаться в одной и той же
    реплике. Поэтому жёстко конкурируют только кандидаты одного intent_type.
    """
    if not candidates:
        return []

    sorted_candidates = sorted(
        candidates,
        key=lambda x: (
            -float(x.get("score", 0.0)),
            -(int(x["char_end"]) - int(x["char_start"])),
            -int(x.get("rule_frequency", 0)),
            int(x["char_start"]),
            str(x["intent_type"]),
        ),
    )

    selected: list[dict[str, Any]] = []

    for cand in sorted_candidates:
        has_overlap = any(
            str(cand.get("intent_type")) == str(chosen.get("intent_type"))
            and spans_overlap(
                int(cand["char_start"]),
                int(cand["char_end"]),
                int(chosen["char_start"]),
                int(chosen["char_end"]),
            )
            for chosen in selected
        )
        if not has_overlap:
            selected.append(cand)

    selected.sort(key=lambda x: (int(x["char_start"]), int(x["char_end"]), str(x.get("intent_type", ""))))
    return selected



def predict_for_record(
    record: dict[str, Any],
    compiled_lexicon: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Запускает rule-based baseline на одной реплике.

    Дополнительно протаскивает полезный контекст из record в итоговые предсказания:
    - speaker_label
    - start_time / end_time
    - source_window_id / source_window_row_id
    """
    dialogue_id = record.get("dialogue_id", "")
    utterance_id = record.get("utterance_id", "")
    speaker_name = record.get("speaker_name", "unknown")
    text = record.get("text", "")

    candidates = collect_candidates_for_text(
        dialogue_id=dialogue_id,
        utterance_id=utterance_id,
        speaker_name=speaker_name,
        text=text,
        compiled_lexicon=compiled_lexicon,
    )

    passthrough_fields = [
        "speaker_label",
        "start_time",
        "end_time",
        "source_window_id",
        "source_window_row_id",
        "source_film",
        "source_start_sec",
        "source_end_sec",
    ]

    expanded: list[dict[str, Any]] = []
    for cand in candidates:
        item = dict(cand)
        for field in passthrough_fields:
            if field in record:
                item[field] = record[field]

        p_start, p_end, phrase_text = expand_match_to_phrase(
            text=text,
            char_start=int(item["char_start"]),
            char_end=int(item["char_end"]),
            rule_expression=str(item["rule_expression"]),
        )
        item["char_start"] = p_start
        item["char_end"] = p_end
        item["expression"] = phrase_text
        item["token_start"], item["token_end"] = char_span_to_token_span(text, p_start, p_end)
        score_candidate(item)
        expanded.append(item)

    # Схлопнуть точные дублирующиеся спаны в рамках одного намерения, оставив наиболее высокооцнённый.
    best_by_span: dict[tuple[str, int, int], dict[str, Any]] = {}
    for cand in expanded:
        key = (str(cand["intent_type"]), int(cand["char_start"]), int(cand["char_end"]))
        prev = best_by_span.get(key)
        cand_key = (float(cand.get("score", 0.0)), len(str(cand.get("expression", ""))))
        prev_key = (
            (float(prev.get("score", 0.0)), len(str(prev.get("expression", ""))))
            if prev is not None
            else None
        )
        if prev is None or (prev_key is not None and cand_key > prev_key):
            best_by_span[key] = cand

    filtered = [
        cand
        for cand in best_by_span.values()
        if float(cand.get("score", 0.0)) >= acceptance_threshold(cand)
    ]
    predictions = resolve_overlaps(filtered)
    return predictions



def predict_for_records(
    records: list[dict[str, Any]],
    compiled_lexicon: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Запускает baseline на наборе реплик.
    """
    all_predictions: list[dict[str, Any]] = []

    for record in records:
        preds = predict_for_record(record, compiled_lexicon)
        all_predictions.extend(preds)

    return all_predictions



def build_gold_span_index(records: list[dict[str, Any]]) -> dict[str, set[tuple[int, int, str]]]:
    """
    Строит индекс gold-span'ов по utterance_id:
        utterance_id -> {(char_start, char_end, intent_type), ...}
    """
    gold_index: dict[str, set[tuple[int, int, str]]] = defaultdict(set)

    for record in records:
        utterance_id = record["utterance_id"]
        annotations = record.get("annotations", [])

        for ann in annotations:
            gold_index[utterance_id].add(
                (
                    int(ann["char_start"]),
                    int(ann["char_end"]),
                    ann["intent_type"],
                )
            )

    return gold_index



def build_pred_span_index(predictions: list[dict[str, Any]]) -> dict[str, set[tuple[int, int, str]]]:
    """
    Строит индекс предсказанных span'ов по utterance_id.
    """
    pred_index: dict[str, set[tuple[int, int, str]]] = defaultdict(set)

    for pred in predictions:
        utterance_id = pred["utterance_id"]
        pred_index[utterance_id].add(
            (
                int(pred["char_start"]),
                int(pred["char_end"]),
                pred["intent_type"],
            )
        )

    return pred_index



def compute_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """
    Считает precision / recall / f1.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }



def evaluate_predictions(
    gold_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Оценивает baseline по exact-match на span + label.
    """
    gold_index = build_gold_span_index(gold_records)
    pred_index = build_pred_span_index(predictions)

    utterance_ids = sorted(set(gold_index.keys()) | set(pred_index.keys()))

    overall_tp = overall_fp = overall_fn = 0
    per_label_counts = {
        "contact_open": {"tp": 0, "fp": 0, "fn": 0},
        "contact_close": {"tp": 0, "fp": 0, "fn": 0},
    }

    for utterance_id in utterance_ids:
        gold_spans = gold_index.get(utterance_id, set())
        pred_spans = pred_index.get(utterance_id, set())

        tp_set = gold_spans & pred_spans
        fp_set = pred_spans - gold_spans
        fn_set = gold_spans - pred_spans

        overall_tp += len(tp_set)
        overall_fp += len(fp_set)
        overall_fn += len(fn_set)

        for _, _, label in tp_set:
            per_label_counts[label]["tp"] += 1
        for _, _, label in fp_set:
            per_label_counts[label]["fp"] += 1
        for _, _, label in fn_set:
            per_label_counts[label]["fn"] += 1

    metrics = {
        "overall": {
            "tp": overall_tp,
            "fp": overall_fp,
            "fn": overall_fn,
            **compute_prf(overall_tp, overall_fp, overall_fn),
        },
        "per_label": {},
    }

    for label, counts in per_label_counts.items():
        metrics["per_label"][label] = {
            **counts,
            **compute_prf(counts["tp"], counts["fp"], counts["fn"]),
        }

    return metrics



def print_lexicon_stats(lexicon: dict[str, Any]) -> None:
    """
    Печатает краткую статистику по словарю baseline.
    """
    print("\n=== RULE LEXICON STATS ===")
    for intent_type, items in lexicon.items():
        print(f"{intent_type}: {len(items)} выражений")

        top_items = items[:10]
        if top_items:
            print("  Топ-10:")
            for item in top_items:
                print(f"    {item['expression_text']} ({item['frequency']})")



def print_metrics(metrics: dict[str, Any]) -> None:
    """
    Печатает метрики baseline.
    """
    print("\n=== RULE-BASED BASELINE METRICS ===")
    overall = metrics["overall"]
    print(
        f"overall -> TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
        f"P={overall['precision']} R={overall['recall']} F1={overall['f1']}"
    )

    print("\nПо классам:")
    for label, values in metrics["per_label"].items():
        print(
            f"{label} -> TP={values['tp']} FP={values['fp']} FN={values['fn']} "
            f"P={values['precision']} R={values['recall']} F1={values['f1']}"
        )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rule-based baseline для извлечения контактоустанавливающих и контактозавершающих выражений."
    )
    parser.add_argument(
        "--fit-input",
        type=str,
        default="data/processed/gold_dialogues.jsonl",
        help="JSONL-файл, из которого строится словарь baseline.",
    )
    parser.add_argument(
        "--predict-input",
        type=str,
        default="data/processed/gold_dialogues.jsonl",
        help="JSONL-файл, на котором выполняется rule-based inference.",
    )
    parser.add_argument(
        "--lexicon-output",
        type=str,
        default="data/processed/rule_lexicon.json",
        help="Куда сохранить словарь baseline.",
    )
    parser.add_argument(
        "--predictions-output",
        type=str,
        default="artifacts/rule_based_predictions.jsonl",
        help="Куда сохранить предсказания baseline.",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="artifacts/rule_based_metrics.json",
        help="Куда сохранить метрики baseline.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Минимальная частота выражения в gold для включения в словарь.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    fit_input = Path(args.fit_input)
    predict_input = Path(args.predict_input)
    lexicon_output = Path(args.lexicon_output)
    predictions_output = Path(args.predictions_output)
    metrics_output = Path(args.metrics_output)

    if not fit_input.exists():
        raise FileNotFoundError(f"Не найден fit-input: {fit_input}")
    if not predict_input.exists():
        raise FileNotFoundError(f"Не найден predict-input: {predict_input}")

    fit_records = load_jsonl(fit_input)
    predict_records = load_jsonl(predict_input)

    lexicon = extract_lexicon_from_gold(fit_records, min_freq=args.min_freq)
    save_json(lexicon, lexicon_output)
    print_lexicon_stats(lexicon)

    compiled_lexicon = compile_lexicon(lexicon)
    predictions = predict_for_records(predict_records, compiled_lexicon)
    save_jsonl(predictions, predictions_output)

    metrics = evaluate_predictions(predict_records, predictions)
    save_json(metrics, metrics_output)
    print_metrics(metrics)

    print("\n=== OUTPUT PATHS ===")
    print(f"Словарь сохранён в: {lexicon_output}")
    print(f"Предсказания сохранены в: {predictions_output}")
    print(f"Метрики сохранены в: {metrics_output}")


if __name__ == "__main__":
    main()
