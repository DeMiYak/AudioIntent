"""
Опциональный Viterbi-декодер по последовательности реплик.

Включается флагом --use-sequence-decoder в pipeline.py.
По умолчанию отключён — стандартный вывод pipeline не изменяется.

Алгоритм:
  - Состояния: none / contact_open / contact_close
  - Эмиссии: на основе уверенности существующих предсказаний
  - Штрафные переходы: изолированные слабые события, немедленный close→open
"""
from __future__ import annotations

import math
from typing import Any

STATES = ["none", "contact_open", "contact_close"]
_STATE_IDX: dict[str, int] = {s: i for i, s in enumerate(STATES)}

# Матрица логарифмов переходных вероятностей [from_state][to_state]
# Штрафы: isolated открытие/закрытие, немедленный close→open
_LOG_TRANS: list[list[float]] = [
    # none → none/open/close
    [math.log(0.95), math.log(0.03), math.log(0.02)],
    # contact_open → none/open/close
    [math.log(0.70), math.log(0.25), math.log(0.05)],
    # contact_close → none/open/close  (close→open сильно штрафуется)
    [math.log(0.90), math.log(0.05), math.log(0.05)],
]

_LOG_INIT: list[float] = [math.log(1 / 3)] * 3
_SMOOTHING: float = 1e-6


def _emission_log_probs(
    utt_id: str,
    open_conf: dict[str, float],
    close_conf: dict[str, float],
) -> list[float]:
    """Логарифм вероятности наблюдения для каждого состояния."""
    p_open = open_conf.get(utt_id, 0.0)
    p_close = close_conf.get(utt_id, 0.0)
    p_none = max(_SMOOTHING, 1.0 - p_open - p_close)

    total = p_none + p_open + p_close
    return [
        math.log((p_none / total) + _SMOOTHING),
        math.log((p_open / total) + _SMOOTHING),
        math.log((p_close / total) + _SMOOTHING),
    ]


def viterbi_decode(
    utterances: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Применяет Viterbi-декодирование к последовательности реплик.

    Возвращает подмножество predictions, соответствующих наилучшей
    последовательности меток по Viterbi. Предсказания для реплик,
    которым Viterbi присвоил 'none', отбрасываются.
    """
    if not utterances or not predictions:
        return predictions

    # Максимальная уверенность per utterance_id
    open_conf: dict[str, float] = {}
    close_conf: dict[str, float] = {}
    for p in predictions:
        uid = str(p.get("utterance_id", ""))
        intent = str(p.get("intent_type", ""))
        conf = float(p.get("confidence", 1.0))
        if intent == "contact_open":
            open_conf[uid] = max(open_conf.get(uid, 0.0), conf)
        elif intent == "contact_close":
            close_conf[uid] = max(close_conf.get(uid, 0.0), conf)

    # Сортировка реплик по времени начала
    sorted_utts = sorted(utterances, key=lambda u: float(u.get("start_time", 0.0)))
    utt_ids = [str(u.get("utterance_id", "")) for u in sorted_utts]
    n = len(utt_ids)
    if n == 0:
        return predictions

    n_states = len(STATES)
    dp = [[float("-inf")] * n_states for _ in range(n)]
    backtrack = [[0] * n_states for _ in range(n)]

    emits = _emission_log_probs(utt_ids[0], open_conf, close_conf)
    for s in range(n_states):
        dp[0][s] = _LOG_INIT[s] + emits[s]

    for t in range(1, n):
        emits = _emission_log_probs(utt_ids[t], open_conf, close_conf)
        for s in range(n_states):
            best_prev, best_score = 0, float("-inf")
            for prev in range(n_states):
                score = dp[t - 1][prev] + _LOG_TRANS[prev][s]
                if score > best_score:
                    best_score, best_prev = score, prev
            dp[t][s] = best_score + emits[s]
            backtrack[t][s] = best_prev

    # Backtrack
    best_final = max(range(n_states), key=lambda s: dp[n - 1][s])
    decoded = [0] * n
    decoded[n - 1] = best_final
    for t in range(n - 2, -1, -1):
        decoded[t] = backtrack[t + 1][decoded[t + 1]]

    # Множество (utterance_id, intent_type), поддержанных Viterbi
    viterbi_keep: set[tuple[str, str]] = set()
    for t, s in enumerate(decoded):
        if s != _STATE_IDX["none"]:
            viterbi_keep.add((utt_ids[t], STATES[s]))

    return [
        p for p in predictions
        if (str(p.get("utterance_id", "")), str(p.get("intent_type", ""))) in viterbi_keep
    ]
