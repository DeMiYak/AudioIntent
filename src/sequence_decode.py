"""
Optional Viterbi decoder for contact-opening/contact-closing predictions.

Disabled by default in pipeline.py via --use-sequence-decoder.  The decoder is a
post-filter: it does not create new predictions from utterances with no local
support; it only keeps predictions whose label is consistent with the best
sequence over a window.
"""
from __future__ import annotations

import math
from typing import Any

STATES = ["none", "contact_open", "contact_close"]
_STATE_IDX = {state: idx for idx, state in enumerate(STATES)}

# Tuned starting values for sparse film contact events.
ML_WEIGHT = 1.0
RARITY_PENALTY = -1.8
TIER_A_BONUS = 4.0
TIER_B_BONUS = 1.5
TIER_C_BONUS = 0.5
LEXICON_STRONG_BONUS = 2.0
LEXICON_WEAK_BONUS = 0.7
BOUNDARY_BONUS_STRONG = 0.9
BOUNDARY_BONUS_WEAK = 0.4
ANTI_BOUNDARY_PENALTY = -0.8
LONG_GAP_SEC = 6.0
REPEAT_WINDOW = 3
REPEAT_PENALTY = -0.5
LOGIT_CLIP = 3.0

# Transition log-penalties, not probabilities.  Higher is better.
TRANSITION = {
    "none": {"none": 0.0, "contact_open": -0.1, "contact_close": -0.2},
    "contact_open": {"none": 0.0, "contact_open": -0.7, "contact_close": -1.0},
    "contact_close": {"none": 0.0, "contact_open": -2.5, "contact_close": -0.5},
}
TRANSITION_AFTER_GAP = TRANSITION["none"]

CLOSE_WORDS = ("до свидания", "пока", "прощай", "увидимся", "созвонимся", "до встречи")
OPEN_WORDS = ("здравствуйте", "здравствуй", "здрасте", "познаком", "меня зовут")


def _lexical_contact_bonus(state: str, text: str) -> float:
    t = text.lower().replace("ё", "е")

    open_markers = (
        "привет", "здравств", "здрасте", "доброе утро", "добрый день",
        "добрый вечер", "алло", "познаком", "меня зовут", "рад знаком"
    )
    close_markers = (
        "пока", "до свидания", "досвидания", "до встречи", "увидимся",
        "созвонимся", "прощай", "чао", "всего доброго", "всего хорошего",
        "счастливо"
    )

    if state == "contact_open" and any(m in t for m in open_markers):
        return 1.0
    if state == "contact_close" and any(m in t for m in close_markers):
        return 1.0
    return 0.0

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _logit(prob: float) -> float:
    p = min(0.98, max(0.02, float(prob)))
    return max(-LOGIT_CLIP, min(LOGIT_CLIP, math.log(p / (1.0 - p))))


def _rule_bonus(pred: dict[str, Any]) -> float:
    rule = str(pred.get("rule_expression", ""))
    if rule == "ml_classifier":
        return 0.0
    try:
        from .rule_based_intent import MANUAL_RULES_TIER_A, MANUAL_RULES_TIER_B, MANUAL_RULES_TIER_C
    except Exception:
        MANUAL_RULES_TIER_A = frozenset()
        MANUAL_RULES_TIER_B = frozenset()
        MANUAL_RULES_TIER_C = frozenset()

    if rule in MANUAL_RULES_TIER_A:
        return TIER_A_BONUS
    if rule in MANUAL_RULES_TIER_B:
        return TIER_B_BONUS
    if rule in MANUAL_RULES_TIER_C:
        return TIER_C_BONUS

    freq = _safe_float(pred.get("rule_frequency"), 0.0)
    if freq >= 3:
        return LEXICON_STRONG_BONUS
    if freq > 0:
        return LEXICON_WEAK_BONUS
    return 0.0


def _boundary_bonus(state: str, idx: int, n: int, gap_before: float, gap_after: float) -> float:
    if state == "contact_open":
        bonus = 0.0
        if idx == 0 or gap_before >= LONG_GAP_SEC:
            bonus += BOUNDARY_BONUS_STRONG
        elif n > 1 and idx / max(1, n - 1) < 0.15:
            bonus += BOUNDARY_BONUS_WEAK
        elif 0.25 <= idx / max(1, n - 1) <= 0.75 and gap_before < LONG_GAP_SEC:
            bonus += ANTI_BOUNDARY_PENALTY
        return bonus
    if state == "contact_close":
        bonus = 0.0
        if idx == n - 1 or gap_after >= LONG_GAP_SEC:
            bonus += BOUNDARY_BONUS_STRONG
        elif n > 1 and idx / max(1, n - 1) > 0.85:
            bonus += BOUNDARY_BONUS_WEAK
        elif 0.25 <= idx / max(1, n - 1) <= 0.75 and gap_after < LONG_GAP_SEC:
            bonus += ANTI_BOUNDARY_PENALTY
        return bonus
    return 0.0


def _prediction_support(predictions: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Keep the strongest local prediction per (utterance_id, intent_type)."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for pred in predictions:
        uid = str(pred.get("utterance_id", ""))
        intent = str(pred.get("intent_type", ""))
        if intent not in {"contact_open", "contact_close"}:
            continue
        key = (uid, intent)
        score = _safe_float(pred.get("confidence"), 1.0) + _rule_bonus(pred)
        prev = best.get(key)
        if prev is None or score > (_safe_float(prev.get("confidence"), 1.0) + _rule_bonus(prev)):
            best[key] = pred
    return best


def _emission_score(
    uid: str,
    state: str,
    idx: int,
    n: int,
    gap_before: float,
    gap_after: float,
    support: dict[tuple[str, str], dict[str, Any]],
) -> float:
    if state == "none":
        return 0.0

    pred = support.get((uid, state))
    # This decoder is a filter.  Do not invent contact events with no local evidence.
    if pred is None:
        return -8.0

    conf = _safe_float(pred.get("confidence"), 1.0)
    score = ML_WEIGHT * _logit(conf)
    score += RARITY_PENALTY
    score += _rule_bonus(pred) 
    rule_expr = str(pred.get("rule_expression", ""))
    text = str(pred.get("source_text") or pred.get("expression") or "")
    num_words = len(text.split())

    # ML-only predictions are much less reliable than rule/lexicon-supported events.
    if rule_expr == "ml_classifier":
        score -= 1.2

        # Contact openings/closings are usually short. Long ML-only utterances are often false positives.
        if num_words > 8:
            score -= 1.0

        # Very long ML-only utterances should almost never survive as contact acts.
        if num_words > 14:
            score -= 1.5
    score += _boundary_bonus(state, idx, n, gap_before, gap_after)
    score += _lexical_contact_bonus(state, text)
    return score


def _filter_local_repeats(
    utterance_ids: list[str],
    decoded_states: list[str],
    support: dict[tuple[str, str], dict[str, Any]],
) -> set[tuple[str, str]]:
    """Drop low-support repeated same-label events inside a short local window."""
    kept: set[tuple[str, str]] = set()
    recent_by_state: dict[str, list[int]] = {"contact_open": [], "contact_close": []}

    for idx, (uid, state) in enumerate(zip(utterance_ids, decoded_states)):
        if state == "none":
            continue
        pred = support.get((uid, state))
        if pred is None:
            continue

        recent = [j for j in recent_by_state[state] if idx - j <= REPEAT_WINDOW]
        recent_by_state[state] = recent
        rule = str(pred.get("rule_expression", ""))
        try:
            from .rule_based_intent import MANUAL_RULES_TIER_A
        except Exception:
            MANUAL_RULES_TIER_A = frozenset()

        # Keep Tier A repetitions (e.g. multi-speaker farewell exchange), but suppress weak repeats.
        if recent and rule not in MANUAL_RULES_TIER_A and _rule_bonus(pred) < TIER_B_BONUS:
            continue

        kept.add((uid, state))
        recent_by_state[state].append(idx)

    return kept


def viterbi_decode(
    utterances: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return predictions supported by the best sequence over the window."""
    if not utterances or not predictions:
        return predictions

    sorted_utts = sorted(utterances, key=lambda u: (_safe_float(u.get("start_time")), _safe_float(u.get("end_time"))))
    utterance_ids = [str(u.get("utterance_id", "")) for u in sorted_utts]
    starts = [_safe_float(u.get("start_time")) for u in sorted_utts]
    ends = [_safe_float(u.get("end_time"), starts[i]) for i, u in enumerate(sorted_utts)]
    n = len(utterance_ids)
    support = _prediction_support(predictions)

    dp: list[dict[str, float]] = []
    back: list[dict[str, str | None]] = []

    for idx, uid in enumerate(utterance_ids):
        gap_before = starts[idx] - ends[idx - 1] if idx > 0 else LONG_GAP_SEC
        gap_after = starts[idx + 1] - ends[idx] if idx < n - 1 else LONG_GAP_SEC
        dp_row: dict[str, float] = {}
        back_row: dict[str, str | None] = {}

        for state in STATES:
            emit = _emission_score(uid, state, idx, n, gap_before, gap_after, support)
            if idx == 0:
                dp_row[state] = emit
                back_row[state] = None
                continue

            trans_table = TRANSITION_AFTER_GAP if gap_before >= LONG_GAP_SEC else None
            best_prev = None
            best_score = -1e18
            for prev_state in STATES:
                trans = (trans_table[state] if trans_table is not None else TRANSITION[prev_state][state])
                score = dp[idx - 1][prev_state] + trans + emit
                if state != "none" and prev_state == state:
                    score += REPEAT_PENALTY
                if score > best_score:
                    best_score = score
                    best_prev = prev_state
            dp_row[state] = best_score
            back_row[state] = best_prev

        dp.append(dp_row)
        back.append(back_row)

    final_state = max(STATES, key=lambda s: dp[-1][s])
    decoded = ["none"] * n
    decoded[-1] = final_state
    for idx in range(n - 1, 0, -1):
        prev = back[idx][decoded[idx]]
        decoded[idx - 1] = prev or "none"

    keep = _filter_local_repeats(utterance_ids, decoded, support)
    filtered: list[dict[str, Any]] = []
    for pred in predictions:
        key = (str(pred.get("utterance_id", "")), str(pred.get("intent_type", "")))
        if key in keep:
            p = dict(pred)
            p["sequence_decoder"] = "kept"
            filtered.append(p)
    return filtered
