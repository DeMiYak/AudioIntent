# OUTDATED
from dataclasses import dataclass
from typing import Optional

@dataclass
class WordTimestamp:
    word: str
    start_time: float
    end_time: float

@dataclass
class SpeakerSegment:
    start_time: float
    end_time: float
    speaker_label: str

@dataclass
class Utterance:
    utterance_id: str
    start_time: float
    end_time: float
    speaker_label: str
    speaker_name: Optional[str] = None
    text: str
    words: list[WordTimestamp]

@dataclass
class IntentPrediction:
    utterance_id: str
    expression: str
    intent_type: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    confidence: Optional[float] = None

@dataclass
class FinalPrediction:
    prediction_id: str
    start_time: float
    end_time: float
    speaker_label: str
    speaker_name: str
    expression: str
    intent_type: str
    source_utterance: str
    confidence: Optional[float] = None

