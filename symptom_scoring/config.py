"""Runtime settings and the PHQ-style prior vocabulary.

This module covers the *prior* side only: the PHQ-9 anchors patient priors are
written in, and the process-wide scoring settings. The rating side — topics,
relevance patterns, score range — lives in `symptom_scoring.instrument`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from symptom_scoring.instrument import Instrument
from symptom_scoring.instruments import DEFAULT_INSTRUMENT


PHQ_SYMPTOM_LABELS = {
    "lack_of_pleasure": "Little interest or pleasure in doing things",
    "depressed_mood": "Feeling down, depressed, or hopeless",
    "sleep_problems": "Sleep problems",
    "low_energy": "Feeling tired or having little energy",
    "appetite_changes": "Poor appetite or overeating",
    "feelings_of_failure_or_guilt": "Feeling bad about yourself, guilty, or like a failure",
    "concentration_problems": "Trouble concentrating",
    "psychomotor_changes": "Moving or speaking slowly, or feeling restless",
    "thoughts_of_death_or_self_harm": (
        "Thoughts that you would be better off dead or of hurting yourself"
    ),
}

FREQUENCY_VALUES = {
    "not at all": 0,
    "several days": 1,
    "more than half the days": 2,
    "nearly every day": 3,
}

KNOWN_OPENING_PROMPT = (
    "You have just arrived for a therapy session. "
    "The therapist is present and waiting. Say what brings you here."
)


@dataclass(frozen=True)
class ScoringConfig:
    """Runtime settings shared by every session in one evaluation process."""

    instrument: Instrument = field(default=DEFAULT_INSTRUMENT)
    model_id: str = "webesama/MADRS-BERT"
    translator_model_id: str = "Helsinki-NLP/opus-mt-en-de"
    source_language: str = "en"
    device: str = "auto"
    batch_size: int | None = None
    max_length: int = 512
    relevance_mode: str = "hybrid"
    combined_similarity_threshold: float = 0.50
    patient_similarity_threshold: float = 0.35
    relevance_acceptance_threshold: float = 0.80
    aggregation_method: str = "maximum_relevant_evidence"
