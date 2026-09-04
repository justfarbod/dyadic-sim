"""Runtime settings and the PHQ-style prior vocabulary.

This module covers the *prior* side only: the PHQ-9 anchors patient priors are
written in, and the process-wide scoring settings. The rating side (topics,
relevance patterns, score range) lives in `symptom_scoring.instrument`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from symptom_scoring.instrument import Instrument
from symptom_scoring.instruments import DEFAULT_INSTRUMENT
from symptom_scoring.prior_vocabulary import LABELS

# The PHQ-9 vocabulary lives in symptom_scoring.prior_vocabulary, named for its
# role (what priors are written in) rather than for the instrument. It holds all three
# registers (KEYS, LABELS, REFERENCES) side by side so they stop being
# mistaken for copies of one another. Re-exported here under its historical
# name because the prior side has always imported it from config.
PHQ_SYMPTOM_LABELS = LABELS

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
