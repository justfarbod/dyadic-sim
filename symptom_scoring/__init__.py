"""Session-level symptom severity scoring for dyadic-sim.

The pipeline is instrument-agnostic; see `symptom_scoring.instrument`.
"""

from symptom_scoring.config import ScoringConfig
from symptom_scoring.instrument import Instrument
from symptom_scoring.instruments import MADRS, get_instrument

__all__ = [
    "MADRS",
    "Instrument",
    "ScoringConfig",
    "SymptomScoringPipeline",
    "get_instrument",
]


def __getattr__(name: str):
    if name == "SymptomScoringPipeline":
        from symptom_scoring.pipeline import SymptomScoringPipeline

        return SymptomScoringPipeline
    raise AttributeError(name)
