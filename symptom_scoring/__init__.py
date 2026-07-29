"""Session-level MADRS proxy scoring for dyadic-sim."""

from symptom_scoring.config import ScoringConfig

__all__ = ["ScoringConfig", "SymptomScoringPipeline"]


def __getattr__(name: str):
    if name == "SymptomScoringPipeline":
        from symptom_scoring.pipeline import SymptomScoringPipeline

        return SymptomScoringPipeline
    raise AttributeError(name)
