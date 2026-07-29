"""End-to-end composition of the symptom-scoring stages."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from symptom_scoring.config import ScoringConfig
from symptom_scoring.evaluation import compare_prior_expression_prediction
from symptom_scoring.madrs_bert_model import MadrsBertRegressor
from symptom_scoring.relevance_detector import HybridRelevanceDetector
from symptom_scoring.session_aggregator import aggregate_topics, project_to_phq
from symptom_scoring.transcript_parser import (
    build_turn_pairs,
    load_prior_levels,
    load_session_source,
)
from symptom_scoring.translator import (
    IdentityTranslator,
    MarianEnglishGermanTranslator,
)
from symptom_scoring.turn_scorer import TurnScorer


class SymptomScoringPipeline:
    """Own model instances once and reuse them across one or many sessions."""

    def __init__(
        self,
        config: ScoringConfig,
        *,
        relevance_detector=None,
        translator=None,
        regressor=None,
    ):
        self.config = config
        self.relevance_detector = relevance_detector or HybridRelevanceDetector(
            mode=config.relevance_mode,
            combined_threshold=config.combined_similarity_threshold,
            patient_threshold=config.patient_similarity_threshold,
            acceptance_threshold=config.relevance_acceptance_threshold,
        )
        if translator is not None:
            self.translator = translator
        elif config.source_language == "de":
            self.translator = IdentityTranslator()
        else:
            self.translator = MarianEnglishGermanTranslator(
                config.translator_model_id,
                device=config.device,
                batch_size=config.batch_size,
            )
        self.regressor = regressor or MadrsBertRegressor(
            config.model_id,
            device=config.device,
            batch_size=config.batch_size,
            max_length=config.max_length,
        )
        self.turn_scorer = TurnScorer(
            self.relevance_detector,
            self.translator,
            self.regressor,
        )

    def score_session(self, path: str | Path) -> dict:
        source = load_session_source(path)
        pairs = build_turn_pairs(source)
        turn_results = self.turn_scorer.score_session(pairs)
        aggregates = aggregate_topics(turn_results)
        phq_results = project_to_phq(aggregates)
        prior_levels, prior_source = load_prior_levels(source.metadata)
        comparison = compare_prior_expression_prediction(
            prior_levels,
            prior_source,
            pairs,
            turn_results,
            phq_results,
        )

        warnings: list[str] = [
            "Outputs are research-only MADRS proxy scores, not clinical diagnoses.",
            "PHQ prior anchors and MADRS severity scores are not numerically comparable.",
        ]
        if any(pair.context_synthesized for pair in pairs):
            warnings.append(
                "The initial prompt for at least one turn was reconstructed for a legacy session."
            )
        if prior_source in {"case_name_inferred_incomplete", "unavailable"}:
            warnings.append(f"Patient prior levels are incomplete ({prior_source}).")

        return {
            "schema_version": "1.0",
            "patient_id": source.metadata.get("patient_id"),
            "session_id": source.metadata.get("session_id", source.session_dir.name),
            "source_path": str(source.session_dir),
            "source_language": source.metadata.get(
                "conversation_language", self.config.source_language
            ),
            "created_at": datetime.now(UTC).isoformat(),
            "model": {
                "madrs_model_id": self.config.model_id,
                "translation_model_id": self.translator.model_id,
                "device": str(self.regressor.device),
                "batch_size": self.regressor.batch_size,
                "max_length": self.config.max_length,
            },
            "relevance": {
                "method": self.config.relevance_mode,
                "acceptance_threshold": self.config.relevance_acceptance_threshold,
                "semantic_thresholds": {
                    "combined": self.config.combined_similarity_threshold,
                    "patient": self.config.patient_similarity_threshold,
                },
                "confidence_is_calibrated_probability": False,
            },
            "aggregation_method": self.config.aggregation_method,
            "turn_results": [item.to_dict() for item in turn_results],
            "madrs_topics": {
                topic: aggregate.to_dict()
                for topic, aggregate in aggregates.items()
            },
            "symptoms": phq_results,
            "prior_comparison": comparison,
            "warnings": warnings,
        }

