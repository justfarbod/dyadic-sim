"""Maximum-relevant-evidence aggregation and PHQ proxy projection."""

from __future__ import annotations

from symptom_scoring.config import (
    MADRS_TOPICS,
    PHQ_MAPPING_QUALITY,
    PHQ_TO_MADRS,
)
from symptom_scoring.types import TopicAggregate, TurnTopicResult


def aggregate_topics(results: list[TurnTopicResult]) -> dict[str, TopicAggregate]:
    aggregates: dict[str, TopicAggregate] = {}
    for topic in MADRS_TOPICS:
        eligible = [
            result
            for result in results
            if result.madrs_topic == topic
            and result.accepted_for_scoring
            and result.raw_score is not None
        ]
        eligible.sort(
            key=lambda item: (
                -float(item.raw_score),
                -item.relevance_confidence,
                item.turn_index,
            )
        )
        relevant_turns = sorted({item.turn_index for item in eligible})
        if not eligible:
            aggregates[topic] = TopicAggregate(
                madrs_topic=topic,
                raw_session_score=None,
                rounded_session_score=None,
                source_turn=None,
                relevant_turns=[],
                evidence=None,
                number_of_relevant_turns=0,
            )
            continue

        maximum = eligible[0]
        aggregates[topic] = TopicAggregate(
            madrs_topic=topic,
            raw_session_score=maximum.raw_score,
            rounded_session_score=maximum.rounded_score,
            source_turn=maximum.turn_index,
            relevant_turns=relevant_turns,
            evidence=maximum.evidence,
            number_of_relevant_turns=len(relevant_turns),
            therapist_text=maximum.therapist_text,
            patient_text=maximum.patient_text,
        )
    return aggregates


def project_to_phq(
    aggregates: dict[str, TopicAggregate],
) -> dict[str, dict]:
    projected: dict[str, dict] = {}
    for symptom, topic in PHQ_TO_MADRS.items():
        if topic is None:
            projected[symptom] = {
                "output_type": "MADRS proxy score",
                "madrs_topic": None,
                "mapping_quality": PHQ_MAPPING_QUALITY[symptom],
                "raw_session_score": None,
                "rounded_session_score": None,
                "source_turn": None,
                "relevant_turns": [],
                "evidence": None,
                "number_of_relevant_turns": 0,
                "reason": "No directly corresponding selected MADRS-BERT topic",
            }
            continue

        aggregate = aggregates[topic]
        projected[symptom] = {
            "output_type": "MADRS proxy score",
            "madrs_topic": topic,
            "mapping_quality": PHQ_MAPPING_QUALITY[symptom],
            "raw_session_score": aggregate.raw_session_score,
            "rounded_session_score": aggregate.rounded_session_score,
            "source_turn": aggregate.source_turn,
            "relevant_turns": aggregate.relevant_turns,
            "evidence": aggregate.evidence,
            "number_of_relevant_turns": aggregate.number_of_relevant_turns,
        }
    return projected

