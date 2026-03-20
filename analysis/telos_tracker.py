"""
analysis/telos_tracker.py

Marker 6: Telos Tracking (internal purposiveness and self-dissolution)

The therapeutic dyad has its own ending as its immanent goal. A dyad
that has genuinely instantiated the therapeutic structure will orient
toward its own dissolution: the patient grows more autonomous, the
therapist steps back.

A dyad drifting toward mutual perpetuation (interesting conversation
for its own sake) has lost its telos.

Four sub-measures:
  1. Initiative shift: does the patient generate proportionally more over time?
  2. Therapist length trend: are therapist turns getting shorter?
  3. Autonomy markers: patient self-generating meaning
  4. Dependency markers: patient pulling toward therapist
"""

import re
import numpy as np


# Patient autonomy markers: self-generated insight
AUTONOMY_PATTERNS = [
    r"\bi (realise|realize|wonder|think maybe|see now)\b",
    r"\bi('m| am) (beginning|starting) to\b",
    r"\bmaybe (i|it('s| is))\b",
    r"\bi never (thought|considered|noticed)\b",
    r"\bsomething (is |has )?(shifting|changing|different)\b",
    r"\bi can (see|feel|sense)\b.*\bnow\b",
    r"\bfor the first time\b",
    r"\bi didn't (realise|realize|see|know)\b",
    r"\bwhat if i\b",
    r"\bi could try\b",
]

# Patient dependency markers: pulling toward therapist
DEPENDENCY_PATTERNS = [
    r"\btell me\b",
    r"\bwhat should i\b",
    r"\bi need you\b",
    r"\bdon't leave\b",
    r"\bwill you be here\b",
    r"\bcan i call you\b",
    r"\bwhat do you think i should\b",
    r"\bplease (help|tell|fix)\b",
    r"\bi can't (do this|manage) (alone|without|by myself)\b",
    r"\bdo you (think|believe) i\b.*\b(can|will)\b",
]


def analyse_telos(
    therapist_turns: list[str],
    patient_turns: list[str],
) -> dict:
    """
    Track orientation toward dissolution across a session.

    Args:
        therapist_turns:  Therapist outputs, one per turn
        patient_turns:    Patient outputs, one per turn

    Returns:
        {
          "initiative_trend":        "toward_patient" | "toward_therapist" | "stable",
          "therapist_length_trend":  "decreasing" | "increasing" | "stable",
          "autonomy_turns":          list of turn numbers,
          "dependency_turns":        list of turn numbers,
          "telos_score":             float 0-1,
          "interpretation":          str
        }
    """
    n = min(len(therapist_turns), len(patient_turns))
    if n == 0:
        return {
            "initiative_trend": "stable",
            "therapist_length_trend": "stable",
            "autonomy_turns": [],
            "dependency_turns": [],
            "telos_score": 0.0,
            "interpretation": "No turns.",
        }

    # Sub-measure 1: initiative shift (patient word count as proportion)
    patient_lengths = [len(t.split()) for t in patient_turns[:n]]
    therapist_lengths = [len(t.split()) for t in therapist_turns[:n]]

    ratios = []
    for pl, tl in zip(patient_lengths, therapist_lengths):
        total = pl + tl
        ratios.append(pl / total if total > 0 else 0.5)

    initiative_trend = _trend_direction(ratios, "toward_patient", "toward_therapist")

    # Sub-measure 2: therapist length trend
    therapist_length_trend = _trend_direction(
        therapist_lengths, "increasing", "decreasing",
    )

    # Sub-measure 3: autonomy markers
    autonomy_turns = []
    for i, text in enumerate(patient_turns[:n]):
        if any(re.search(p, text.lower()) for p in AUTONOMY_PATTERNS):
            autonomy_turns.append(i + 1)

    # Sub-measure 4: dependency markers
    dependency_turns = []
    for i, text in enumerate(patient_turns[:n]):
        if any(re.search(p, text.lower()) for p in DEPENDENCY_PATTERNS):
            dependency_turns.append(i + 1)

    # Composite telos score
    telos_score = _compute_telos_score(
        initiative_trend, therapist_length_trend,
        autonomy_turns, dependency_turns, n,
    )

    interpretation = _interpret(
        telos_score, initiative_trend, therapist_length_trend,
        autonomy_turns, dependency_turns, n,
    )

    return {
        "initiative_trend": initiative_trend,
        "therapist_length_trend": therapist_length_trend,
        "autonomy_turns": autonomy_turns,
        "dependency_turns": dependency_turns,
        "telos_score": round(telos_score, 4),
        "interpretation": interpretation,
    }


def _trend_direction(
    values: list[float],
    increasing_label: str,
    decreasing_label: str,
) -> str:
    if len(values) < 3:
        return "stable"

    x = np.arange(len(values))
    slope = float(np.polyfit(x, values, 1)[0])

    if slope > 0.005:
        return increasing_label
    if slope < -0.005:
        return decreasing_label
    return "stable"


def _compute_telos_score(
    initiative: str, therapist_length: str,
    autonomy: list, dependency: list, n: int,
) -> float:
    score = 0.0

    # Initiative shifting toward patient = good
    if initiative == "toward_patient":
        score += 0.30
    elif initiative == "stable":
        score += 0.10

    # Therapist getting shorter = good
    if therapist_length == "decreasing":
        score += 0.25
    elif therapist_length == "stable":
        score += 0.10

    # Autonomy markers
    autonomy_rate = len(autonomy) / n if n else 0
    score += 0.25 * min(autonomy_rate / 0.25, 1.0)

    # Dependency markers reduce score
    dependency_rate = len(dependency) / n if n else 0
    score -= 0.20 * min(dependency_rate / 0.25, 1.0)

    return max(0.0, min(1.0, score))


def _interpret(
    score: float, initiative: str, therapist_length: str,
    autonomy: list, dependency: list, n: int,
) -> str:
    if score < 0.2:
        return (
            f"Low telos score ({score:.2f}) across {n} turns. "
            "The dyad does not appear oriented toward its own dissolution. "
            f"Initiative trend: {initiative}. Therapist length: {therapist_length}. "
            f"Autonomy markers: {len(autonomy)}, dependency markers: {len(dependency)}. "
            "May indicate mutual perpetuation or insufficient session length."
        )
    if score < 0.5:
        return (
            f"Moderate telos score ({score:.2f}) across {n} turns. "
            "Some orientation toward dissolution. "
            f"Initiative trend: {initiative}. Therapist length: {therapist_length}. "
            f"Autonomy markers: {len(autonomy)}, dependency markers: {len(dependency)}. "
            "The patient shows some self-generated insight, but dependence persists."
        )
    return (
        f"Strong telos score ({score:.2f}) across {n} turns. "
        "The dyad is oriented toward its own dissolution: the patient is becoming "
        "more autonomous, the therapist is stepping back. "
        f"Initiative trend: {initiative}. Therapist length: {therapist_length}. "
        f"Autonomy markers: {len(autonomy)}, dependency markers: {len(dependency)}. "
        "This trajectory is consistent with genuine internal purposiveness."
    )