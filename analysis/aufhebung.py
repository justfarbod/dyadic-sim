"""
analysis/aufhebung.py

Marker 4: Aufhebung Structure (Hegelian dialectical development)

Detects whether the conversation has cumulative depth (later turns
preserving and transforming earlier ones) or whether each turn is
essentially stateless.

Three sub-measures:
  1. Back-reference rate: explicit references to earlier turns
  2. Transformation rate: position shifts that integrate rather than replace
  3. Phase boundaries: turns where session character shifts substantially

Markers 1 and 2 are pattern-based. Phase boundaries use embedding-based
semantic jump detection when available, falling back to a simpler heuristic.
"""

import re
import numpy as np

from analysis.embeddings import get_model as _get_model, EMBEDDINGS_AVAILABLE


# Back-reference patterns: language pointing to earlier moments
BACK_REFERENCE_PATTERNS = [
    r"\bearlier\b",
    r"\byou mentioned\b",
    r"\byou said\b",
    r"\bcoming back to\b",
    r"\bback to what\b",
    r"\bremember when\b",
    r"\bbefore,?\s+you\b",
    r"\bat the start\b",
    r"\bwhen we first\b",
    r"\blast time\b",
    r"\bthat moment\b",
    r"\bwhat you said about\b",
    r"\bi keep thinking about\b",
]

# Transformation patterns: language marking position shifts
TRANSFORMATION_PATTERNS = [
    r"\band yet\b",
    r"\bsomething shifted\b",
    r"\bnot just\b",
    r"\bmore than\b.*\bthought\b",
    r"\bi('m| am) (beginning|starting) to\b",
    r"\bnow i (see|think|realise|realize|understand|feel)\b",
    r"\bbut also\b",
    r"\bat the same time\b",
    r"\bi used to\b.*\bbut now\b",
    r"\bit's different\b",
    r"\bactually\b.*\bmaybe\b",
    r"\bi didn't see\b",
]


def analyse_aufhebung(
    turns: list[str],
    role: str,
) -> dict:
    """
    Detect aufhebung structure in an agent's turns.

    Args:
        turns:  List of agent outputs, one per turn
        role:   "therapist" or "patient"

    Returns:
        {
          "back_reference_turns":  list of turn numbers,
          "transformation_turns":  list of turn numbers,
          "phase_boundaries":      list of turn numbers where phase shifts detected,
          "cumulative_score":      float 0-1,
          "interpretation":        str
        }
    """
    if not turns:
        return {
            "back_reference_turns": [],
            "transformation_turns": [],
            "phase_boundaries": [],
            "cumulative_score": 0.0,
            "interpretation": "No turns.",
        }

    back_ref_turns = []
    transform_turns = []

    for i, text in enumerate(turns):
        text_lower = text.lower()
        turn_num = i + 1

        if any(re.search(p, text_lower) for p in BACK_REFERENCE_PATTERNS):
            back_ref_turns.append(turn_num)

        if any(re.search(p, text_lower) for p in TRANSFORMATION_PATTERNS):
            transform_turns.append(turn_num)

    phase_boundaries = _detect_phase_boundaries(turns)

    # Composite score: weighted combination of the three sub-measures
    n = len(turns)
    back_ref_rate = len(back_ref_turns) / n if n else 0
    transform_rate = len(transform_turns) / n if n else 0
    phase_rate = len(phase_boundaries) / max(n - 1, 1) if n > 1 else 0

    # Weight: back-references and transformations matter more than phases
    cumulative_score = min(1.0, (
        0.35 * min(back_ref_rate / 0.3, 1.0) +
        0.45 * min(transform_rate / 0.3, 1.0) +
        0.20 * min(phase_rate / 0.15, 1.0)
    ))

    interpretation = _interpret(
        cumulative_score, back_ref_turns, transform_turns,
        phase_boundaries, role, n,
    )

    return {
        "back_reference_turns": back_ref_turns,
        "transformation_turns": transform_turns,
        "phase_boundaries": phase_boundaries,
        "cumulative_score": round(cumulative_score, 4),
        "interpretation": interpretation,
    }


def _detect_phase_boundaries(turns: list[str]) -> list[int]:
    """
    Detect turns where the session character shifts substantially.
    Uses embedding-based semantic jump if available, otherwise
    falls back to length-ratio heuristic.
    """
    if len(turns) < 3:
        return []

    if EMBEDDINGS_AVAILABLE:
        return _phase_boundaries_embedding(turns)
    return _phase_boundaries_heuristic(turns)


def _phase_boundaries_embedding(turns: list[str]) -> list[int]:
    model = _get_model()
    embeddings = model.encode(turns, convert_to_numpy=True)

    # Compute consecutive cosine distances
    distances = []
    for i in range(1, len(embeddings)):
        sim = np.dot(embeddings[i - 1], embeddings[i]) / (
            np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i])
        )
        distances.append(float(1.0 - sim))

    if not distances:
        return []

    # A phase boundary is a turn where the semantic jump exceeds
    # the mean + 1 standard deviation
    mean_d = np.mean(distances)
    std_d = np.std(distances)
    threshold = mean_d + std_d

    return [
        i + 2  # turn number (1-indexed, boundary is at the arriving turn)
        for i, d in enumerate(distances)
        if d > threshold and std_d > 0.01
    ]


def _phase_boundaries_heuristic(turns: list[str]) -> list[int]:
    """Fallback: detect large changes in turn length as a proxy."""
    lengths = [len(t.split()) for t in turns]
    boundaries = []

    for i in range(1, len(lengths)):
        if lengths[i - 1] == 0:
            continue
        ratio = lengths[i] / lengths[i - 1]
        if ratio > 2.0 or ratio < 0.5:
            boundaries.append(i + 1)

    return boundaries


def _interpret(
    score: float, back_refs: list, transforms: list,
    phases: list, role: str, n: int,
) -> str:
    if score < 0.15:
        return (
            f"Low cumulative structure ({score:.2f}) in {n} {role} turns. "
            "The conversation appears largely stateless: each turn responds "
            "to the immediate input without preserving or transforming earlier ones. "
            "No evidence of dialectical development."
        )
    if score < 0.4:
        return (
            f"Moderate cumulative structure ({score:.2f}) in {n} {role} turns. "
            f"Back-references in {len(back_refs)} turns, transformations in "
            f"{len(transforms)} turns, {len(phases)} phase boundaries. "
            "Some evidence that later turns build on earlier ones, but not "
            "consistently sustained."
        )
    return (
        f"Strong cumulative structure ({score:.2f}) in {n} {role} turns. "
        f"Back-references in {len(back_refs)} turns, transformations in "
        f"{len(transforms)} turns, {len(phases)} phase boundaries. "
        "The conversation shows genuine Aufhebung: earlier positions are "
        "preserved and transformed, not simply replaced. This is evidence "
        "of dialectical development within the encounter."
    )