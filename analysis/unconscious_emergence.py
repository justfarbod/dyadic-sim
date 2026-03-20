"""
analysis/unconscious_emergence.py

Special Analysis: Unconscious Agenda Emergence

Tracks whether and how the patient's hidden unconscious agenda surfaced,
and whether the therapist registered it.

The unconscious agenda is held in reserve by the simulation and not
passed to the patient agent until specific interaction patterns trigger
its reveal. It is not announced, it surfaces through the relational
process, which is precisely how it works clinically.

Measures:
  - When the agenda was revealed (turn number)
  - How it surfaced (gradually vs. abruptly)
  - Whether the therapist shifted after the reveal
  - Direction of the therapist's shift
"""

import numpy as np

from analysis.embeddings import get_model as _get_model, EMBEDDINGS_AVAILABLE


def analyse_emergence(
    transcript: list[dict],
    unconscious_content: str,
) -> dict:
    """
    Analyse how the unconscious agenda emerged in the session.

    Args:
        transcript:           List of turn dicts with at least
                              "patient_text", "therapist_text",
                              "unconscious_revealed" fields
        unconscious_content:  The text of the unconscious agenda

    Returns:
        {
          "reveal_turn":              int or None,
          "emergence_pattern":        "gradual" | "abrupt" | "not_revealed",
          "therapist_shift_score":    float,
          "therapist_shift_direction": "attuned" | "withdrawn" | "unchanged",
          "interpretation":           str
        }
    """
    if not transcript:
        return _empty("No transcript.")

    # Find the turn where unconscious was revealed
    reveal_turn = None
    for t in transcript:
        if t.get("unconscious_revealed"):
            reveal_turn = t.get("turn", None)
            break

    if reveal_turn is None:
        return _empty(
            "Unconscious agenda was not revealed during this session. "
            "The interaction patterns did not match the reveal trigger, "
            "or the session was too short."
        )

    # Analyse emergence pattern (gradual vs abrupt)
    patient_turns = [t["patient_text"] for t in transcript]
    emergence_pattern = _detect_emergence_pattern(
        patient_turns, unconscious_content, reveal_turn,
    )

    # Analyse therapist shift after reveal
    therapist_turns = [t["therapist_text"] for t in transcript]
    shift_score, shift_direction = _detect_therapist_shift(
        therapist_turns, reveal_turn,
    )

    interpretation = _interpret(
        reveal_turn, emergence_pattern, shift_score,
        shift_direction, len(transcript),
    )

    return {
        "reveal_turn": reveal_turn,
        "emergence_pattern": emergence_pattern,
        "therapist_shift_score": round(shift_score, 4),
        "therapist_shift_direction": shift_direction,
        "interpretation": interpretation,
    }


def _empty(reason: str) -> dict:
    return {
        "reveal_turn": None,
        "emergence_pattern": "not_revealed",
        "therapist_shift_score": 0.0,
        "therapist_shift_direction": "unchanged",
        "interpretation": reason,
    }


def _detect_emergence_pattern(
    patient_turns: list[str],
    unconscious_content: str,
    reveal_turn: int,
) -> str:
    """
    Determine if the patient's language was already converging toward the
    unconscious content before the reveal (gradual) or not (abrupt).
    """
    if not EMBEDDINGS_AVAILABLE or reveal_turn <= 1:
        return "abrupt"

    model = _get_model()

    # Compare pre-reveal patient turns to the unconscious content
    pre_reveal = patient_turns[:reveal_turn - 1]
    if len(pre_reveal) < 2:
        return "abrupt"

    unconscious_emb = model.encode([unconscious_content], convert_to_numpy=True)[0]
    pre_embeddings = model.encode(pre_reveal, convert_to_numpy=True)

    # Compute similarity trend: are pre-reveal turns getting closer
    # to the unconscious content over time?
    similarities = []
    for emb in pre_embeddings:
        sim = float(np.dot(emb, unconscious_emb) / (
            np.linalg.norm(emb) * np.linalg.norm(unconscious_emb)
        ))
        similarities.append(sim)

    if len(similarities) < 2:
        return "abrupt"

    # Fit trend
    x = np.arange(len(similarities))
    slope = float(np.polyfit(x, similarities, 1)[0])

    # If similarity to unconscious content was increasing before reveal
    return "gradual" if slope > 0.005 else "abrupt"


def _detect_therapist_shift(
    therapist_turns: list[str],
    reveal_turn: int,
) -> tuple[float, str]:
    """
    Detect whether the therapist's behaviour shifted after the reveal.
    Returns (shift_score, direction).
    """
    # We need at least 2 turns before and after
    if reveal_turn < 3 or reveal_turn >= len(therapist_turns) - 1:
        return 0.0, "unchanged"

    pre = therapist_turns[max(0, reveal_turn - 3):reveal_turn - 1]
    post = therapist_turns[reveal_turn - 1:reveal_turn + 2]

    if not pre or not post:
        return 0.0, "unchanged"

    # Length shift: therapist getting shorter = withdrawn, longer = attuned
    pre_len = np.mean([len(t.split()) for t in pre])
    post_len = np.mean([len(t.split()) for t in post])

    if pre_len == 0:
        return 0.0, "unchanged"

    length_ratio = post_len / pre_len

    # Embedding-based shift if available
    if EMBEDDINGS_AVAILABLE:
        model = _get_model()
        pre_emb = model.encode(pre, convert_to_numpy=True)
        post_emb = model.encode(post, convert_to_numpy=True)

        pre_mean = np.mean(pre_emb, axis=0)
        post_mean = np.mean(post_emb, axis=0)

        semantic_shift = float(1.0 - np.dot(pre_mean, post_mean) / (
            np.linalg.norm(pre_mean) * np.linalg.norm(post_mean)
        ))
    else:
        semantic_shift = abs(1.0 - length_ratio) * 0.5

    # Determine direction
    if length_ratio > 1.15 and semantic_shift > 0.05:
        direction = "attuned"
    elif length_ratio < 0.75 and semantic_shift > 0.05:
        direction = "withdrawn"
    else:
        direction = "unchanged"

    return semantic_shift, direction


def _interpret(
    reveal_turn: int, pattern: str, shift_score: float,
    shift_dir: str, total_turns: int,
) -> str:
    parts = [
        f"Unconscious agenda surfaced at turn {reveal_turn} of {total_turns}.",
    ]

    if pattern == "gradual":
        parts.append(
            "Emergence was gradual: the patient's language was already "
            "converging toward the unconscious content before the formal reveal. "
            "The relational process was doing real work."
        )
    else:
        parts.append(
            "Emergence was abrupt: no detectable convergence toward the "
            "unconscious content before the reveal trigger was met."
        )

    if shift_dir == "attuned":
        parts.append(
            f"Therapist shifted after reveal (score: {shift_score:.3f}, direction: attuned). "
            "Longer, more engaged responses: consistent with registering "
            "new material and deepening the encounter."
        )
    elif shift_dir == "withdrawn":
        parts.append(
            f"Therapist shifted after reveal (score: {shift_score:.3f}, direction: withdrawn). "
            "Shorter, more distant responses: may indicate defensiveness "
            "or difficulty integrating the unconscious material."
        )
    else:
        parts.append(
            "No detectable shift in therapist behaviour after the reveal."
        )

    return " ".join(parts)