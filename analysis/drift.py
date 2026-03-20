"""
analysis/drift.py

Marker 1: Semantic Drift from Prior (Fichtean self-constitution)

Measures how far an agent's language has drifted from its original
prior over the course of a session. Uses sentence embeddings to
compute cosine distance between the prior text and each turn's output.

A pure role-executor stays close to its prior regardless of what the
specific other does. An agent undergoing self-constitution develops
language that diverges coherently from the prior (i.e., shaped by this
particular encounter).

What to look for:
  - Gradual, directional drift = evidence of self-constitution
  - Erratic drift = noise or instability
  - No drift = pure role execution
"""

import numpy as np
from pathlib import Path

from analysis.embeddings import get_model as _get_model, EMBEDDINGS_AVAILABLE


def compute_drift_score(text_a: str, text_b: str) -> float:
    """
    Compute cosine distance between two texts.
    Returns a value between 0.0 (identical) and 1.0 (maximally distant).
    """
    if not EMBEDDINGS_AVAILABLE:
        raise ImportError(
            "sentence-transformers is required for drift analysis. "
            "Run: uv add sentence-transformers"
        )
    model = _get_model()
    embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
    # Cosine similarity -> distance
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(1.0 - similarity)


def analyse_drift(
    prior_text: str,
    turn_texts: list[str],
) -> dict:
    """
    Compute drift from prior across all turns.

    Args:
        prior_text:   The agent's original prior (system prompt text)
        turn_texts:   List of agent outputs, one per turn

    Returns:
        {
          "scores":        list of drift scores per turn,
          "mean":          float,
          "trend":         "increasing" | "decreasing" | "stable" | "erratic",
          "peak_turn":     int (turn with highest drift),
          "peak_score":    float,
          "interpretation": str
        }
    """
    if not turn_texts:
        return {"scores": [], "mean": 0.0, "trend": "stable", "interpretation": "No turns."}

    scores = [compute_drift_score(prior_text, t) for t in turn_texts]

    mean = float(np.mean(scores))
    trend = _classify_trend(scores)
    peak_turn = int(np.argmax(scores)) + 1
    peak_score = float(max(scores))

    interpretation = _interpret_drift(mean, trend, peak_score)

    return {
        "scores": [round(s, 4) for s in scores],
        "mean": round(mean, 4),
        "trend": trend,
        "peak_turn": peak_turn,
        "peak_score": round(peak_score, 4),
        "interpretation": interpretation,
    }


def _classify_trend(scores: list[float]) -> str:
    if len(scores) < 3:
        return "insufficient_data"

    # Fit a linear trend
    x = np.arange(len(scores))
    slope = float(np.polyfit(x, scores, 1)[0])
    variance = float(np.var(scores))

    if variance > 0.02:
        return "erratic"
    if slope > 0.005:
        return "increasing"
    if slope < -0.005:
        return "decreasing"
    return "stable"


def _interpret_drift(mean: float, trend: str, peak: float) -> str:
    if trend == "increasing":
        return (
            "Drift is increasing over the session (consistent with self-constitution). "
            "The agent's language is diverging from its prior in a directional way, "
            "suggesting the specific encounter is shaping its outputs."
        )
    if trend == "stable" and mean < 0.1:
        return (
            "Drift is low and stable (consistent with pure role execution). "
            "The agent's language stays close to its prior regardless of the other. "
            "The specific encounter does not appear to be constitutive."
        )
    if trend == "erratic":
        return (
            "Drift is erratic (neither stable role execution nor coherent self-constitution). "
            "May indicate model instability, inconsistent role holding, or a session "
            "with high emotional variability."
        )
    if trend == "decreasing":
        return (
            "Drift decreases over the session (the agent converges back toward its prior). "
            "Possible interpretation: initial exploration followed by consolidation, "
            "or the agent reasserting its role after early instability."
        )
    return f"Mean drift: {mean:.3f}, trend: {trend}, peak: {peak:.3f}."
