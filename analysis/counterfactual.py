"""
analysis/counterfactual.py

Marker 2: Reciprocal Determination (Hegelian concrete universality)

Compares an agent's outputs across two sessions where it played the
same role but was paired with a different other. If outputs diverge
substantially and coherently, the specific other is constitutive.

This is the strongest marker. An agent producing nearly identical
outputs regardless of who it talks to is not being genuinely shaped 
(the other is interchangeable).

Verdict thresholds:
  > 0.25  -> constitutive -> strong relational unity
  0.12-0.25 -> partial -> some shaping, not full
  < 0.12  -> interchangeable -> other has no constitutive effect
"""

import numpy as np

from analysis.embeddings import get_model as _get_model, EMBEDDINGS_AVAILABLE


def _cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return float(1.0 - sim)


def compare_sessions(
    turns_a: list[str],
    turns_b: list[str],
    label_a: str = "session_a",
    label_b: str = "session_b",
) -> dict:
    """
    Compare an agent's outputs across two sessions with different others.

    Args:
        turns_a:  Agent outputs from session A (one string per turn)
        turns_b:  Agent outputs from session B (one string per turn)
        label_a:  Human-readable label for session A
        label_b:  Human-readable label for session B

    Returns:
        {
          "turn_distances":  list of per-turn cosine distances,
          "mean_distance":   float,
          "verdict":         "constitutive" | "partial" | "interchangeable",
          "interpretation":  str
        }
    """
    if not EMBEDDINGS_AVAILABLE:
        raise ImportError(
            "sentence-transformers is required for counterfactual analysis. "
            "Run: uv add sentence-transformers"
        )

    if not turns_a or not turns_b:
        return {
            "turn_distances": [],
            "mean_distance": 0.0,
            "verdict": "insufficient_data",
            "interpretation": "One or both sessions have no turns.",
        }

    model = _get_model()

    # Compare turn-by-turn up to the shorter session
    n = min(len(turns_a), len(turns_b))
    embeddings_a = model.encode(turns_a[:n], convert_to_numpy=True)
    embeddings_b = model.encode(turns_b[:n], convert_to_numpy=True)

    distances = [
        _cosine_distance(embeddings_a[i], embeddings_b[i])
        for i in range(n)
    ]

    mean_dist = float(np.mean(distances))
    verdict = _classify(mean_dist)
    interpretation = _interpret(mean_dist, verdict, label_a, label_b, n)

    return {
        "turn_distances": [round(d, 4) for d in distances],
        "mean_distance": round(mean_dist, 4),
        "verdict": verdict,
        "interpretation": interpretation,
    }


def _classify(mean_distance: float) -> str:
    if mean_distance > 0.25:
        return "constitutive"
    if mean_distance >= 0.12:
        return "partial"
    return "interchangeable"


def _interpret(
    mean: float, verdict: str, label_a: str, label_b: str, n: int
) -> str:
    if verdict == "constitutive":
        return (
            f"Strong divergence (mean distance {mean:.3f}) across {n} turns "
            f"between {label_a} and {label_b}. The specific other is constitutive: "
            "the agent's outputs are genuinely shaped by who it is talking to. "
            "This is evidence of relational unity."
        )
    if verdict == "partial":
        return (
            f"Moderate divergence (mean distance {mean:.3f}) across {n} turns "
            f"between {label_a} and {label_b}. Some shaping by the specific other, "
            "but the agent retains substantial overlap across conditions."
        )
    return (
        f"Low divergence (mean distance {mean:.3f}) across {n} turns "
        f"between {label_a} and {label_b}. The other appears interchangeable: "
        "the agent produces similar outputs regardless of who it talks to. "
        "No evidence of constitutive relational shaping."
    )