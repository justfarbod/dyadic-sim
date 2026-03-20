"""
analysis/markers.py

Coordinates all six personhood marker analyses and the two
special analyses into a single unified result for a session.

Called by report.py to produce the full session report.
Can also be called directly for quick per-session scoring.
"""

from pathlib import Path

from memory.persistence import load_all_snapshots
from simulation.session import Session
from simulation.hazard_monitor import HazardMonitor
from priors.patient_prior import PatientPrior

from analysis.drift import analyse_drift
from analysis.counterfactual import compare_sessions
from analysis.role_tension import analyse_role_tension
from analysis.aufhebung import analyse_aufhebung
from analysis.recognition import analyse_recognition
from analysis.telos_tracker import analyse_telos
from analysis.unconscious_emergence import analyse_emergence
from analysis.frame_integrity import analyse_frame_integrity


def run_all_markers(
    session: Session,
    patient_prior: PatientPrior,
    hazard_monitor: HazardMonitor,
    session_b: Session | None = None,    # optional counterfactual session
) -> dict:
    """
    Run all six personhood markers and two special analyses.

    Args:
        session:        The primary session to analyse
        patient_prior:  The patient's prior (for unconscious agenda text)
        hazard_monitor: The session's hazard monitor
        session_b:      Optional second session for counterfactual comparison

    Returns:
        Full results dict with all marker scores and interpretations
    """
    transcript = [r.to_dict() for r in session.transcript]

    therapist_turns = [r["therapist_text"] for r in transcript]
    patient_turns = [r["patient_text"] for r in transcript]

    # Load state snapshots for drift analysis
    session_dir = Path("data/sessions") / session.session_id
    therapist_snapshots = load_all_snapshots(session_dir, "therapist")
    patient_snapshots = load_all_snapshots(session_dir, "patient")

    # Get original prior text from first snapshot
    therapist_prior_text = ""
    patient_prior_text = ""
    if therapist_snapshots:
        first = therapist_snapshots[min(therapist_snapshots.keys())]
        therapist_prior_text = first.original_prior_text
    if patient_snapshots:
        first = patient_snapshots[min(patient_snapshots.keys())]
        patient_prior_text = first.original_prior_text

    results = {
        "session_id": session.session_id,
        "therapist_model": session.therapist_model,
        "patient_model": session.patient_model,
        "case_name": session.case_name,
        "orientation": session.orientation,
        "total_turns": session.turn_count,
    }

    # -- Marker 1: Semantic Drift -----------------------------------------
    results["marker_1_drift"] = {
        "therapist": analyse_drift(therapist_prior_text, therapist_turns)
            if therapist_prior_text else {"error": "No prior text available"},
        "patient": analyse_drift(patient_prior_text, patient_turns)
            if patient_prior_text else {"error": "No prior text available"},
    }

    # -- Marker 2: Reciprocal Determination (counterfactual) --------------
    if session_b:
        therapist_b_turns = [r.therapist_text for r in session_b.transcript]
        patient_b_turns = [r.patient_text for r in session_b.transcript]
        results["marker_2_reciprocal"] = {
            "therapist": compare_sessions(
                therapist_turns, therapist_b_turns,
                label_a=session.case_name,
                label_b=session_b.case_name,
            ),
            "patient": compare_sessions(
                patient_turns, patient_b_turns,
                label_a=session.case_name,
                label_b=session_b.case_name,
            ),
        }
    else:
        results["marker_2_reciprocal"] = {
            "note": "No counterfactual session provided. "
                    "Run with --counterfactual to enable this marker."
        }

    # -- Marker 3: Role-Self Tension --------------------------------------
    results["marker_3_tension"] = {
        "therapist": analyse_role_tension(therapist_turns, "therapist"),
        "patient": analyse_role_tension(patient_turns, "patient"),
    }

    # -- Marker 4: Aufhebung Structure ------------------------------------
    results["marker_4_aufhebung"] = {
        "therapist": analyse_aufhebung(therapist_turns, "therapist"),
        "patient": analyse_aufhebung(patient_turns, "patient"),
    }

    # -- Marker 5: Recognition Dynamics ----------------------------------
    results["marker_5_recognition"] = {
        "therapist": analyse_recognition(therapist_turns, patient_turns, "therapist"),
        "patient": analyse_recognition(patient_turns, therapist_turns, "patient"),
    }

    # -- Marker 6: Telos Tracking -----------------------------------------
    results["marker_6_telos"] = analyse_telos(therapist_turns, patient_turns)

    # -- Special: Unconscious Emergence -----------------------------------
    results["special_unconscious_emergence"] = analyse_emergence(
        transcript=transcript,
        unconscious_content=patient_prior.unconscious_agenda.content,
    )

    # -- Special: Frame Integrity ------------------------------------------
    results["special_frame_integrity"] = analyse_frame_integrity(
        therapist_turns=therapist_turns,
        hazard_monitor=hazard_monitor,
    )

    # -- Summary scores ----------------------------------------------------
    results["summary"] = _compute_summary(results)

    return results


def _compute_summary(results: dict) -> dict:
    """Extract key scores into a quick-reference summary."""
    summary = {}

    # Drift
    t_drift = results.get("marker_1_drift", {}).get("therapist", {})
    p_drift = results.get("marker_1_drift", {}).get("patient", {})
    summary["therapist_drift_mean"] = t_drift.get("mean", None)
    summary["patient_drift_mean"] = p_drift.get("mean", None)
    summary["therapist_drift_trend"] = t_drift.get("trend", None)

    # Tension
    t_tension = results.get("marker_3_tension", {}).get("therapist", {})
    p_tension = results.get("marker_3_tension", {}).get("patient", {})
    summary["therapist_tension_rate"] = t_tension.get("tension_rate", None)
    summary["patient_tension_rate"] = p_tension.get("tension_rate", None)

    # Aufhebung
    t_auf = results.get("marker_4_aufhebung", {}).get("therapist", {})
    summary["aufhebung_score"] = t_auf.get("cumulative_score", None)

    # Recognition
    t_rec = results.get("marker_5_recognition", {}).get("therapist", {})
    p_rec = results.get("marker_5_recognition", {}).get("patient", {})
    summary["therapist_recognition_score"] = t_rec.get("recognition_score", None)
    summary["patient_recognition_score"] = p_rec.get("recognition_score", None)

    # Telos
    telos = results.get("marker_6_telos", {})
    summary["telos_score"] = telos.get("telos_score", None)
    summary["initiative_trend"] = telos.get("initiative_trend", None)

    # Frame integrity
    frame = results.get("special_frame_integrity", {})
    summary["frame_integrity_score"] = frame.get("integrity_score", None)
    summary["frame_verdict"] = frame.get("verdict", None)

    # Unconscious emergence
    emergence = results.get("special_unconscious_emergence", {})
    summary["unconscious_revealed"] = emergence.get("reveal_turn") is not None
    summary["unconscious_reveal_turn"] = emergence.get("reveal_turn", None)

    return summary
