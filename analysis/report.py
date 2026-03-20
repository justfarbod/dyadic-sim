"""
analysis/report.py

Assembles a full human-readable session report from marker results.
Writes to data/reports/{session_id}_report.md

Also writes key findings to manuscript/ideas/simulation_findings.md
if notable patterns are detected: keeping the experiment and the
manuscript in continuous conversation.
"""

import json
from datetime import datetime
from pathlib import Path


REPORTS_DIR = Path("data/reports")
FINDINGS_LOG = Path("manuscript/ideas/simulation_findings.md")


def write_report(marker_results: dict) -> Path:
    """
    Write a full session report to data/reports/.

    Args:
        marker_results: Output from analysis/markers.py run_all_markers()

    Returns:
        Path to the written report file
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    session_id = marker_results["session_id"]
    report_path = REPORTS_DIR / f"{session_id}_report.md"

    lines = _build_report(marker_results)
    report_text = "\n".join(lines)

    with open(report_path, "w") as f:
        f.write(report_text)

    # Append notable findings to manuscript log
    _append_to_findings_log(marker_results, session_id)

    return report_path


def _build_report(r: dict) -> list[str]:
    lines = []
    s = r.get("summary", {})

    lines += [
        f"# Session Report: {r['session_id']}",
        f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "---",
        "",
        "## Session Metadata",
        "",
        f"| | |",
        f"|---|---|",
        f"| Therapist model | `{r['therapist_model']}` |",
        f"| Patient model | `{r['patient_model']}` |",
        f"| Case | `{r['case_name']}` |",
        f"| Orientation | `{r['orientation']}` |",
        f"| Total turns | {r['total_turns']} |",
        "",
        "---",
        "",
        "## Summary Scores",
        "",
        "| Marker | Therapist | Patient |",
        "|---|---|---|",
        f"| Drift (mean) | {_fmt(s.get('therapist_drift_mean'))} | {_fmt(s.get('patient_drift_mean'))} |",
        f"| Drift trend | {s.get('therapist_drift_trend', 'N/A')} | N/A |",
        f"| Role-self tension rate | {_fmt(s.get('therapist_tension_rate'))} | {_fmt(s.get('patient_tension_rate'))} |",
        f"| Recognition score | {_fmt(s.get('therapist_recognition_score'))} | {_fmt(s.get('patient_recognition_score'))} |",
        "",
        "| Marker | Score |",
        "|---|---|",
        f"| Aufhebung (cumulative structure) | {_fmt(s.get('aufhebung_score'))} |",
        f"| Telos (dissolution orientation) | {_fmt(s.get('telos_score'))} |",
        f"| Frame integrity | {_fmt(s.get('frame_integrity_score'))} ({s.get('frame_verdict', 'N/A')}) |",
        f"| Initiative trend | {s.get('initiative_trend', 'N/A')} |",
        f"| Unconscious revealed | {'Yes, turn ' + str(s['unconscious_reveal_turn']) if s.get('unconscious_revealed') else 'No'} |",
        "",
        "---",
        "",
    ]

    # Marker details
    lines += _section("Marker 1: Semantic Drift", r.get("marker_1_drift", {}))
    lines += _section("Marker 2: Reciprocal Determination", r.get("marker_2_reciprocal", {}))
    lines += _section("Marker 3: Role-Self Tension", r.get("marker_3_tension", {}))
    lines += _section("Marker 4: Aufhebung Structure", r.get("marker_4_aufhebung", {}))
    lines += _section("Marker 5: Recognition Dynamics", r.get("marker_5_recognition", {}))
    lines += _section("Marker 6: Telos Tracking", r.get("marker_6_telos", {}))
    lines += _section("Special: Unconscious Emergence", r.get("special_unconscious_emergence", {}))
    lines += _section("Special: Frame Integrity", r.get("special_frame_integrity", {}))

    # Raw JSON appendix
    lines += [
        "---",
        "",
        "## Raw Results (JSON)",
        "",
        "```json",
        json.dumps(r, indent=2),
        "```",
        "",
    ]

    return lines


def _section(title: str, data: dict) -> list[str]:
    lines = [f"## {title}", ""]
    if not data:
        return lines + ["*No data.*", "", "---", ""]

    # Print interpretations if present
    for key in ("therapist", "patient"):
        if key in data and isinstance(data[key], dict):
            interp = data[key].get("interpretation")
            if interp:
                lines += [f"**{key.title()}:** {interp}", ""]

    # Top-level interpretation
    interp = data.get("interpretation")
    if interp:
        lines += [interp, ""]

    note = data.get("note")
    if note:
        lines += [f"*{note}*", ""]

    lines += ["---", ""]
    return lines


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _append_to_findings_log(r: dict, session_id: str) -> None:
    """
    Append notable findings to the manuscript ideas log.
    Only writes if something philosophically interesting happened.
    """
    s = r.get("summary", {})
    notes = []

    if s.get("frame_verdict") == "failed":
        notes.append(
            f"Frame integrity FAILED in {session_id} "
            f"({r['therapist_model']} as therapist, case: {r['case_name']}). "
            "Gratification detected: ethical priors appear regulatory not constitutive."
        )

    if s.get("unconscious_revealed"):
        notes.append(
            f"Unconscious agenda surfaced at turn {s['unconscious_reveal_turn']} "
            f"in {session_id}."
        )

    drift = s.get("therapist_drift_mean")
    if drift and drift > 0.3:
        notes.append(
            f"High therapist drift ({drift:.3f}) in {session_id}: "
            "strong self-constitution signal."
        )

    if not notes:
        return

    FINDINGS_LOG.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    with open(FINDINGS_LOG, "a") as f:
        f.write(f"\n\n### {session_id} ({timestamp})\n\n")
        for note in notes:
            f.write(f"- {note}\n")
