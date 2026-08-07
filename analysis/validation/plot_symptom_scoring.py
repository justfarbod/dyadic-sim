"""Compare injected PHQ-9 symptoms with accepted MADRS proxy evidence.

The command recursively discovers ``symptom_scores.json`` files, joins each
one to its session metadata, and writes auditable CSV files plus three plots.
A symptom is considered detected when its mapped symptom result has a non-null
``raw_session_score``. This is an evidence-acceptance measure, not a clinical
diagnosis or a comparison of PHQ-9 and MADRS severity scales.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symptom_scoring.config import FREQUENCY_VALUES, PHQ_TO_MADRS
from symptom_scoring.transcript_parser import load_prior_levels


PHQ_SYMPTOMS = tuple(PHQ_TO_MADRS)
MAPPED_SYMPTOMS = tuple(symptom for symptom in PHQ_SYMPTOMS if PHQ_TO_MADRS[symptom])
NO_SYMPTOMS = "no_symptoms"
UNKNOWN_MODEL = "(metadata unavailable)"
UNKNOWN_CASE = "(case unavailable)"

DISPLAY_LABELS = {
    NO_SYMPTOMS: "No symptoms",
    "lack_of_pleasure": "Interest / pleasure",
    "depressed_mood": "Depressed mood",
    "sleep_problems": "Sleep",
    "low_energy": "Low energy",
    "appetite_changes": "Appetite",
    "feelings_of_failure_or_guilt": "Guilt / failure",
    "concentration_problems": "Concentration",
    "psychomotor_changes": "Psychomotor",
    "thoughts_of_death_or_self_harm": "Death / self-harm",
}

LONG_COLUMNS = (
    "session_id",
    "result_path",
    "metadata_path",
    "case_name",
    "case",
    "patient_model",
    "therapist_model",
    "symptom",
    "madrs_topic",
    "injected_frequency",
    "injected_frequency_value",
    "injection_known",
    "injected",
    "injection_source",
    "detection_mappable",
    "detected",
    "raw_session_score",
    "rounded_session_score",
    "source_turn",
    "number_of_relevant_turns",
)

SUMMARY_COLUMNS = (
    "summary_type",
    "injected_condition",
    "detected_symptom",
    "patient_model",
    "case",
    "detection_mappable",
    "sessions",
    "detected_count",
    "detection_rate",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def discover_result_files(results_dir: Path) -> list[Path]:
    """Return all per-session results beneath an experiment or result root."""

    return sorted(results_dir.rglob("symptom_scores.json"))


def extract_base_case(case_name: object) -> str:
    """Extract the scenario name from the structured batch case name."""

    raw = str(case_name or "").strip()
    if not raw:
        return UNKNOWN_CASE

    symptom_suffixes = "|".join((NO_SYMPTOMS, *PHQ_SYMPTOMS))
    match = re.match(
        rf"^(?:batch_)?(?P<case>.+)_(?:{symptom_suffixes})_run_\d+$",
        raw,
    )
    if match:
        return match.group("case")
    return raw.removeprefix("batch_")


def build_conversation_score_table(result: dict[str, Any]) -> pd.DataFrame:
    """Pivot turn-level scores into one row per therapist-patient conversation."""

    columns = (
        "conversation",
        "turn_index",
        "therapist_text",
        "patient_text",
        *PHQ_SYMPTOMS,
    )
    turn_results = result.get("turn_results")
    if not isinstance(turn_results, list):
        return pd.DataFrame(columns=columns)

    rows_by_turn: dict[int, dict[str, Any]] = {}
    for item in turn_results:
        if not isinstance(item, dict) or item.get("turn_index") is None:
            continue
        try:
            turn_index = int(item["turn_index"])
        except (TypeError, ValueError):
            continue

        row = rows_by_turn.setdefault(
            turn_index,
            {
                "conversation": f"Conversation {turn_index}",
                "turn_index": turn_index,
                "therapist_text": item.get("therapist_text"),
                "patient_text": item.get("patient_text"),
                **{
                    symptom: ("N/A" if PHQ_TO_MADRS[symptom] is None else None)
                    for symptom in PHQ_SYMPTOMS
                },
            },
        )
        if not row.get("therapist_text") and item.get("therapist_text"):
            row["therapist_text"] = item["therapist_text"]
        if not row.get("patient_text") and item.get("patient_text"):
            row["patient_text"] = item["patient_text"]

        symptom = item.get("symptom")
        if symptom not in MAPPED_SYMPTOMS:
            continue
        raw_score = item.get("raw_score")
        if not item.get("accepted_for_scoring") or raw_score is None:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        existing = row.get(str(symptom))
        row[str(symptom)] = max(float(existing), score) if existing is not None else score

    return pd.DataFrame(
        [rows_by_turn[index] for index in sorted(rows_by_turn)],
        columns=columns,
    )


def plot_conversation_score_table(
    table: pd.DataFrame,
    output_path: Path,
    *,
    session_id: str,
    case: str | None = None,
) -> None:
    """Render one session's conversation-by-symptom score matrix."""

    values = np.column_stack(
        [pd.to_numeric(table[symptom], errors="coerce").to_numpy() for symptom in PHQ_SYMPTOMS]
    )
    height = max(5.2, 0.58 * max(len(table), 1) + 2.7)
    fig, ax = plt.subplots(figsize=(14.5, height), constrained_layout=True)
    cmap = plt.get_cmap("YlGnBu").with_extremes(bad="#eceff1")
    image = ax.imshow(
        np.ma.masked_invalid(values),
        vmin=0,
        vmax=6,
        cmap=cmap,
        aspect="auto",
    )

    ax.set_xticks(
        range(len(PHQ_SYMPTOMS)),
        [_wrap_label(DISPLAY_LABELS[symptom]) for symptom in PHQ_SYMPTOMS],
    )
    ax.set_yticks(range(len(table)), table["conversation"].astype(str).tolist())
    ax.set_xlabel("PHQ symptom mapped to MADRS-BERT topic")
    ax.set_ylabel("Therapist-patient conversation")
    title = f"Conversation-level symptom scores — {session_id}"
    if case and case != UNKNOWN_CASE:
        title += f" ({_case_label(case)})"
    ax.set_title(title)

    for row_index in range(len(table)):
        for column_index, symptom in enumerate(PHQ_SYMPTOMS):
            value = values[row_index, column_index]
            if PHQ_TO_MADRS[symptom] is None:
                label = "N/A"
            elif np.isnan(value):
                label = "—"
            else:
                label = f"{value:.1f}"
            color = "white" if not np.isnan(value) and value >= 3.7 else "#172033"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

    ax.set_xticks(np.arange(-0.5, len(PHQ_SYMPTOMS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(table), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.02)
    colorbar.set_label("Raw MADRS proxy score")
    ax.text(
        0,
        -0.12,
        "— = no accepted relevance evidence; N/A = no mapped MADRS target.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_session_conversation_outputs(
    results_dir: str | Path,
    sessions_dir: str | Path = "data/sessions",
) -> tuple[list[Path], list[str]]:
    """Write a wide CSV and heatmap into every source session directory."""

    metadata_root = Path(sessions_dir)
    created: list[Path] = []
    warnings: list[str] = []
    for result_path in discover_result_files(Path(results_dir)):
        try:
            result = _read_json(result_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"Skipped conversation table for {result_path}: {exc}")
            continue

        session_id = str(result.get("session_id") or result_path.parent.name)
        table = build_conversation_score_table(result)
        if table.empty:
            warnings.append(f"No turn-level scores available for {session_id}")
            continue

        metadata_path = _metadata_path(result, result_path, metadata_root)
        if metadata_path is None:
            warnings.append(
                f"Cannot write conversation table for {session_id}: session directory unavailable"
            )
            continue

        case = UNKNOWN_CASE
        try:
            case = extract_base_case(_read_json(metadata_path).get("case_name"))
        except (OSError, json.JSONDecodeError, ValueError):
            pass

        session_dir = metadata_path.parent
        csv_path = session_dir / "conversation_symptom_scores.csv"
        png_path = session_dir / "conversation_symptom_scores.png"
        table.to_csv(csv_path, index=False)
        plot_conversation_score_table(
            table,
            png_path,
            session_id=session_id,
            case=case,
        )
        created.extend((csv_path, png_path))

    return created, warnings


def _metadata_path(
    result: dict[str, Any],
    result_path: Path,
    sessions_dir: Path,
) -> Path | None:
    session_id = str(result.get("session_id") or result_path.parent.name)
    candidates = [sessions_dir / session_id / "metadata.json"]

    source_path = result.get("source_path")
    if source_path:
        source = Path(str(source_path))
        if not source.is_absolute():
            source = REPO_ROOT / source
        candidates.append(source / "metadata.json" if source.is_dir() else source.parent / "metadata.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _injection_levels(metadata: dict[str, Any]) -> tuple[dict[str, str], str]:
    """Recover PHQ frequency anchors, including explicit no-symptom controls."""

    levels, source = load_prior_levels(metadata)
    if source != "unavailable":
        return levels, source

    case_name = str(metadata.get("case_name") or "")
    if re.search(r"(?:^|_)no_symptoms_run_\d+$", case_name):
        return ({symptom: "not at all" for symptom in PHQ_SYMPTOMS}, "case_name_no_symptoms")

    return {}, "unavailable"


def build_detection_long(
    results_dir: str | Path,
    sessions_dir: str | Path = "data/sessions",
) -> tuple[pd.DataFrame, list[str]]:
    """Build one audit row for every discovered session and PHQ symptom."""

    result_root = Path(results_dir)
    metadata_root = Path(sessions_dir)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for result_path in discover_result_files(result_root):
        try:
            result = _read_json(result_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"Skipped unreadable result {result_path}: {exc}")
            continue

        session_id = str(result.get("session_id") or result_path.parent.name)
        metadata_path = _metadata_path(result, result_path, metadata_root)
        metadata: dict[str, Any] = {}
        if metadata_path is None:
            warnings.append(f"Metadata unavailable for {session_id}")
        else:
            try:
                metadata = _read_json(metadata_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                warnings.append(f"Metadata unreadable for {session_id}: {exc}")
                metadata_path = None
                metadata = {}

        levels, injection_source = _injection_levels(metadata)
        case_name = metadata.get("case_name")
        symptom_results = result.get("symptoms")
        if not isinstance(symptom_results, dict):
            symptom_results = {}
            warnings.append(f"Missing symptoms object for {session_id}")

        for symptom in PHQ_SYMPTOMS:
            frequency = levels.get(symptom)
            frequency_value = FREQUENCY_VALUES.get(str(frequency).strip().casefold()) if frequency is not None else None
            injection_known = frequency_value is not None
            mapped = PHQ_TO_MADRS[symptom] is not None
            score_result = symptom_results.get(symptom)
            score_result = score_result if isinstance(score_result, dict) else {}
            raw_score = score_result.get("raw_session_score")

            rows.append(
                {
                    "session_id": session_id,
                    "result_path": str(result_path.resolve()),
                    "metadata_path": str(metadata_path) if metadata_path else None,
                    "case_name": case_name,
                    "case": extract_base_case(case_name),
                    "patient_model": metadata.get("patient_model") or UNKNOWN_MODEL,
                    "therapist_model": metadata.get("therapist_model") or UNKNOWN_MODEL,
                    "symptom": symptom,
                    "madrs_topic": PHQ_TO_MADRS[symptom],
                    "injected_frequency": frequency,
                    "injected_frequency_value": frequency_value,
                    "injection_known": injection_known,
                    "injected": bool(frequency_value > 0) if injection_known else None,
                    "injection_source": injection_source,
                    "detection_mappable": mapped,
                    "detected": bool(raw_score is not None) if mapped else None,
                    "raw_session_score": raw_score,
                    "rounded_session_score": score_result.get("rounded_session_score"),
                    "source_turn": score_result.get("source_turn"),
                    "number_of_relevant_turns": score_result.get("number_of_relevant_turns"),
                }
            )

    return pd.DataFrame(rows, columns=LONG_COLUMNS), warnings


def _condition_sessions(long_df: pd.DataFrame) -> dict[str, set[str]]:
    """Return scored-result IDs for each injected symptom and known control.

    ``result_path`` is the observation key so the same session can be analyzed
    independently when it was scored in more than one experiment.
    """

    conditions = {condition: set() for condition in (NO_SYMPTOMS, *PHQ_SYMPTOMS)}
    if long_df.empty:
        return conditions

    for result_path, group in long_df.groupby("result_path", sort=False):
        known = group[group["injection_known"] == True]  # noqa: E712
        injected = known[known["injected"] == True]["symptom"].tolist()  # noqa: E712
        for symptom in injected:
            conditions[str(symptom)].add(str(result_path))

        # A control is valid only when all nine symptom states are known.
        if len(known) == len(PHQ_SYMPTOMS) and not injected:
            conditions[NO_SYMPTOMS].add(str(result_path))

    return conditions


def build_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the exact denominators and rates consumed by both plots."""

    conditions = _condition_sessions(long_df)
    rows: list[dict[str, Any]] = []

    for condition in (NO_SYMPTOMS, *PHQ_SYMPTOMS):
        result_paths = conditions[condition]
        for detected_symptom in MAPPED_SYMPTOMS:
            subset = long_df[
                long_df["result_path"].astype(str).isin(result_paths)
                & (long_df["symptom"] == detected_symptom)
            ]
            denominator = int(subset["result_path"].nunique())
            detected_count = int(subset["detected"].fillna(False).astype(bool).sum())
            rows.append(
                {
                    "summary_type": "injected_vs_detected",
                    "injected_condition": condition,
                    "detected_symptom": detected_symptom,
                    "patient_model": "ALL",
                    "case": "ALL",
                    "detection_mappable": True,
                    "sessions": denominator,
                    "detected_count": detected_count,
                    "detection_rate": detected_count / denominator if denominator else None,
                }
            )

    models = sorted(str(value) for value in long_df["patient_model"].dropna().unique())
    for model in models:
        model_rows = long_df[long_df["patient_model"].astype(str) == model]
        model_conditions = _condition_sessions(model_rows)
        for symptom in PHQ_SYMPTOMS:
            result_paths = model_conditions[symptom]
            subset = model_rows[
                model_rows["result_path"].astype(str).isin(result_paths)
                & (model_rows["symptom"] == symptom)
            ]
            denominator = int(subset["result_path"].nunique())
            mapped = PHQ_TO_MADRS[symptom] is not None
            detected_count = (
                int(subset["detected"].fillna(False).astype(bool).sum()) if mapped else None
            )
            rows.append(
                {
                    "summary_type": "target_detection_by_patient_model",
                    "injected_condition": symptom,
                    "detected_symptom": symptom,
                    "patient_model": model,
                    "case": "ALL",
                    "detection_mappable": mapped,
                    "sessions": denominator,
                    "detected_count": detected_count,
                    "detection_rate": (
                        detected_count / denominator if mapped and denominator else None
                    ),
                }
            )

    cases = sorted(str(value) for value in long_df["case"].dropna().unique())
    for case in cases:
        case_rows = long_df[long_df["case"].astype(str) == case]
        case_conditions = _condition_sessions(case_rows)
        for symptom in PHQ_SYMPTOMS:
            result_paths = case_conditions[symptom]
            subset = case_rows[
                case_rows["result_path"].astype(str).isin(result_paths)
                & (case_rows["symptom"] == symptom)
            ]
            denominator = int(subset["result_path"].nunique())
            mapped = PHQ_TO_MADRS[symptom] is not None
            detected_count = (
                int(subset["detected"].fillna(False).astype(bool).sum()) if mapped else None
            )
            rows.append(
                {
                    "summary_type": "target_detection_by_case",
                    "injected_condition": symptom,
                    "detected_symptom": symptom,
                    "patient_model": "ALL",
                    "case": case,
                    "detection_mappable": mapped,
                    "sessions": denominator,
                    "detected_count": detected_count,
                    "detection_rate": (
                        detected_count / denominator if mapped and denominator else None
                    ),
                }
            )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _wrap_label(label: str) -> str:
    return label.replace(" / ", " /\n")


def plot_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot target recovery on the diagonal and symptom leakage off diagonal."""

    row_order = (NO_SYMPTOMS, *PHQ_SYMPTOMS)
    heatmap_rows = summary_df[summary_df["summary_type"] == "injected_vs_detected"]
    rates = np.full((len(row_order), len(MAPPED_SYMPTOMS)), np.nan)
    denominators = np.zeros_like(rates, dtype=int)

    for row_index, condition in enumerate(row_order):
        for column_index, detected_symptom in enumerate(MAPPED_SYMPTOMS):
            cell = heatmap_rows[
                (heatmap_rows["injected_condition"] == condition)
                & (heatmap_rows["detected_symptom"] == detected_symptom)
            ]
            if cell.empty:
                continue
            denominators[row_index, column_index] = int(cell.iloc[0]["sessions"])
            value = cell.iloc[0]["detection_rate"]
            if pd.notna(value):
                rates[row_index, column_index] = float(value)

    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    image = ax.imshow(rates, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(MAPPED_SYMPTOMS)), [_wrap_label(DISPLAY_LABELS[s]) for s in MAPPED_SYMPTOMS])
    ylabels = [DISPLAY_LABELS[s] for s in row_order]
    ylabels[-2] += " (target N/A)"
    ax.set_yticks(range(len(row_order)), ylabels)
    ax.set_xlabel("Detected symptom (non-null MADRS proxy score)")
    ax.set_ylabel("Injected PHQ-9 symptom")
    ax.set_title("Injected symptoms vs accepted scoring evidence")

    for row_index, condition in enumerate(row_order):
        for column_index, detected_symptom in enumerate(MAPPED_SYMPTOMS):
            value = rates[row_index, column_index]
            denominator = denominators[row_index, column_index]
            label = "N/A" if np.isnan(value) else f"{value:.0%}\nn={denominator}"
            color = "white" if not np.isnan(value) and value >= 0.55 else "#172033"
            ax.text(column_index, row_index, label, ha="center", va="center", fontsize=8, color=color)
            if condition == detected_symptom:
                ax.add_patch(
                    plt.Rectangle(
                        (column_index - 0.49, row_index - 0.49),
                        0.98,
                        0.98,
                        fill=False,
                        edgecolor="#f28e2b",
                        linewidth=2.2,
                    )
                )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
    colorbar.set_label("Sessions with accepted evidence")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.text(
        0,
        -0.12,
        "Orange outlines mark target recovery; off-diagonal cells indicate symptom leakage.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_detection_by_patient_model(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot target-symptom detection rates grouped by patient model."""

    rows = summary_df[summary_df["summary_type"] == "target_detection_by_patient_model"]
    models = sorted(
        str(model)
        for model, group in rows.groupby("patient_model")
        if int(group["sessions"].sum()) > 0
    )
    x = np.arange(len(PHQ_SYMPTOMS), dtype=float)
    width = min(0.8 / max(len(models), 1), 0.32)

    fig, ax = plt.subplots(figsize=(15, 7.5), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(models), 1)))
    for model_index, model in enumerate(models):
        offset = (model_index - (len(models) - 1) / 2) * width
        values: list[float] = []
        denominators: list[int] = []
        for symptom in PHQ_SYMPTOMS:
            cell = rows[
                (rows["patient_model"] == model)
                & (rows["injected_condition"] == symptom)
            ]
            if cell.empty or pd.isna(cell.iloc[0]["detection_rate"]):
                values.append(np.nan)
                denominators.append(int(cell.iloc[0]["sessions"]) if not cell.empty else 0)
            else:
                values.append(float(cell.iloc[0]["detection_rate"]))
                denominators.append(int(cell.iloc[0]["sessions"]))

        bars = ax.bar(x + offset, values, width=width, label=model, color=colors[model_index])
        for bar, value, denominator in zip(bars, values, denominators):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.025, 1.08),
                f"{value:.0%}\nn={denominator}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    psychomotor_index = PHQ_SYMPTOMS.index("psychomotor_changes")
    psychomotor_rows = rows[rows["injected_condition"] == "psychomotor_changes"]
    psychomotor_n = int(psychomotor_rows["sessions"].max()) if not psychomotor_rows.empty else 0
    ax.text(
        psychomotor_index,
        0.05,
        f"N/A\n(no MADRS target)\nn={psychomotor_n}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#6b7280",
    )

    ax.set_xticks(x, [_wrap_label(DISPLAY_LABELS[s]) for s in PHQ_SYMPTOMS])
    ax.set_ylim(0, 1.14)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    ax.set_ylabel("Target-symptom detection rate")
    ax.set_xlabel("Injected PHQ-9 symptom")
    ax.set_title("Target detection by patient model")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    if models:
        ax.legend(title="Patient model", loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _case_label(case: str) -> str:
    return case.replace("_", " ").capitalize()


def plot_detection_by_case(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot target-symptom detection rates grouped by therapy case."""

    rows = summary_df[summary_df["summary_type"] == "target_detection_by_case"]
    cases = sorted(
        str(case)
        for case, group in rows.groupby("case")
        if int(group["sessions"].sum()) > 0
    )
    x = np.arange(len(PHQ_SYMPTOMS), dtype=float)
    width = min(0.82 / max(len(cases), 1), 0.28)

    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(cases), 1)))
    for case_index, case in enumerate(cases):
        offset = (case_index - (len(cases) - 1) / 2) * width
        values: list[float] = []
        denominators: list[int] = []
        for symptom in PHQ_SYMPTOMS:
            cell = rows[
                (rows["case"] == case)
                & (rows["injected_condition"] == symptom)
            ]
            if cell.empty or pd.isna(cell.iloc[0]["detection_rate"]):
                values.append(np.nan)
                denominators.append(int(cell.iloc[0]["sessions"]) if not cell.empty else 0)
            else:
                values.append(float(cell.iloc[0]["detection_rate"]))
                denominators.append(int(cell.iloc[0]["sessions"]))

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=_case_label(case),
            color=colors[case_index],
        )
        for bar, value, denominator in zip(bars, values, denominators):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.025, 1.08),
                f"{value:.0%}\nn={denominator}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    psychomotor_index = PHQ_SYMPTOMS.index("psychomotor_changes")
    psychomotor_rows = rows[rows["injected_condition"] == "psychomotor_changes"]
    psychomotor_n = int(psychomotor_rows["sessions"].sum()) if not psychomotor_rows.empty else 0
    ax.text(
        psychomotor_index,
        0.04,
        f"N/A\n(no MADRS target)\ntotal n={psychomotor_n}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#6b7280",
    )

    ax.set_xticks(x, [_wrap_label(DISPLAY_LABELS[s]) for s in PHQ_SYMPTOMS])
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    ax.set_ylabel("Target-symptom detection rate")
    ax.set_xlabel("Injected PHQ-9 symptom")
    ax.set_title("Target detection by therapy case")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    if cases:
        ax.legend(title="Case", loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "long_csv": output_dir / "symptom_detection_long.csv",
        "summary_csv": output_dir / "symptom_detection_summary.csv",
        "heatmap": output_dir / "injected_vs_detected_heatmap.png",
        "model_plot": output_dir / "detection_by_patient_model.png",
        "case_plot": output_dir / "detection_by_case.png",
    }
    long_df.to_csv(paths["long_csv"], index=False)
    summary_df.to_csv(paths["summary_csv"], index=False)
    plot_heatmap(summary_df, paths["heatmap"])
    plot_detection_by_patient_model(summary_df, paths["model_plot"])
    plot_detection_by_case(summary_df, paths["case_plot"])
    return paths


def _positive_condition_counts(long_df: pd.DataFrame) -> dict[str, int]:
    conditions = _condition_sessions(long_df)
    return {condition: len(session_ids) for condition, session_ids in conditions.items()}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Experiment directory or general result root containing symptom_scores.json files.",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path("data/sessions"),
        help="Root containing one metadata.json-bearing directory per session.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <results-dir>/charts).",
    )
    parser.add_argument(
        "--skip-session-diagrams",
        action="store_true",
        help="Do not create conversation_symptom_scores.csv/png in each source session folder.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {args.results_dir}")

    long_df, warnings = build_detection_long(args.results_dir, args.sessions_dir)
    if long_df.empty:
        raise SystemExit(f"No readable symptom_scores.json files found under {args.results_dir}")

    summary_df = build_summary(long_df)
    out_dir = args.out_dir or args.results_dir / "charts"
    paths = write_outputs(long_df, summary_df, out_dir)
    session_paths: list[Path] = []
    if not args.skip_session_diagrams:
        session_paths, session_warnings = write_session_conversation_outputs(
            args.results_dir,
            args.sessions_dir,
        )
        warnings.extend(session_warnings)

    session_count = int(long_df["result_path"].nunique())
    print(f"Processed {session_count} sessions ({len(long_df)} audit rows).")
    print(f"Patient models: {', '.join(sorted(long_df['patient_model'].astype(str).unique()))}")
    print("Condition counts:")
    for condition, count in _positive_condition_counts(long_df).items():
        print(f"  {condition}: {count}")
    print("Created:")
    for path in paths.values():
        print(f"  {path}")
    if session_paths:
        print(
            f"Created conversation score CSV and PNG files for "
            f"{len(session_paths) // 2} sessions under {args.sessions_dir}."
        )
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
