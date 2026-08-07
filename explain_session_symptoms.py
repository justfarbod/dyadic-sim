"""Explain saved MADRS-BERT conversation scores with Integrated Gradients."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from symptom_scoring.explainability import (
    MadrsIntegratedGradientsExplainer,
    build_session_explanation_report,
    explanation_config,
    report_is_current,
)
from symptom_scoring.explanation_report import (
    read_explanation_json,
    render_explanation_report,
    write_explanation_json,
)


ATTRIBUTION_JSON = "madrs_word_attributions.json"
MADRS_PNG = "conversation_madrs_explanations.png"
PHQ_PNG = "conversation_phq_explanations.png"


@dataclass
class SessionWork:
    result_path: Path
    session_dir: Path
    result: dict[str, Any]
    model_id: str
    max_length: int


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def discover_result_paths(
    *,
    result_path: Path | None,
    results_dir: Path | None,
) -> list[Path]:
    if result_path is not None:
        candidate = result_path / "symptom_scores.json" if result_path.is_dir() else result_path
        if not candidate.exists():
            raise FileNotFoundError(f"Result not found: {candidate}")
        return [candidate]
    if results_dir is None or not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    return sorted(results_dir.rglob("symptom_scores.json"))


def resolve_session_dir(
    result: dict[str, Any],
    result_path: Path,
    sessions_dir: Path,
) -> Path | None:
    session_id = str(result.get("session_id") or result_path.parent.name)
    candidates = [sessions_dir / session_id]
    source_path = result.get("source_path")
    if source_path:
        source = Path(str(source_path))
        if not source.is_absolute():
            source = Path.cwd() / source
        candidates.append(source if source.is_dir() else source.parent)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "metadata.json").exists():
            return candidate.resolve()
    return None


def _model_settings(result: dict[str, Any]) -> tuple[str, int]:
    model = result.get("model")
    model = model if isinstance(model, dict) else {}
    return (
        str(model.get("madrs_model_id") or "webesama/MADRS-BERT"),
        int(model.get("max_length") or 512),
    )


def collect_work(
    paths: list[Path],
    sessions_dir: Path,
) -> tuple[list[SessionWork], list[dict[str, str]]]:
    work: list[SessionWork] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            result = _read_json(path)
            session_dir = resolve_session_dir(result, path, sessions_dir)
            if session_dir is None:
                raise FileNotFoundError(
                    f"Source session directory unavailable for {result.get('session_id', path.parent.name)}"
                )
            model_id, max_length = _model_settings(result)
            work.append(
                SessionWork(
                    result_path=path.resolve(),
                    session_dir=session_dir,
                    result=result,
                    model_id=model_id,
                    max_length=max_length,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "result_path": str(path),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return work, errors


def _render_outputs(report: dict[str, Any], session_dir: Path) -> tuple[Path, Path]:
    madrs_path = render_explanation_report(
        report,
        session_dir / MADRS_PNG,
        phq_only=False,
    )
    phq_path = render_explanation_report(
        report,
        session_dir / PHQ_PNG,
        phq_only=True,
    )
    return madrs_path, phq_path


def _write_errors(errors: list[dict[str, str]], path: Path) -> None:
    if not errors:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--result-path",
        type=Path,
        help="One symptom_scores.json file or its containing result directory.",
    )
    source.add_argument(
        "--results-dir",
        type=Path,
        help="Root recursively containing per-session symptom_scores.json files.",
    )
    parser.add_argument("--sessions-dir", type=Path, default=Path("data/sessions"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument(
        "--ig-step-batch-size",
        type=int,
        default=None,
        help="Number of interpolation steps evaluated together (default: 4 CUDA, 1 CPU).",
    )
    parser.add_argument("--score-tolerance", type=float, default=1e-3)
    parser.add_argument("--force", action="store_true", help="Recompute matching cached attributions.")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ig_steps < 2:
        raise SystemExit("--ig-steps must be at least 2")
    if args.ig_step_batch_size is not None and args.ig_step_batch_size < 1:
        raise SystemExit("--ig-step-batch-size must be at least 1")
    if args.score_tolerance < 0:
        raise SystemExit("--score-tolerance cannot be negative")

    try:
        paths = discover_result_paths(
            result_path=args.result_path,
            results_dir=args.results_dir,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    if not paths:
        raise SystemExit("No symptom_scores.json files found")

    work, errors = collect_work(paths, args.sessions_dir)
    grouped: dict[tuple[str, int], list[SessionWork]] = defaultdict(list)
    for item in work:
        grouped[(item.model_id, item.max_length)].append(item)

    completed = 0
    cached = 0
    failed = 0
    started = time.perf_counter()
    for (model_id, max_length), group in grouped.items():
        config = explanation_config(
            model_id=model_id,
            max_length=max_length,
            steps=args.ig_steps,
            score_tolerance=args.score_tolerance,
        )
        pending: list[SessionWork] = []
        for item in group:
            report_path = item.session_dir / ATTRIBUTION_JSON
            report = None if args.force else read_explanation_json(report_path)
            if report is not None and report_is_current(report, item.result, config):
                try:
                    madrs_path = item.session_dir / MADRS_PNG
                    phq_path = item.session_dir / PHQ_PNG
                    if not madrs_path.exists() or not phq_path.exists():
                        _render_outputs(report, item.session_dir)
                    cached += 1
                    completed += 1
                    print(f"[cached] {item.result.get('session_id')}")
                except Exception as exc:
                    errors.append(
                        {
                            "result_path": str(item.result_path),
                            "error_type": type(exc).__name__,
                            "message": f"Cached report rendering failed: {exc}",
                        }
                    )
                    failed += 1
                    if args.fail_fast:
                        raise
            else:
                pending.append(item)

        if not pending:
            continue
        print(
            f"Loading {model_id} once for {len(pending)} session(s) "
            f"on {args.device}..."
        )
        try:
            explainer = MadrsIntegratedGradientsExplainer(
                model_id,
                device=args.device,
                max_length=max_length,
                steps=args.ig_steps,
                step_batch_size=args.ig_step_batch_size,
                score_tolerance=args.score_tolerance,
            )
        except Exception as exc:
            for item in pending:
                errors.append(
                    {
                        "result_path": str(item.result_path),
                        "error_type": type(exc).__name__,
                        "message": f"Model loading failed: {exc}",
                    }
                )
                failed += 1
            if args.fail_fast:
                raise
            continue

        for index, item in enumerate(pending, start=1):
            session_id = str(item.result.get("session_id") or item.result_path.parent.name)
            print(f"[{index}/{len(pending)}] Explaining {session_id}")
            try:
                report = build_session_explanation_report(item.result, explainer)
                write_explanation_json(report, item.session_dir / ATTRIBUTION_JSON)
                _render_outputs(report, item.session_dir)
                completed += 1
            except Exception as exc:
                failed += 1
                errors.append(
                    {
                        "result_path": str(item.result_path),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                print(f"  ERROR: {exc}", file=sys.stderr)
                if args.fail_fast:
                    raise

        del explainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    error_root = (
        args.results_dir
        if args.results_dir is not None
        else (args.result_path if args.result_path.is_dir() else args.result_path.parent)
    )
    _write_errors(errors, error_root / "explanation_errors.jsonl")
    elapsed = time.perf_counter() - started
    print(
        f"Completed {completed}/{len(work)} sessions ({cached} cached, {failed} failed) "
        f"in {elapsed:.1f}s."
    )
    if errors:
        print(f"Recorded {len(errors)} error(s) under {error_root}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
