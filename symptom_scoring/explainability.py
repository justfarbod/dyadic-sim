"""Native PyTorch Integrated Gradients explanations for MADRS-BERT."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime
from typing import Any

import torch

from symptom_scoring.config import MADRS_TOPICS, PHQ_TO_MADRS
from symptom_scoring.translator import resolve_device


EXPLANATION_SCHEMA_VERSION = "1.0"
EXPLANATION_METHOD = "integrated_gradients_input_embeddings"
BASELINE_METHOD = "preserve_special_tokens_pad_content"

_TOPIC_ORDER = {topic: index for index, topic in enumerate(MADRS_TOPICS)}
_TOPIC_TO_PHQ = {
    topic: symptom for symptom, topic in PHQ_TO_MADRS.items() if topic is not None
}
_STRUCTURAL_WORDS = {"topic:", "dialogue:", "interviewer:", "patient:"}


def input_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def accepted_turn_topics(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return accepted MADRS-BERT jobs in deterministic conversation/topic order."""

    rows = [
        row
        for row in result.get("turn_results", [])
        if isinstance(row, dict)
        and row.get("accepted_for_scoring")
        and row.get("model_input")
        and row.get("raw_model_output") is not None
    ]
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("turn_index", 0)),
            _TOPIC_ORDER.get(str(row.get("madrs_topic")), len(_TOPIC_ORDER)),
        ),
    )


def source_fingerprint(result: dict[str, Any]) -> str:
    """Fingerprint every accepted input and stored output for cache validation."""

    digest = hashlib.sha256()
    digest.update(str(result.get("session_id") or "").encode("utf-8"))
    for row in accepted_turn_topics(result):
        parts = (
            str(row.get("turn_index")),
            str(row.get("madrs_topic")),
            str(row.get("symptom")),
            input_sha256(str(row["model_input"])),
            repr(float(row["raw_model_output"])),
            str(bool(row.get("input_compacted"))),
        )
        digest.update("\x1f".join(parts).encode("utf-8"))
    return digest.hexdigest()


def explanation_config(
    *,
    model_id: str,
    max_length: int,
    steps: int,
    score_tolerance: float,
) -> dict[str, Any]:
    return {
        "method": EXPLANATION_METHOD,
        "baseline": BASELINE_METHOD,
        "model_id": model_id,
        "max_length": int(max_length),
        "steps": int(steps),
        "score_tolerance": float(score_tolerance),
        "explains_output": "unclipped_regression_logit",
        "interpretation": (
            "Positive attribution raises and negative attribution lowers the raw model "
            "output relative to the padding baseline; this is local model sensitivity, "
            "not proof of linguistic causality."
        ),
    }


def report_is_current(
    report: dict[str, Any],
    result: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    return bool(
        report.get("schema_version") == EXPLANATION_SCHEMA_VERSION
        and report.get("session_id") == result.get("session_id")
        and report.get("source_fingerprint") == source_fingerprint(result)
        and report.get("explainer") == config
        and isinstance(report.get("items"), list)
    )


def _scalar_logit(outputs: Any) -> torch.Tensor:
    logits = outputs.logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    if logits.ndim == 1:
        return logits
    raise ValueError(
        f"Expected one regression logit per input; received shape {tuple(logits.shape)}"
    )


def _segment_for_span(text: str, start: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    if line.startswith("Topic:"):
        return "topic"
    if line.startswith("Dialogue:"):
        return "structure"
    if line.startswith("Interviewer:"):
        return "interviewer"
    if line.startswith("Patient:"):
        return "patient"
    return "other"


def aggregate_word_attributions(
    text: str,
    offsets: list[tuple[int, int]],
    token_attributions: list[float],
) -> list[dict[str, Any]]:
    """Aggregate token attributions into exact non-whitespace input spans."""

    spans = [match.span() for match in re.finditer(r"\S+", text)]
    values = [0.0 for _ in spans]
    for (token_start, token_end), attribution in zip(offsets, token_attributions):
        if token_end <= token_start:
            continue
        best_index = None
        best_overlap = 0
        for index, (word_start, word_end) in enumerate(spans):
            if word_start >= token_end:
                break
            overlap = max(0, min(token_end, word_end) - max(token_start, word_start))
            if overlap > best_overlap:
                best_index = index
                best_overlap = overlap
        if best_index is not None:
            values[best_index] += float(attribution)

    max_abs = max((abs(value) for value in values), default=0.0)
    words: list[dict[str, Any]] = []
    for (start, end), value in zip(spans, values):
        normalized = value / max_abs if max_abs else 0.0
        word = text[start:end]
        words.append(
            {
                "text": word,
                "start": start,
                "end": end,
                "attribution": value,
                "normalized_attribution": normalized,
                "direction": "raises" if value > 0 else "lowers" if value < 0 else "neutral",
                "segment": _segment_for_span(text, start),
                "structural": word.casefold() in _STRUCTURAL_WORDS,
            }
        )
    return words


class MadrsIntegratedGradientsExplainer:
    """Explain the actual MADRS-BERT regressor through its input embeddings."""

    def __init__(
        self,
        model_id: str = "webesama/MADRS-BERT",
        *,
        device: str = "auto",
        max_length: int = 512,
        steps: int = 32,
        step_batch_size: int | None = None,
        score_tolerance: float = 1e-3,
        tokenizer=None,
        model=None,
    ):
        if steps < 2:
            raise ValueError("Integrated Gradients requires at least two steps")
        if step_batch_size is not None and step_batch_size < 1:
            raise ValueError("step_batch_size must be at least one")
        if score_tolerance < 0:
            raise ValueError("score_tolerance cannot be negative")

        self.model_id = model_id
        self.device = resolve_device(device)
        self.max_length = int(max_length)
        self.steps = int(steps)
        self.step_batch_size = step_batch_size or (4 if self.device.type == "cuda" else 1)
        self.score_tolerance = float(score_tolerance)

        if tokenizer is None or model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer with character offsets is required")

        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()
        self.embedding_layer = self.model.get_input_embeddings()

    @property
    def config(self) -> dict[str, Any]:
        return explanation_config(
            model_id=self.model_id,
            max_length=self.max_length,
            steps=self.steps,
            score_tolerance=self.score_tolerance,
        )

    def _encode(self, text: str) -> dict[str, Any]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        offsets_tensor = encoded.pop("offset_mapping")
        special_tensor = encoded.pop("special_tokens_mask")
        offsets = [tuple(map(int, item)) for item in offsets_tensor[0].tolist()]
        special_mask = special_tensor[0].bool()
        tensors = {key: value.to(self.device) for key, value in encoded.items()}
        return {
            "tensors": tensors,
            "offsets": offsets,
            "special_mask": special_mask.to(self.device),
        }

    def explain(self, text: str, *, expected_raw_output: float) -> dict[str, Any]:
        encoded = self._encode(text)
        tensors = encoded["tensors"]
        input_ids = tensors["input_ids"]
        offsets: list[tuple[int, int]] = encoded["offsets"]
        special_mask: torch.Tensor = encoded["special_mask"]
        consumed_end = max((end for start, end in offsets if end > start), default=0)
        effective_input = text[:consumed_end]
        tokenizer_truncated = consumed_end < len(text.rstrip())

        baseline_ids = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        baseline_ids[0, ~special_mask] = int(pad_id)

        passthrough = {key: value for key, value in tensors.items() if key != "input_ids"}
        with torch.inference_mode():
            reproduced = float(_scalar_logit(self.model(**tensors))[0].detach().cpu())
            baseline_embeddings = self.embedding_layer(baseline_ids)
            baseline_output = float(
                _scalar_logit(
                    self.model(inputs_embeds=baseline_embeddings, **passthrough)
                )[0].detach().cpu()
            )

        difference = abs(reproduced - float(expected_raw_output))
        base = {
            "status": "ok" if difference <= self.score_tolerance else "score_mismatch",
            "model_input": text,
            "effective_model_input": effective_input,
            "input_sha256": input_sha256(text),
            "tokenizer_truncated": tokenizer_truncated,
            "omitted_suffix": text[consumed_end:] if tokenizer_truncated else "",
            "expected_raw_output": float(expected_raw_output),
            "reproduced_raw_output": reproduced,
            "score_difference": difference,
            "baseline_raw_output": baseline_output,
            "attribution_sum": None,
            "output_delta_from_baseline": reproduced - baseline_output,
            "completeness_error": None,
            "words": [],
        }
        if difference > self.score_tolerance:
            return base

        with torch.no_grad():
            input_embeddings = self.embedding_layer(input_ids).detach()
            baseline_embeddings = baseline_embeddings.detach()
            embedding_delta = input_embeddings - baseline_embeddings

        gradient_sum = torch.zeros_like(input_embeddings)
        alphas = (torch.arange(self.steps, device=self.device, dtype=input_embeddings.dtype) + 0.5) / self.steps
        for start in range(0, self.steps, self.step_batch_size):
            alpha_batch = alphas[start : start + self.step_batch_size]
            interpolated = (
                baseline_embeddings
                + alpha_batch[:, None, None] * embedding_delta
            ).detach().requires_grad_(True)
            repeated = {
                key: value.expand(len(alpha_batch), *value.shape[1:])
                for key, value in passthrough.items()
            }
            outputs = _scalar_logit(self.model(inputs_embeds=interpolated, **repeated))
            gradients = torch.autograd.grad(outputs.sum(), interpolated)[0]
            gradient_sum += gradients.sum(dim=0, keepdim=True)

        average_gradient = gradient_sum / self.steps
        token_attributions = (embedding_delta * average_gradient).sum(dim=-1)[0]
        token_attributions = token_attributions.masked_fill(special_mask, 0.0)
        attribution_values = token_attributions.detach().cpu().tolist()
        attribution_sum = float(token_attributions.sum().detach().cpu())
        output_delta = reproduced - baseline_output
        completeness_error = attribution_sum - output_delta
        if not math.isfinite(completeness_error):
            raise ValueError("Non-finite Integrated Gradients completeness error")

        base.update(
            {
                "attribution_sum": attribution_sum,
                "completeness_error": completeness_error,
                "words": aggregate_word_attributions(
                    effective_input,
                    offsets,
                    attribution_values,
                ),
            }
        )
        return base


def build_session_explanation_report(
    result: dict[str, Any],
    explainer: MadrsIntegratedGradientsExplainer,
) -> dict[str, Any]:
    """Explain every accepted MADRS topic in one saved session result."""

    items: list[dict[str, Any]] = []
    for row in accepted_turn_topics(result):
        explanation = explainer.explain(
            str(row["model_input"]),
            expected_raw_output=float(row["raw_model_output"]),
        )
        items.append(
            {
                "turn_index": int(row["turn_index"]),
                "madrs_topic": row.get("madrs_topic"),
                "symptom": row.get("symptom") or _TOPIC_TO_PHQ.get(row.get("madrs_topic")),
                "raw_model_output": float(row["raw_model_output"]),
                "raw_score": float(row["raw_score"]),
                "rounded_score": int(row["rounded_score"]),
                "input_compacted": bool(row.get("input_compacted")),
                "evidence": row.get("evidence"),
                **explanation,
            }
        )

    return {
        "schema_version": EXPLANATION_SCHEMA_VERSION,
        "session_id": result.get("session_id"),
        "source_path": result.get("source_path"),
        "created_at": datetime.now(UTC).isoformat(),
        "source_fingerprint": source_fingerprint(result),
        "explainer": explainer.config,
        "accepted_madrs_items": len(items),
        "accepted_phq_mapped_items": sum(item.get("symptom") is not None for item in items),
        "items": items,
        "warnings": [
            "Integrated Gradients describes local model sensitivity, not proven linguistic causality.",
            "Attributions explain the unclipped regression logit; displayed MADRS proxy scores are clipped to 0-6.",
        ],
    }
