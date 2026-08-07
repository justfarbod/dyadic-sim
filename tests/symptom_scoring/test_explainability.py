from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from explain_session_symptoms import collect_work, resolve_session_dir
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


class TinyTokenizer:
    is_fast = True
    pad_token_id = 0

    def __init__(self):
        self.ids = {"alpha": 2, "beta": 3}

    def __call__(self, text, **kwargs):
        import re

        matches = list(re.finditer(r"\S+", text))
        max_length = kwargs.get("max_length", 512)
        content = matches[: max(0, max_length - 2)]
        input_ids = [1, *(self.ids.get(match.group(0), 0) for match in content), 4]
        offsets = [(0, 0), *(match.span() for match in content), (0, 0)]
        special = [1, *(0 for _ in content), 1]
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
            "offset_mapping": torch.tensor([offsets], dtype=torch.long),
            "special_tokens_mask": torch.tensor([special], dtype=torch.long),
        }


class TinyRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(5, 1)
        with torch.no_grad():
            self.embeddings.weight[:, 0] = torch.tensor([0.0, 0.0, 2.0, -1.0, 0.0])

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        values = inputs_embeds if inputs_embeds is not None else self.embeddings(input_ids)
        logits = values.sum(dim=(1, 2)).unsqueeze(1)
        return SimpleNamespace(logits=logits)


def fixture_result(session_id: str = "session_test"):
    return {
        "session_id": session_id,
        "source_path": f"data/sessions/{session_id}",
        "model": {"madrs_model_id": "tiny", "max_length": 16},
        "turn_results": [
            {
                "turn_index": 1,
                "madrs_topic": "Schlaf",
                "symptom": "sleep_problems",
                "accepted_for_scoring": True,
                "model_input": "alpha beta",
                "raw_model_output": 1.0,
                "raw_score": 1.0,
                "rounded_score": 1,
                "input_compacted": True,
                "evidence": "alpha",
            },
            {
                "turn_index": 1,
                "madrs_topic": "Anspannung",
                "symptom": None,
                "accepted_for_scoring": True,
                "model_input": "alpha beta",
                "raw_model_output": 1.0,
                "raw_score": 1.0,
                "rounded_score": 1,
                "input_compacted": False,
                "evidence": "beta",
            },
            {
                "turn_index": 2,
                "madrs_topic": "Schlaf",
                "symptom": "sleep_problems",
                "accepted_for_scoring": False,
                "model_input": None,
                "raw_model_output": None,
            },
        ],
    }


class ExplainabilityTests(unittest.TestCase):
    def setUp(self):
        self.explainer = MadrsIntegratedGradientsExplainer(
            "tiny",
            device="cpu",
            max_length=16,
            steps=8,
            step_batch_size=2,
            tokenizer=TinyTokenizer(),
            model=TinyRegressor(),
        )

    def test_integrated_gradients_has_expected_signs_and_completeness(self):
        explanation = self.explainer.explain("alpha beta", expected_raw_output=1.0)

        self.assertEqual(explanation["status"], "ok")
        words = {word["text"]: word for word in explanation["words"]}
        self.assertGreater(words["alpha"]["attribution"], 0)
        self.assertLess(words["beta"]["attribution"], 0)
        self.assertAlmostEqual(explanation["attribution_sum"], 1.0, places=6)
        self.assertAlmostEqual(explanation["completeness_error"], 0.0, places=6)
        self.assertEqual(explanation["effective_model_input"], "alpha beta")

    def test_score_mismatch_is_recorded_without_attributions(self):
        explanation = self.explainer.explain("alpha beta", expected_raw_output=2.0)

        self.assertEqual(explanation["status"], "score_mismatch")
        self.assertEqual(explanation["words"], [])
        self.assertAlmostEqual(explanation["score_difference"], 1.0)

    def test_session_report_includes_all_madrs_and_phq_counts(self):
        result = fixture_result()
        report = build_session_explanation_report(result, self.explainer)

        self.assertEqual(report["accepted_madrs_items"], 2)
        self.assertEqual(report["accepted_phq_mapped_items"], 1)
        sleep = next(item for item in report["items"] if item["madrs_topic"] == "Schlaf")
        self.assertEqual(sleep["model_input"], "alpha beta")
        self.assertTrue(sleep["input_compacted"])

    def test_cache_validation_uses_config_and_source_fingerprint(self):
        result = fixture_result()
        report = build_session_explanation_report(result, self.explainer)

        self.assertTrue(report_is_current(report, result, self.explainer.config))
        changed = fixture_result()
        changed["turn_results"][0]["model_input"] = "beta alpha"
        self.assertFalse(report_is_current(report, changed, self.explainer.config))
        other_config = explanation_config(
            model_id="tiny", max_length=16, steps=32, score_tolerance=1e-3
        )
        self.assertFalse(report_is_current(report, result, other_config))

    def test_json_and_both_views_render_nonempty(self):
        report = build_session_explanation_report(fixture_result(), self.explainer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = write_explanation_json(report, root / "attributions.json")
            madrs_path = render_explanation_report(
                report, root / "madrs.png", phq_only=False
            )
            phq_path = render_explanation_report(
                report, root / "phq.png", phq_only=True
            )

            self.assertEqual(read_explanation_json(json_path)["accepted_madrs_items"], 2)
            for path in (json_path, madrs_path, phq_path):
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

    def test_empty_session_still_renders_both_views(self):
        result = fixture_result()
        result["turn_results"] = []
        report = build_session_explanation_report(result, self.explainer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for phq_only, name in ((False, "madrs.png"), (True, "phq.png")):
                path = render_explanation_report(report, root / name, phq_only=phq_only)
                self.assertGreater(path.stat().st_size, 0)

    def test_source_session_resolution_and_work_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = root / "sessions"
            session_dir = sessions / "session_test"
            session_dir.mkdir(parents=True)
            (session_dir / "metadata.json").write_text("{}", encoding="utf-8")
            result_dir = root / "results" / "session_test"
            result_dir.mkdir(parents=True)
            result_path = result_dir / "symptom_scores.json"
            result_path.write_text(json.dumps(fixture_result()), encoding="utf-8")

            loaded = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(resolve_session_dir(loaded, result_path, sessions), session_dir)
            work, errors = collect_work([result_path], sessions)
            self.assertEqual(errors, [])
            self.assertEqual(work[0].session_dir, session_dir.resolve())


if __name__ == "__main__":
    unittest.main()
