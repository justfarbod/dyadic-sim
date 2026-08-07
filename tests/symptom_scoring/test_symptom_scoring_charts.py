from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.validation.plot_symptom_scoring import (
    PHQ_SYMPTOMS,
    build_conversation_score_table,
    build_detection_long,
    build_summary,
    extract_base_case,
    write_session_conversation_outputs,
    write_outputs,
)
from symptom_scoring.config import PHQ_SYMPTOM_LABELS, PHQ_TO_MADRS


class SymptomScoringChartTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.results_dir = self.root / "results"
        self.sessions_dir = self.root / "sessions"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _all_levels(self, **overrides: str) -> dict[str, str]:
        levels = {symptom: "not at all" for symptom in PHQ_SYMPTOMS}
        levels.update(overrides)
        return levels

    def _legacy_text(self, levels: dict[str, str]) -> str:
        return "\n".join(
            f"- {PHQ_SYMPTOM_LABELS[symptom]}: {levels[symptom]} — fixture"
            for symptom in PHQ_SYMPTOMS
        )

    def _add_fixture(
        self,
        session_id: str,
        *,
        metadata: dict[str, object] | None,
        detected: set[str] | None = None,
    ) -> None:
        session_dir = self.sessions_dir / session_id
        result_dir = self.results_dir / "nested_experiment" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        if metadata is not None:
            metadata = {"session_id": session_id, **metadata}
            (session_dir / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

        detected = detected or set()
        symptoms = {}
        for index, symptom in enumerate(PHQ_SYMPTOMS, start=1):
            is_detected = symptom in detected and PHQ_TO_MADRS[symptom] is not None
            symptoms[symptom] = {
                "raw_session_score": float(index) if is_detected else None,
                "rounded_session_score": index if is_detected else None,
                "source_turn": index if is_detected else None,
                "number_of_relevant_turns": 1 if is_detected else 0,
            }
        turn_results = []
        for turn_index in (1, 2):
            for index, symptom in enumerate(PHQ_SYMPTOMS, start=1):
                if PHQ_TO_MADRS[symptom] is None:
                    continue
                is_detected = symptom in detected and turn_index == 1
                turn_results.append(
                    {
                        "turn_index": turn_index,
                        "symptom": symptom,
                        "accepted_for_scoring": is_detected,
                        "raw_score": float(index) if is_detected else None,
                        "therapist_text": f"Therapist turn {turn_index}",
                        "patient_text": f"Patient turn {turn_index}",
                    }
                )
        result = {
            "session_id": session_id,
            "source_path": str(session_dir),
            "symptoms": symptoms,
            "turn_results": turn_results,
        }
        (result_dir / "symptom_scores.json").write_text(
            json.dumps(result), encoding="utf-8"
        )

    def _build_fixture_dataset(self):
        self._add_fixture(
            "raw_depressed",
            metadata={
                "patient_model": "model-a",
                "therapist_model": "therapist-x",
                "case_name": "case_depressed_mood_run_1",
                "patient_symptom_levels": self._all_levels(
                    depressed_mood="nearly every day"
                ),
            },
            detected={"depressed_mood", "low_energy"},
        )
        legacy_levels = self._all_levels(sleep_problems="several days")
        self._add_fixture(
            "legacy_sleep",
            metadata={
                "patient_model": "model-a",
                "therapist_model": "therapist-x",
                "case_name": "case_sleep_problems_run_1",
                "patient_symptoms": self._legacy_text(legacy_levels),
            },
            detected={"sleep_problems"},
        )
        self._add_fixture(
            "control",
            metadata={
                "patient_model": "model-b",
                "therapist_model": "therapist-y",
                "case_name": "case_no_symptoms_run_1",
                "patient_symptoms": "",
            },
            detected={"depressed_mood"},
        )

        self._add_fixture(
            "multiple",
            metadata={
                "patient_model": "model-b",
                "therapist_model": "therapist-y",
                "case_name": "case_multiple_run_1",
                "patient_symptom_levels": self._all_levels(
                    lack_of_pleasure="several days",
                    appetite_changes="more than half the days",
                ),
            },
            detected={"lack_of_pleasure"},
        )
        self._add_fixture(
            "psychomotor",
            metadata={
                "patient_model": "model-a",
                "therapist_model": "therapist-x",
                "case_name": "case_psychomotor_changes_run_1",
                "patient_symptom_levels": self._all_levels(
                    psychomotor_changes="nearly every day"
                ),
            },
        )
        self._add_fixture(
            "missing_metadata",
            metadata=None,
            detected={"depressed_mood"},
        )

    def test_extracts_base_case_from_batch_case_name(self):
        self.assertEqual(
            extract_base_case("batch_afraid_of_dogs_sleep_problems_run_3"),
            "afraid_of_dogs",
        )

    def test_builds_one_conversation_row_with_symptom_score_columns(self):
        self._add_fixture(
            "turn_table",
            metadata={"case_name": "batch_afraid_of_dogs_sleep_problems_run_1"},
            detected={"depressed_mood", "sleep_problems"},
        )
        result_path = next(self.results_dir.rglob("symptom_scores.json"))
        result = json.loads(result_path.read_text(encoding="utf-8"))

        table = build_conversation_score_table(result)

        self.assertEqual(len(table), 2)
        self.assertEqual(table.iloc[0]["conversation"], "Conversation 1")
        self.assertEqual(table.iloc[0]["depressed_mood"], 2.0)
        self.assertEqual(table.iloc[0]["sleep_problems"], 3.0)
        self.assertTrue(pd.isna(table.iloc[1]["depressed_mood"]))
        self.assertEqual(table.iloc[0]["psychomotor_changes"], "N/A")

    def test_writes_conversation_csv_and_diagram_beside_session_result(self):
        self._add_fixture(
            "turn_output",
            metadata={"case_name": "batch_afraid_of_dogs_sleep_problems_run_1"},
            detected={"sleep_problems"},
        )

        created, warnings = write_session_conversation_outputs(
            self.results_dir,
            self.sessions_dir,
        )

        self.assertEqual(warnings, [])
        self.assertEqual(len(created), 2)
        for output_path in created:
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertEqual(output_path.parent, self.sessions_dir / "turn_output")

    def test_builds_auditable_rows_for_all_metadata_variants(self):
        self._build_fixture_dataset()
        long_df, warnings = build_detection_long(self.results_dir, self.sessions_dir)

        self.assertEqual(len(long_df), 6 * len(PHQ_SYMPTOMS))
        self.assertTrue(any("Metadata unavailable" in warning for warning in warnings))

        raw = long_df[
            (long_df["session_id"] == "raw_depressed")
            & (long_df["symptom"] == "depressed_mood")
        ].iloc[0]
        self.assertEqual(raw["injection_source"], "metadata_raw")
        self.assertEqual(raw["case"], "case")
        self.assertTrue(raw["injected"])
        self.assertTrue(raw["detected"])
        self.assertEqual(raw["source_turn"], 2)

        legacy = long_df[long_df["session_id"] == "legacy_sleep"]
        self.assertEqual(set(legacy["injection_source"]), {"metadata_legacy_formatted"})
        self.assertEqual(int(legacy["injection_known"].sum()), len(PHQ_SYMPTOMS))

        control = long_df[long_df["session_id"] == "control"]
        self.assertEqual(set(control["injection_source"]), {"case_name_no_symptoms"})
        self.assertEqual(int(control["injected"].sum()), 0)

        multiple = long_df[
            (long_df["session_id"] == "multiple") & (long_df["injected"] == True)  # noqa: E712
        ]
        self.assertEqual(
            set(multiple["symptom"]), {"lack_of_pleasure", "appetite_changes"}
        )

        missing = long_df[long_df["session_id"] == "missing_metadata"]
        self.assertEqual(int(missing["injection_known"].sum()), 0)

    def test_summary_uses_valid_denominators_and_groups_patient_models(self):
        self._build_fixture_dataset()
        long_df, _ = build_detection_long(self.results_dir, self.sessions_dir)
        summary = build_summary(long_df)

        control_depressed = summary[
            (summary["summary_type"] == "injected_vs_detected")
            & (summary["injected_condition"] == "no_symptoms")
            & (summary["detected_symptom"] == "depressed_mood")
        ].iloc[0]
        self.assertEqual(control_depressed["sessions"], 1)
        self.assertEqual(control_depressed["detection_rate"], 1.0)

        model_b_pleasure = summary[
            (summary["summary_type"] == "target_detection_by_patient_model")
            & (summary["patient_model"] == "model-b")
            & (summary["injected_condition"] == "lack_of_pleasure")
        ].iloc[0]
        self.assertEqual(model_b_pleasure["sessions"], 1)
        self.assertEqual(model_b_pleasure["detection_rate"], 1.0)

        model_b_appetite = summary[
            (summary["summary_type"] == "target_detection_by_patient_model")
            & (summary["patient_model"] == "model-b")
            & (summary["injected_condition"] == "appetite_changes")
        ].iloc[0]
        self.assertEqual(model_b_appetite["sessions"], 1)
        self.assertEqual(model_b_appetite["detection_rate"], 0.0)

        psychomotor = summary[
            (summary["summary_type"] == "target_detection_by_patient_model")
            & (summary["patient_model"] == "model-a")
            & (summary["injected_condition"] == "psychomotor_changes")
        ].iloc[0]
        self.assertEqual(psychomotor["sessions"], 1)
        self.assertFalse(psychomotor["detection_mappable"])
        self.assertTrue(pd.isna(psychomotor["detection_rate"]))

        by_case = summary[
            (summary["summary_type"] == "target_detection_by_case")
            & (summary["case"] == "case")
            & (summary["injected_condition"] == "depressed_mood")
        ].iloc[0]
        self.assertEqual(by_case["sessions"], 1)
        self.assertEqual(by_case["detection_rate"], 1.0)

    def test_writes_nonempty_csv_and_png_outputs(self):
        self._build_fixture_dataset()
        long_df, _ = build_detection_long(self.results_dir, self.sessions_dir)
        summary = build_summary(long_df)
        output_paths = write_outputs(long_df, summary, self.root / "charts")

        for output_path in output_paths.values():
            with self.subTest(output_path=output_path):
                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
