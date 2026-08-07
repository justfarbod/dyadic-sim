from __future__ import annotations

import unittest
from pathlib import Path

from symptom_scoring.config import KNOWN_OPENING_PROMPT
from symptom_scoring.transcript_parser import (
    build_turn_pairs,
    load_prior_levels,
)
from symptom_scoring.types import SessionSource


class TranscriptParserTests(unittest.TestCase):
    def test_pairs_patient_with_previous_rows_therapist(self):
        source = SessionSource(
            session_dir=Path("session"),
            metadata={"initial_patient_prompt": "Opening"},
            transcript=[
                {"turn": 1, "patient_text": "First patient", "therapist_text": "Question one"},
                {"turn": 2, "patient_text": "Second patient", "therapist_text": "Question two"},
                {"turn": 3, "patient_text": "Yes.", "therapist_text": "Question three"},
            ],
        )

        pairs = build_turn_pairs(source)

        self.assertEqual(pairs[0].therapist_text, "Opening")
        self.assertEqual(pairs[1].therapist_text, "Question one")
        self.assertEqual(pairs[2].therapist_text, "Question two")
        self.assertEqual(pairs[2].earlier_therapist_text, "Question one")
        self.assertEqual(pairs[2].earlier_patient_text, "Second patient")

    def test_legacy_opening_prompt_is_labeled_synthesized(self):
        source = SessionSource(
            session_dir=Path("session"),
            metadata={},
            transcript=[{"turn": 1, "patient_text": "Hello", "therapist_text": "Welcome"}],
        )
        pair = build_turn_pairs(source)[0]
        self.assertEqual(pair.therapist_text, KNOWN_OPENING_PROMPT)
        self.assertTrue(pair.context_synthesized)
        self.assertEqual(pair.context_source, "legacy_known_opening_prompt")

    def test_echo_artifact_is_excluded(self):
        source = SessionSource(
            session_dir=Path("session"),
            metadata={"initial_patient_prompt": "Opening"},
            transcript=[
                {"turn": 1, "patient_text": "First", "therapist_text": "Please continue"},
                {"turn": 2, "patient_text": "Please continue", "therapist_text": "Next"},
            ],
        )
        pair = build_turn_pairs(source)[1]
        self.assertFalse(pair.valid)
        self.assertEqual(pair.exclusion_reason, "verbatim_echo_artifact")

    def test_prior_levels_prefer_raw_metadata(self):
        levels, source = load_prior_levels(
            {
                "patient_symptom_levels": {
                    "sleep_problems": "nearly every day",
                }
            }
        )
        self.assertEqual(levels["sleep_problems"], "nearly every day")
        self.assertEqual(source, "metadata_raw")

    def test_prior_levels_recover_whitespace_prefixed_legacy_frequencies(self):
        levels, source = load_prior_levels(
            {
                "patient_symptoms": (
                    "- Sleep problems: nearly every day — trouble staying asleep\n"
                    "- Trouble concentrating: not at all — no difficulty"
                )
            }
        )

        self.assertEqual(
            levels,
            {
                "sleep_problems": "nearly every day",
                "concentration_problems": "not at all",
            },
        )
        self.assertEqual(source, "metadata_legacy_formatted")


if __name__ == "__main__":
    unittest.main()
