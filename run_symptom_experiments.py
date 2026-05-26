from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


CASES_DIR = Path("config/priors/patient/cases")
BASE_CASE_NAME = "empty_and_invisible"

SYMPTOM_KEYS = [
    "lack_of_pleasure",
    "depressed_mood",
    "sleep_problems",
    "low_energy",
    "appetite_changes",
    "feelings_of_failure_or_guilt",
    "concentration_problems",
    "psychomotor_changes",
    "thoughts_of_death_or_self_harm",
]


def load_case(case_name: str) -> dict:
    path = CASES_DIR / f"{case_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Case not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Case file must contain a YAML mapping: {path}")

    return data


def write_case(case_name: str, data: dict) -> Path:
    path = CASES_DIR / f"{case_name}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return path


def build_no_symptoms_case(base_case: dict) -> dict:
    case_data = copy.deepcopy(base_case)
    case_data.pop("symptoms", None)
    return case_data


def build_single_symptom_case(base_case: dict, active_symptom: str) -> dict:
    case_data = copy.deepcopy(base_case)
    case_data["symptoms"] = {
        key: ("nearly every day" if key == active_symptom else "not at all")
        for key in SYMPTOM_KEYS
    }
    return case_data


def run_session(
    therapist: str,
    patient: str,
    case_name: str,
    orientation: str,
    turns: int,
    max_tokens: int,
    no_compress: bool,
) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        "run.py",
        "--therapist",
        therapist,
        "--patient",
        patient,
        "--case",
        case_name,
        "--orientation",
        orientation,
        "--turns",
        str(turns),
        "--max-tokens",
        str(max_tokens),
    ]

    if no_compress:
        cmd.append("--no-compress")

    print(f"\nRunning case: {case_name}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run symptom-isolated dyadic simulations for afraid_of_dogs."
    )
    parser.add_argument("--therapist", required=True, help="Therapist model name")
    parser.add_argument("--patient", required=True, help="Patient model name")
    parser.add_argument("--orientation", default="cbt", help="Therapist orientation")
    parser.add_argument("--turns", type=int, default=10, help="Turns per run")
    parser.add_argument("--max-tokens", type=int, default=300, help="Max tokens per response")
    parser.add_argument("--repeats", type=int, default=5, help="How many runs per condition")
    parser.add_argument(
        "--prefix",
        default="batch",
        help="Prefix for generated temporary case names",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Pass through to run.py",
    )
    parser.add_argument(
        "--keep-generated-cases",
        action="store_true",
        help="Keep generated YAML files instead of deleting them afterward",
    )

    args = parser.parse_args()

    base_case = load_case(BASE_CASE_NAME)
    generated_case_paths: list[Path] = []

    try:
        for i in range(1, args.repeats + 1):
            case_name = f"{args.prefix}_{BASE_CASE_NAME}_no_symptoms_run_{i}"
            case_data = build_no_symptoms_case(base_case)
            generated_case_paths.append(write_case(case_name, case_data))

            run_session(
                therapist=args.therapist,
                patient=args.patient,
                case_name=case_name,
                orientation=args.orientation,
                turns=args.turns,
                max_tokens=args.max_tokens,
                no_compress=args.no_compress,
            )

        for symptom_key in SYMPTOM_KEYS:
            for i in range(1, args.repeats + 1):
                case_name = f"{args.prefix}_{BASE_CASE_NAME}_{symptom_key}_run_{i}"
                case_data = build_single_symptom_case(base_case, symptom_key)
                generated_case_paths.append(write_case(case_name, case_data))

                run_session(
                    therapist=args.therapist,
                    patient=args.patient,
                    case_name=case_name,
                    orientation=args.orientation,
                    turns=args.turns,
                    max_tokens=args.max_tokens,
                    no_compress=args.no_compress,
                )

        print("\nAll runs completed successfully.")
        print("Sessions were saved automatically under data/sessions/.")

    finally:
        if not args.keep_generated_cases:
            for path in generated_case_paths:
                if path.exists():
                    path.unlink()
            print("Temporary generated case files deleted.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nA run failed with exit code {e.returncode}.", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)