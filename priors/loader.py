"""
priors/loader.py

Loads and validates YAML prior files from config/priors/.
Returns raw dicts. The dataclasses in therapist_prior.py and
patient_prior.py handle structure and prompt construction.
"""

from pathlib import Path
import yaml


PRIORS_DIR = Path("config/priors")


def load_therapist_base() -> dict:
    """Load the therapist base priors (always loaded)."""
    path = PRIORS_DIR / "therapist" / "base.yaml"
    return _load(path)


def load_therapist_variant(orientation: str) -> dict:
    """
    Load an orientation-specific variant, layered on top of base.

    Args:
        orientation: "psychodynamic", "person_centred", or "cbt"
    """
    path = PRIORS_DIR / "therapist" / "variants" / f"{orientation}.yaml"
    if not path.exists():
        available = [p.stem for p in (PRIORS_DIR / "therapist" / "variants").glob("*.yaml")]
        raise FileNotFoundError(
            f"Orientation '{orientation}' not found. "
            f"Available: {available}"
        )
    return _load(path)


def load_patient_case(case_name: str) -> dict:
    """
    Load a patient case file.

    Args:
        case_name: filename without .yaml, e.g. "only_love_can_save_me"
    """
    path = PRIORS_DIR / "patient" / "cases" / f"{case_name}.yaml"
    if not path.exists():
        available = [
            p.stem for p in (PRIORS_DIR / "patient" / "cases").glob("*.yaml")
            if not p.stem.startswith("_")
        ]
        raise FileNotFoundError(
            f"Patient case '{case_name}' not found. "
            f"Available: {available}"
        )
    return _load(path)


def list_cases() -> list[str]:
    """Return all available patient case names."""
    return [
        p.stem
        for p in (PRIORS_DIR / "patient" / "cases").glob("*.yaml")
        if not p.stem.startswith("_")
    ]


def list_orientations() -> list[str]:
    """Return all available therapist orientation variants."""
    return [
        p.stem
        for p in (PRIORS_DIR / "therapist" / "variants").glob("*.yaml")
    ]


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Prior file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Prior file must be a YAML mapping: {path}")
    return data
