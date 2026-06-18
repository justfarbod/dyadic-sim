"""Canonical input/output locations for the symptom-validation analyses.

Two buckets under ``data/`` (see also the project README / AUDIT_SYMPTOMS.md):

  * EXTERNAL (imported)  -- ``data/ext-session-logs/`` : session logs generated
    elsewhere (the PR-1 author's runs), shipped as zips + their ``_extracted/``
    form. Read-only inputs for the baseline analyses.
  * RESULTS (local)      -- ``data/results/`` : everything we compute here.
      - ``baseline_full/``  full 300-session baseline analysis outputs
      - ``baseline_v0/``    frozen pre-fix snapshot (provenance in BASELINE.md)
      - ``<prefix>/``       per-pilot outputs (pilotfix/, pfix20/, efix20/, ...)

Locally generated *raw* sessions live in ``data/sessions/`` (gitignored) and are
passed explicitly via ``--sessions`` to the post-fix pilots.
"""
from __future__ import annotations
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# external / imported raw sessions
EXTERNAL = os.path.join(REPO_ROOT, "data", "ext-session-logs", "_extracted")
EXTERNAL_SETS = {
    "llama3.1": os.path.join(EXTERNAL, "set1", "*"),
    "mistral-nemo": os.path.join(EXTERNAL, "set2", "sessions", "*"),
}

# local results
RESULTS = os.path.join(REPO_ROOT, "data", "results")
BASELINE_FULL = os.path.join(RESULTS, "baseline_full")
BASELINE_V0 = os.path.join(RESULTS, "baseline_v0")