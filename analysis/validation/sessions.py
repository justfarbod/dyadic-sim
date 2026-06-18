"""
Shared session loading for the symptom analyses.

Centralises the logic that was duplicated across symptom_manifest.py,
symptom_embed.py, symptom_embed_robust.py: parsing the generated case name,
extracting patient text (dropping the verbatim-echo artifact), and iterating
session directories from an arbitrary glob.

`parse_case` handles ANY run prefix (e.g. "batch_", "pilotfix_"), not just the
hardcoded "batch_" the original scripts assumed.
"""
from __future__ import annotations
import json
import glob
import os
import re

CASES = ("only_love_can_save_me", "empty_and_invisible", "afraid_of_dogs")

# <prefix>_<case>_<symptom-or-no_symptoms>_run_<n>   ; prefix may contain underscores
_CASE_RE = re.compile(r"^(?P<prefix>.+?)_(?P<body>.+)_run_(?P<run>\d+)$")


def parse_case(case_name: str):
    """Return (prefix, case, symptom, run) from a generated case_name.

    symptom is "no_symptoms" for the baseline condition. Returns
    (None, None, None, None) if the name doesn't match the expected pattern.
    """
    m = _CASE_RE.match(case_name)
    if not m:
        return None, None, None, None
    prefix, body, run = m.group("prefix"), m.group("body"), int(m.group("run"))
    for c in CASES:
        # body looks like "<prefix-tail?>_<case>_<symptom>"; find the case marker
        idx = body.find(c)
        if idx != -1:
            tail = body[idx + len(c):].lstrip("_")
            return prefix, c, (tail or "no_symptoms"), run
    return prefix, None, None, run


def patient_text(session_dir: str) -> str:
    """Concatenate patient utterances, dropping verbatim echoes of the prior
    therapist turn (the echo data artifact)."""
    path = os.path.join(session_dir, "transcript.jsonl")
    tx = [json.loads(l) for l in open(path) if l.strip()]
    chunks = []
    for i, t in enumerate(tx):
        p = (t.get("patient_text") or "").strip()
        if i > 0 and p and p == (tx[i - 1].get("therapist_text") or "").strip():
            continue
        if p:
            chunks.append(p)
    return " ".join(chunks)


def therapist_text(session_dir: str) -> str:
    path = os.path.join(session_dir, "transcript.jsonl")
    tx = [json.loads(l) for l in open(path) if l.strip()]
    return " ".join((t.get("therapist_text") or "") for t in tx)


def iter_sessions(globs):
    """Yield (session_dir, metadata, prefix, case, symptom, run) for each session
    directory matched by `globs` (str or list of str) that has a metadata.json
    and a parseable case_name. Sessions whose name doesn't parse are skipped.
    """
    if isinstance(globs, str):
        globs = [globs]
    seen = set()
    for g in globs:
        for d in sorted(glob.glob(g)):
            if d in seen or not os.path.isdir(d):
                continue
            mpath = os.path.join(d, "metadata.json")
            if not os.path.exists(mpath):
                continue
            meta = json.load(open(mpath))
            prefix, case, symptom, run = parse_case(meta.get("case_name", ""))
            if case is None or symptom is None:
                continue
            seen.add(d)
            yield d, meta, prefix, case, symptom, run