"""Freeze a copy of result artifacts before the pipeline that produced them changes.

Run this BEFORE any pipeline rebuild. Rerunning an analysis overwrites its
outputs, and for these two corpora that would destroy the only record of what the
historical numbers were computed from — which is not recoverable, because the
configuration sidecars are already incomplete.

**What is already lost.** `symptom_embed.py` and `naive_prevalence.py` both wrote
`analysis_config.json` into the same output directory, so whichever ran second
replaced the first. The damage is symmetric and complete:

    <corpus A>/analysis_config.json   describes symptom_embed.py
                                      -> the lexicon config is GONE
    <corpus B>/analysis_config.json   describes naive_prevalence.py
                                      -> the embedding config is GONE

Whichever sidecar survived is a coin toss, and the one that did not survive may
be the one describing a headline result. Those CLI arguments are recoverable
from this conversation's record but not from any artifact, so they are recorded
below as `reconstructed`, never as `known`.

Both sidecars also record `inputs: {n_sessions: N}` and nothing else — a count,
not an identity. Which sessions produced a historical number cannot be
established from the artifact alone. That is what `provenance.py` is being
changed to fix; this module preserves the evidence of the gap.

`data/` is gitignored, so the archived files themselves stay untracked. The
manifest is tracked, and it carries a SHA-256 for every archived file, so the
archive can be verified later even though it is not in version control.

    PYTHONPATH=. python analysis/validation/archive.py --label pre_pipeline_rebuild
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime

ARCHIVE_ROOT = os.path.join("data", "_archive")

# What each script is expected to leave behind, so a missing artifact is
# reported rather than silently absent from the manifest.
EXPECTED = {
    "symptom_embed.py": ("symptom_embed_long.csv",),
    "naive_prevalence.py": ("naive_prevalence_long.csv", "naive_prevalence_evidence.csv"),
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return f"sha256:{h.hexdigest()}"


def describe_config(result_dir: str) -> dict:
    """Which script's configuration survived here, and which was overwritten."""
    path = os.path.join(result_dir, "analysis_config.json")
    if not os.path.exists(path):
        return {"present": False, "describes": None,
                "lost": sorted(EXPECTED), "note": "no sidecar at all"}
    with open(path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    describes = cfg.get("script")
    lost = [s for s in EXPECTED if s != describes]
    return {
        "present": True,
        "describes": describes,
        "lost": lost,
        "written_at": cfg.get("written_at"),
        "git_commit": cfg.get("git_commit"),
        "records_input_identity": bool(
            set(cfg.get("inputs", {})) - {"n_sessions"}
        ),
        "note": (
            f"single shared filename; {', '.join(lost)} configuration was "
            f"overwritten and is not recoverable from artifacts"
        ) if lost else None,
    }


def archive(names: list[str], label: str, results_root: str) -> dict:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    dest_root = os.path.join(ARCHIVE_ROOT, f"{label}_{stamp}")
    manifest = {
        "label": label,
        "archived_at": datetime.now(UTC).isoformat(),
        "reason": (
            "Frozen before rebuilding the analysis pipeline. Reruns overwrite "
            "these files; the configuration sidecars are already incomplete, so "
            "the historical numbers could not otherwise be traced."
        ),
        "corpora": {},
    }
    for name in names:
        src = os.path.join(results_root, name)
        if not os.path.isdir(src):
            print(f"  skip {name}: not found")
            continue
        dest = os.path.join(dest_root, name)
        os.makedirs(dest, exist_ok=True)
        files = {}
        for entry in sorted(os.listdir(src)):
            spath = os.path.join(src, entry)
            if not os.path.isfile(spath):
                continue
            shutil.copy2(spath, os.path.join(dest, entry))
            files[entry] = {"sha256": sha256(spath), "bytes": os.path.getsize(spath)}
        cfg = describe_config(src)
        manifest["corpora"][name] = {
            "archived_to": dest,
            "n_files": len(files),
            "configuration": cfg,
            "provenance_status": {
                # The distinction the plan requires: what we actually know,
                # versus what we are asserting from memory.
                "known": ([cfg["describes"]] if cfg["present"] else []),
                "reconstructed": cfg["lost"] if cfg["present"] else [],
                "irretrievable": (
                    ["exact input session identities"]
                    if not cfg.get("records_input_identity") else []
                ),
            },
            "files": files,
        }
        print(f"  archived {name}: {len(files)} files -> {dest}")
        if cfg["lost"]:
            print(f"    ! configuration for {', '.join(cfg['lost'])} was overwritten")
        if not cfg.get("records_input_identity"):
            print("    ! sidecar records only a session COUNT, not which sessions")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--names", nargs="+", default=["power01"],
                        help="Result directories under --results-root to freeze.")
    parser.add_argument("--label", default="pre_pipeline_rebuild")
    parser.add_argument("--results-root", default=os.path.join("data", "results"))
    args = parser.parse_args()

    os.makedirs(ARCHIVE_ROOT, exist_ok=True)
    manifest = archive(args.names, args.label, args.results_root)
    path = os.path.join(ARCHIVE_ROOT, "MANIFEST.json")
    existing = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if not isinstance(existing, list):
            existing = [existing]
    existing.append(manifest)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nwrote {path} ({len(existing)} archive entr"
          f"{'y' if len(existing) == 1 else 'ies'})")
    print("Rerunning the analyses does NOT recover the lost configurations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
