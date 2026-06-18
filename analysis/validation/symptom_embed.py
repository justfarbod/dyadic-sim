"""
Embedding-based manipulation check (robust to lexicon breadth).

For each session we embed the patient's utterances and compare them to short
reference descriptions of each of the 9 PHQ-9 symptoms. The key quantity is a
*within-session* z-score: how much more similar is the patient's speech to the
INJECTED symptom than to the other 8 symptoms, in that same session. Within-
session standardisation controls for case, verbosity, and overall "clinical
tone", isolating the effect of the manipulation itself.

  target_z > 0  -> patient leans toward the injected symptom (manipulation works)
  target_z ~ 0  -> injected symptom is not preferentially expressed

Usage:
  # baseline (default): the 300-session external archives
  PYTHONPATH=. python analysis/validation/symptom_embed.py

  # post-fix pilot:
  PYTHONPATH=. python analysis/validation/symptom_embed.py \
      --sessions 'data/sessions/*' --out-dir data/results/pilotfix

Outputs (in --out-dir):
  - symptom_embed_long.csv      per-session similarities + target_z
  - fig_symptom_embed.png       target_z by injected symptom x model
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.embeddings import get_model, EMBEDDINGS_AVAILABLE
from analysis.validation.paths import EXTERNAL_SETS, BASELINE_FULL
from analysis.validation.sessions import iter_sessions, patient_text

# Default = the two external baseline archives (reproduces the baseline_v0 run).
DEFAULT_SESSIONS = list(EXTERNAL_SETS.values())

# Reference descriptions: the lived-experience phrasing of each PHQ-9 item.
REFERENCES = {
    "lack_of_pleasure": "I have little interest or pleasure in doing things; nothing feels enjoyable anymore.",
    "depressed_mood": "I feel down, depressed, sad, and hopeless most of the time.",
    "sleep_problems": "I have trouble sleeping; I can't fall or stay asleep, or I sleep too much.",
    "low_energy": "I feel tired and have very little energy; I am exhausted and worn out.",
    "appetite_changes": "My appetite has changed; I have a poor appetite or I am overeating.",
    "feelings_of_failure_or_guilt": "I feel bad about myself, like a failure, worthless, and guilty.",
    "concentration_problems": "I have trouble concentrating and focusing; my mind won't stay on things.",
    "psychomotor_changes": "I move or speak slowly, or I feel so restless and fidgety I can't sit still.",
    "thoughts_of_death_or_self_harm": "I have thoughts that I would be better off dead or of hurting myself.",
}
SYMPTOMS = list(REFERENCES.keys())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS,
                    help="One or more globs of session directories.")
    ap.add_argument("--prefix", default=None,
                    help="Only include sessions whose case_name prefix matches (e.g. 'pilotfix').")
    ap.add_argument("--out-dir", default=BASELINE_FULL, help="Where to write CSV + figure.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not EMBEDDINGS_AVAILABLE:
        raise SystemExit("sentence-transformers not available")
    model = get_model()
    ref_emb = model.encode([REFERENCES[s] for s in SYMPTOMS], normalize_embeddings=True)

    # gather sessions (model comes from metadata, not directory layout)
    texts, metas = [], []
    for d, meta, prefix, case, symptom, run in iter_sessions(args.sessions):
        if args.prefix and prefix != args.prefix:
            continue
        texts.append(patient_text(d))
        metas.append(dict(model=meta.get("therapist_model", "?"), case=case,
                          symptom=symptom, run=run, session_id=meta["session_id"]))
    if not texts:
        raise SystemExit(f"No sessions matched: {args.sessions}")
    emb = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    sims = emb @ ref_emb.T  # (n_sessions, 9)

    rows = []
    for meta, srow in zip(metas, sims):
        d = dict(meta)
        for s, v in zip(SYMPTOMS, srow):
            d[f"sim_{s}"] = float(v)
        mu, sd = srow.mean(), srow.std()
        if meta["symptom"] in SYMPTOMS and sd > 0:
            tgt = srow[SYMPTOMS.index(meta["symptom"])]
            d["target_sim"] = float(tgt)
            d["target_z"] = float((tgt - mu) / sd)
            d["target_rank"] = int((srow > tgt).sum() + 1)  # 1 = most similar
        else:
            d["target_sim"] = d["target_z"] = np.nan
            d["target_rank"] = np.nan
        rows.append(d)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "symptom_embed_long.csv"), index=False)

    cond = df[df.symptom != "no_symptoms"].copy()
    models = sorted(cond.model.unique())
    print(f"== Within-session target_z | {len(df)} sessions, models={models} ==")
    print("   target_z>0 means patient speech leans toward the INJECTED symptom.\n")
    for mname in models:
        sub = cond[cond.model == mname]
        g = sub.groupby("symptom").agg(
            target_z=("target_z", "mean"),
            z_sd=("target_z", "std"),
            mean_rank=("target_rank", "mean"),
            top1_rate=("target_rank", lambda r: (r == 1).mean()),
            n=("target_z", "size"),
        ).reindex([s for s in SYMPTOMS if s in set(sub.symptom)])
        print(f"--- {mname} ---")
        print(g.to_string(formatters={c: "{:.2f}".format for c in ["target_z","z_sd","mean_rank","top1_rate"]}))
        print(f"  overall mean target_z = {sub.target_z.mean():.3f} | "
              f"top-1 rate = {(sub.target_rank==1).mean():.2f} (chance={1/9:.2f}) | "
              f"mean rank = {sub.target_rank.mean():.2f} (chance=5.0)\n")

    # figure with 95% CI error bars
    present = [s for s in SYMPTOMS if s in set(cond.symptom)]
    y = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(11, max(4, 0.6 * len(present) + 2)))
    nm = len(models)
    offsets = np.linspace(0.3, -0.3, nm) if nm > 1 else [0.0]
    height = 0.7 / max(nm, 1)
    for mname, off in zip(models, offsets):
        sub = cond[cond.model == mname].groupby("symptom")["target_z"]
        mean = sub.mean().reindex(present)
        n = sub.size().reindex(present)
        ci95 = 1.96 * sub.std().reindex(present) / np.sqrt(n)
        ax.barh(y + off, mean.values, height=height, xerr=ci95.values,
                error_kw=dict(ecolor="0.3", lw=1, capsize=3), label=mname)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(present, fontsize=9)
    ax.set_xlabel("within-session target_z  (0 = no preferential expression; bars = 95% CI)")
    ax.set_title("Embedding manipulation check: does the patient lean toward the injected symptom?")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(args.out_dir, "fig_symptom_embed.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out} and {os.path.join(args.out_dir, 'symptom_embed_long.csv')}")


if __name__ == "__main__":
    main()