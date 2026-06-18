"""
Reference-robustness check.

The embedding manipulation check (symptom_embed.py) compared patient speech to
hand-written reference sentences. Magnitudes therefore depend on *my* wording.
Here we repeat the identical within-session target_z analysis using the EXACT
PHQ-9 item labels that the simulation itself put into the patient prompt
(priors.patient_prior._format_symptoms -> symptom_labels). If the +/- signature
survives a reference swap, it reflects the data, not the wording.

We run THREE reference sets and correlate the per-(model,symptom) target_z:
  A: hand-written sentences (original, from symptom_embed.py)
  B: PHQ-9 short labels (what the patient actually saw)
  C: PHQ-9 label + "nearly every day" frequency clause (full prompt line)

Outputs:
  - symptom_embed_robust.csv          target_z per model x symptom x refset
  - fig_symptom_embed_robust.png      A vs B vs C, with 95% CI
  - prints Pearson r between ref sets (agreement of the signature)
"""
from __future__ import annotations
import json, glob, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.embeddings import get_model, EMBEDDINGS_AVAILABLE
from analysis.validation.paths import EXTERNAL_SETS, BASELINE_FULL

HERE = BASELINE_FULL  # outputs written next to the full 300-session baseline
SETS = EXTERNAL_SETS

SYMPTOMS = [
    "lack_of_pleasure", "depressed_mood", "sleep_problems", "low_energy",
    "appetite_changes", "feelings_of_failure_or_guilt", "concentration_problems",
    "psychomotor_changes", "thoughts_of_death_or_self_harm",
]

# A: original hand-written sentences
REF_A = {
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
# B: EXACT PHQ-9 labels from patient_prior.py:458-468 (what the patient saw)
REF_B = {
    "lack_of_pleasure": "Little interest or pleasure in doing things",
    "depressed_mood": "Feeling down, depressed, or hopeless",
    "sleep_problems": "Sleep problems",
    "low_energy": "Feeling tired or having little energy",
    "appetite_changes": "Poor appetite or overeating",
    "feelings_of_failure_or_guilt": "Feeling bad about yourself, guilty, or like a failure",
    "concentration_problems": "Trouble concentrating",
    "psychomotor_changes": "Moving or speaking slowly, or feeling restless",
    "thoughts_of_death_or_self_harm": "Thoughts that you would be better off dead or of hurting yourself",
}
# C: full prompt line with the active-symptom frequency clause
_CLAUSE = "; over the last two weeks, you have been bothered by this symptom nearly every day"
REF_C = {k: v + _CLAUSE for k, v in REF_B.items()}
REFSETS = {"A_handwritten": REF_A, "B_phq9_label": REF_B, "C_phq9_full_line": REF_C}


def parse_case(case_name):
    m = re.match(r"batch_(.+?)_run_(\d+)$", case_name)
    body, run = m.group(1), int(m.group(2))
    for c in ("only_love_can_save_me", "empty_and_invisible", "afraid_of_dogs"):
        if body.startswith(c):
            return c, (body[len(c):].lstrip("_") or "no_symptoms"), run
    return body, "?", run


def patient_text(d):
    tx = [json.loads(l) for l in open(os.path.join(d, "transcript.jsonl")) if l.strip()]
    chunks = []
    for i, t in enumerate(tx):
        p = (t.get("patient_text") or "").strip()
        if i > 0 and p and p == (tx[i-1].get("therapist_text") or "").strip():
            continue
        if p:
            chunks.append(p)
    return " ".join(chunks)


def main():
    if not EMBEDDINGS_AVAILABLE:
        raise SystemExit("sentence-transformers not available")
    model = get_model()

    # gather sessions + patient embeddings once
    meta_rows, texts = [], []
    for mname, pat in SETS.items():
        for d in sorted(glob.glob(pat)):
            mp = os.path.join(d, "metadata.json")
            if not (os.path.isdir(d) and os.path.exists(mp)):
                continue
            meta = json.load(open(mp))
            case, symptom, run = parse_case(meta["case_name"])
            if symptom not in SYMPTOMS:
                continue  # drop no_symptoms (no target)
            meta_rows.append(dict(model=mname, case=case, symptom=symptom, run=run))
            texts.append(patient_text(d))
    emb = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    rows = []
    for refname, refmap in REFSETS.items():
        ref_emb = model.encode([refmap[s] for s in SYMPTOMS], normalize_embeddings=True)
        sims = emb @ ref_emb.T
        for meta, srow in zip(meta_rows, sims):
            mu, sd = srow.mean(), srow.std()
            tgt = srow[SYMPTOMS.index(meta["symptom"])]
            rows.append(dict(refset=refname, **meta,
                             target_z=float((tgt - mu) / sd) if sd > 0 else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "symptom_embed_robust.csv"), index=False)

    # per model x symptom x refset mean + CI
    agg = (df.groupby(["refset", "model", "symptom"])
             .agg(z=("target_z", "mean"), sd=("target_z", "std"), n=("target_z", "size"))
             .reset_index())
    agg["ci95"] = 1.96 * agg["sd"] / np.sqrt(agg["n"])

    # agreement of the signature across reference sets (per model)
    print("== Pearson r of per-symptom target_z between reference sets ==")
    print("   (high r = the +/- signature is robust to reference wording)\n")
    wide = agg.pivot_table(index=["model", "symptom"], columns="refset", values="z")
    for mname in SETS:
        w = wide.loc[mname]
        rAB = np.corrcoef(w["A_handwritten"], w["B_phq9_label"])[0, 1]
        rAC = np.corrcoef(w["A_handwritten"], w["C_phq9_full_line"])[0, 1]
        rBC = np.corrcoef(w["B_phq9_label"], w["C_phq9_full_line"])[0, 1]
        print(f"  {mname}: r(A,B)={rAB:.2f}  r(A,C)={rAC:.2f}  r(B,C)={rBC:.2f}")
    print()
    print("== overall mean target_z by refset x model (should stay ~0) ==")
    print(df.groupby(["refset", "model"])["target_z"].mean()
            .unstack("model").to_string(float_format=lambda x: f"{x:+.3f}"))

    # figure: per symptom, 3 refsets side by side, pooled across models
    pooled = (df.groupby(["refset", "symptom"])
                .agg(z=("target_z", "mean"), sd=("target_z", "std"), n=("target_z", "size"))
                .reset_index())
    pooled["ci95"] = 1.96 * pooled["sd"] / np.sqrt(pooled["n"])
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(SYMPTOMS))
    offs = {"A_handwritten": 0.27, "B_phq9_label": 0.0, "C_phq9_full_line": -0.27}
    for refname, off in offs.items():
        sub = pooled[pooled.refset == refname].set_index("symptom").reindex(SYMPTOMS)
        ax.barh(y + off, sub["z"].values, height=0.26, xerr=sub["ci95"].values,
                error_kw=dict(ecolor="0.3", lw=0.8, capsize=2), label=refname)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(SYMPTOMS, fontsize=9)
    ax.set_xlabel("within-session target_z (pooled over models; bars = 95% CI)")
    ax.set_title("Reference-robustness: target_z under 3 reference wordings")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_symptom_embed_robust.png")
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out} and symptom_embed_robust.csv")


if __name__ == "__main__":
    main()