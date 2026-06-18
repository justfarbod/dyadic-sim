"""
Compare post-fix pilot vs frozen baseline on within-session target_z.

Reads two `symptom_embed_long.csv` files (baseline + pilot), restricts to the
symptoms present in BOTH and the same case(s), and plots target_z side by side
with 95% CI, per model. Prints a delta table (pilot - baseline).

Usage:
  PYTHONPATH=. python analysis/validation/compare_baseline.py \
      --baseline data/results/baseline_v0/symptom_embed_long.csv \
      --pilot    data/results/pilotfix/symptom_embed_long.csv \
      --out-dir  data/results/pilotfix

Success criterion for the S1 fix: pilot target_z rises (toward/above 0) for the
injected symptom relative to baseline, especially for verbalizable symptoms
(depressed_mood, psychomotor_changes).
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.validation.paths import RESULTS, BASELINE_V0


def summarise(df):
    df = df[df.symptom != "no_symptoms"].copy()
    g = (df.groupby(["model", "symptom"])
           .agg(z=("target_z", "mean"), sd=("target_z", "std"), n=("target_z", "size"))
           .reset_index())
    g["ci95"] = 1.96 * g["sd"] / np.sqrt(g["n"])
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=os.path.join(BASELINE_V0, "symptom_embed_long.csv"))
    ap.add_argument("--pilot", required=True, help="Pilot symptom_embed_long.csv")
    ap.add_argument("--out-dir", default=os.path.join(RESULTS, "pilotfix"))
    ap.add_argument("--match-case", action="store_true",
                    help="Restrict baseline to the case(s) present in the pilot (fair comparison).")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base = pd.read_csv(args.baseline)
    pilot = pd.read_csv(args.pilot)

    if args.match_case:
        cases = sorted(pilot.case.unique())
        base = base[base.case.isin(cases)]
        print(f"matched baseline to pilot case(s): {cases}")

    bsum, psum = summarise(base), summarise(pilot)
    # symptoms & models common to both
    symptoms = sorted(set(bsum.symptom) & set(psum.symptom))
    models = sorted(set(bsum.model) & set(psum.model))
    if not symptoms or not models:
        raise SystemExit(f"No overlap. baseline models={sorted(set(bsum.model))} "
                         f"symptoms={sorted(set(bsum.symptom))}; pilot models={sorted(set(psum.model))} "
                         f"symptoms={sorted(set(psum.symptom))}")

    # delta table
    merged = psum.merge(bsum, on=["model", "symptom"], suffixes=("_pilot", "_base"))
    merged["delta"] = merged["z_pilot"] - merged["z_base"]
    print("\n== target_z: baseline -> pilot (delta = pilot - baseline) ==")
    for m in models:
        sub = merged[merged.model == m].set_index("symptom").reindex(symptoms)
        print(f"\n--- {m} ---")
        print(sub[["z_base", "z_pilot", "delta", "n_pilot"]].to_string(
            formatters={c: "{:+.2f}".format for c in ["z_base", "z_pilot", "delta"]}))
    merged.to_csv(os.path.join(args.out_dir, "compare_baseline.csv"), index=False)

    # figure: per model, baseline vs pilot bars with CI
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), max(4, 0.6 * len(symptoms) + 2)),
                             squeeze=False, sharey=True)
    y = np.arange(len(symptoms))
    for ax, m in zip(axes[0], models):
        b = bsum[bsum.model == m].set_index("symptom").reindex(symptoms)
        p = psum[psum.model == m].set_index("symptom").reindex(symptoms)
        ax.barh(y + 0.2, b["z"].values, height=0.38, xerr=b["ci95"].values,
                error_kw=dict(ecolor="0.3", lw=0.8, capsize=2), label="baseline")
        ax.barh(y - 0.2, p["z"].values, height=0.38, xerr=p["ci95"].values,
                error_kw=dict(ecolor="0.3", lw=0.8, capsize=2), label="pilot (post-fix)")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(symptoms, fontsize=9)
        ax.set_title(m); ax.set_xlabel("within-session target_z (95% CI)")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("S1 fix: injected-symptom expression, baseline vs post-fix pilot")
    fig.tight_layout()
    out = os.path.join(args.out_dir, "fig_compare_baseline.png")
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out} and {os.path.join(args.out_dir, 'compare_baseline.csv')}")


if __name__ == "__main__":
    main()