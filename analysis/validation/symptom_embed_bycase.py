"""
Symptom manipulation check, SPLIT BY CASE.

Reuses the per-session within-session target_z from symptom_embed.py's output
(symptom_embed_long.csv) and breaks it down by the 3 base cases, which differ
sharply in baseline tone (afraid_of_dogs = phobia, empty_and_invisible &
only_love_can_save_me = depressive/narcissistic). Each case has 5 runs per
(model, symptom) cell.

Outputs:
  - symptom_embed_bycase.csv     model x case x symptom means (+ sd, top1)
  - fig_symptom_embed_bycase.png faceted by case
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.validation.paths import BASELINE_FULL

HERE = BASELINE_FULL  # full 300-session baseline outputs are read & written here
LONG = os.path.join(HERE, "symptom_embed_long.csv")

SYMPTOMS = [
    "lack_of_pleasure", "depressed_mood", "sleep_problems", "low_energy",
    "appetite_changes", "feelings_of_failure_or_guilt", "concentration_problems",
    "psychomotor_changes", "thoughts_of_death_or_self_harm",
]
CASES = ["afraid_of_dogs", "empty_and_invisible", "only_love_can_save_me"]
MODELS = ["llama3.1", "mistral-nemo"]


def main():
    df = pd.read_csv(LONG)
    cond = df[df.symptom != "no_symptoms"].copy()

    # model x case x symptom summary
    g = (cond.groupby(["model", "case", "symptom"])
              .agg(target_z=("target_z", "mean"),
                   z_sd=("target_z", "std"),
                   n=("target_z", "size"),
                   top1=("target_rank", lambda r: (r == 1).mean()))
              .reset_index())
    g.to_csv(os.path.join(HERE, "symptom_embed_bycase.csv"), index=False)

    # overall mean target_z per model x case (the key "does it work here?" number)
    print("== mean within-session target_z by model x case ==")
    print("   (>0 = patient leans toward injected symptom; 0 = chance)\n")
    piv = (cond.groupby(["model", "case"])["target_z"].mean().unstack("case")
                .reindex(index=MODELS, columns=CASES))
    print(piv.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\n== top-1 rate by model x case (chance = 0.111) ==")
    pivt = (cond.assign(top1=(cond.target_rank == 1))
                .groupby(["model", "case"])["top1"].mean().unstack("case")
                .reindex(index=MODELS, columns=CASES))
    print(pivt.to_string(float_format=lambda x: f"{x:.2f}"))

    # faceted figure: one panel per case, bars per symptom, color per model.
    # Error bars = 95% CI (1.96*SEM); n=5 per cell so they are wide.
    g["sem"] = g["z_sd"] / np.sqrt(g["n"])
    g["ci95"] = 1.96 * g["sem"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True, sharex=True)
    y = np.arange(len(SYMPTOMS))
    for ax, case in zip(axes, CASES):
        for i, model in enumerate(MODELS):
            gm = g[(g.case == case) & (g.model == model)].set_index("symptom").reindex(SYMPTOMS)
            ax.barh(y + (0.2 if i == 0 else -0.2), gm["target_z"].values, height=0.38,
                    xerr=gm["ci95"].values,
                    error_kw=dict(ecolor="0.3", lw=0.8, capsize=2), label=model)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(SYMPTOMS, fontsize=8)
        ax.set_title(case)
        ax.set_xlabel("within-session target_z  (bars = 95% CI, n=5)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Manipulation check by case: does the patient lean toward the injected symptom?")
    fig.tight_layout()
    out = os.path.join(HERE, "fig_symptom_embed_bycase.png")
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out} and symptom_embed_bycase.csv")


if __name__ == "__main__":
    main()