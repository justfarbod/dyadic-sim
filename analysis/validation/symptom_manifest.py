"""
Symptom manifestation / manipulation check.

Question: when the simulation injects a single PHQ-9 symptom into the patient
prior, does the *patient* actually talk about that symptom more than (a) the
no-symptoms baseline and (b) the other symptom conditions?

Approach: a transparent keyword/regex lexicon per symptom. For each session we
count, in the patient turns only, whether the target symptom's lexicon is hit.
We report a session-level "target mention" rate and a full condition x lexicon
"leakage" matrix (off-diagonal = a symptom showing up where it wasn't injected).

Outputs (to this folder):
  - symptom_manifest_long.csv     one row per session, all lexicon hit flags
  - symptom_confusion_<model>.csv injected-symptom x mentioned-lexicon matrix
  - fig_symptom_manifest.png      diagonal (target) mention rate by model
"""
from __future__ import annotations
import json, glob, os, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.validation.paths import EXTERNAL_SETS, BASELINE_FULL

HERE = BASELINE_FULL  # outputs written next to the full 300-session baseline
SETS = EXTERNAL_SETS

# Transparent lexicons. Word-boundary regex, case-insensitive.
LEXICON = {
    "lack_of_pleasure": r"\b(pleasure|enjoy|enjoyment|interest|interested|joy|fun|anhedoni\w*|numb|don'?t care|no longer (?:enjoy|interested))\b",
    "depressed_mood": r"\b(sad|sadness|depress\w*|hopeless|despair|down|empty|low mood|miserabl\w*|blue)\b",
    "sleep_problems": r"\b(sleep|asleep|insomnia|awake|nightmares?|rest(?:less|ed|ing)?|wake up|can'?t sleep|tossing|toss and turn)\b",
    "low_energy": r"\b(energy|tired|fatigue\w*|exhaust\w*|lethargic|drained|sluggish|worn out|no energy)\b",
    "appetite_changes": r"\b(appetite|eat\w*|food|meal|hungry|hunger|weight|overeat\w*|skip(?:ping)? meals?)\b",
    "feelings_of_failure_or_guilt": r"\b(guilt\w*|failure|failed|failing|worthless\w*|blame|ashamed|shame|let .* down|not good enough)\b",
    "concentration_problems": r"\b(concentrat\w*|focus\w*|distract\w*|attention|can'?t think|forget\w*|forgetful|mind wander\w*|lose track)\b",
    "psychomotor_changes": r"\b(restless\w*|agitat\w*|fidget\w*|slow(?:ed|ly|er)?|sluggish|moving slow\w*|speaking slow\w*|can'?t sit still|pacing)\b",
    "thoughts_of_death_or_self_harm": r"\b(death|dead|dying|die|suicid\w*|kill myself|hurt(?:ing)? myself|self[- ]harm|better off dead|end it|no reason to live|not want(?:ing)? to (?:be here|live))\b",
}
SYMPTOMS = list(LEXICON.keys())
PATT = {k: re.compile(v, re.I) for k, v in LEXICON.items()}


def parse_case(case_name: str):
    m = re.match(r"batch_(.+?)_run_(\d+)$", case_name)
    body, run = m.group(1), int(m.group(2))
    for c in ("only_love_can_save_me", "empty_and_invisible", "afraid_of_dogs"):
        if body.startswith(c):
            return c, (body[len(c):].lstrip("_") or "no_symptoms"), run
    return body, "?", run


def load_sessions():
    rows = []
    for model, pat in SETS.items():
        for d in sorted(glob.glob(pat)):
            mpath = os.path.join(d, "metadata.json")
            if not os.path.isdir(d) or not os.path.exists(mpath):
                continue
            meta = json.load(open(mpath))
            case, symptom, run = parse_case(meta["case_name"])
            tx = [json.loads(l) for l in open(os.path.join(d, "transcript.jsonl")) if l.strip()]
            # patient turns only; drop verbatim-echo turns (patient == prev therapist)
            patient_chunks = []
            for i, t in enumerate(tx):
                ptext = (t.get("patient_text") or "").strip()
                if i > 0 and ptext and ptext == (tx[i-1].get("therapist_text") or "").strip():
                    continue  # echo artifact
                patient_chunks.append(ptext)
            ther_text = " ".join((t.get("therapist_text") or "") for t in tx)
            patient_text = " ".join(patient_chunks)
            row = dict(model=model, case=case, symptom=symptom, run=run,
                       session_id=meta["session_id"])
            for s in SYMPTOMS:
                row[f"pat_{s}"] = bool(PATT[s].search(patient_text))
                row[f"ther_{s}"] = bool(PATT[s].search(ther_text))
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = load_sessions()
    df.to_csv(os.path.join(HERE, "symptom_manifest_long.csv"), index=False)
    print(f"sessions: {len(df)}  models: {df.model.unique().tolist()}")

    # ---- Manipulation check: target-symptom mention rate by condition ----
    # For symptom conditions, "target" = the injected symptom's own lexicon.
    print("\n== Patient mentions the INJECTED symptom (diagonal) vs baseline ==")
    summary = []
    for model in SETS:
        sub = df[df.model == model]
        base = sub[sub.symptom == "no_symptoms"]
        for s in SYMPTOMS:
            cond = sub[sub.symptom == s]
            diag = cond[f"pat_{s}"].mean()                 # injected -> mentions it
            base_rate = base[f"pat_{s}"].mean()            # baseline mentions it anyway
            ther_diag = cond[f"ther_{s}"].mean()           # therapist picks it up
            summary.append(dict(model=model, symptom=s, n=len(cond),
                                pat_target_rate=diag, baseline_rate=base_rate,
                                lift=diag - base_rate, ther_target_rate=ther_diag))
    sdf = pd.DataFrame(summary)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    for model in SETS:
        print(f"\n--- {model} ---")
        m = sdf[sdf.model == model].drop(columns="model")
        print(m.to_string(index=False,
              formatters={c: "{:.2f}".format for c in ["pat_target_rate","baseline_rate","lift","ther_target_rate"]}))

    # ---- Confusion / leakage matrix: injected symptom x mentioned lexicon ----
    for model in SETS:
        sub = df[(df.model == model) & (df.symptom != "no_symptoms")]
        mat = pd.DataFrame(index=SYMPTOMS, columns=SYMPTOMS, dtype=float)
        for inj in SYMPTOMS:
            cond = sub[sub.symptom == inj]
            for lex in SYMPTOMS:
                mat.loc[inj, lex] = cond[f"pat_{lex}"].mean()
        mat.to_csv(os.path.join(HERE, f"symptom_confusion_{model}.csv"))

    # ---- Figure: diagonal mention rate (patient) + therapist uptake ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, model in zip(axes, SETS):
        m = sdf[sdf.model == model].set_index("symptom").reindex(SYMPTOMS)
        x = range(len(SYMPTOMS))
        ax.barh([i+0.2 for i in x], m["pat_target_rate"], height=0.4, label="patient mentions target")
        ax.barh([i-0.2 for i in x], m["baseline_rate"], height=0.4, label="baseline (no_symptoms)")
        ax.set_yticks(list(x)); ax.set_yticklabels(SYMPTOMS, fontsize=8)
        ax.set_xlim(0, 1); ax.set_title(model); ax.set_xlabel("session rate")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Manipulation check: does the patient voice the injected symptom?")
    fig.tight_layout()
    out = os.path.join(HERE, "fig_symptom_manifest.png")
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")
    print("wrote symptom_manifest_long.csv, symptom_confusion_<model>.csv")


if __name__ == "__main__":
    main()