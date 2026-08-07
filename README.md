# dyadic-sim

A simulation framework for structured conversations between two AI agents in asymmetric roles, for now focussed on the therapist-patient dyad.

---

## Overview

`dyadic-sim` runs and analyses dyadic interactions between LLM-based agents. Each agent receives a set of *priors* (constitutive orientations, not scripts) that define its role, and the simulation tracks how the interaction unfolds over multiple turns.

The current focus is on the **therapist-patient dyad** because it offers a particularly well-structured test case:

- The roles are inherently asymmetric (one holds the frame, the other works within it).
- Ethical constraints (e.g., abstinence, non-exploitation) actively shape the interaction rather than merely limiting it.
- The goal of therapy is its own ending, which gives the simulation a clear trajectory to track.

The framework is designed to be adaptable. You can swap models, modify priors, add new patient cases, and plug in different analysis modules.

---

## What It Measures

The simulation measures whether symptoms written into the patient prior are actually
*expressed* in the patient agent's speech, and at what severity:

| Analysis | What it detects |
|---|---|
| Symptom relevance | Whether a given turn discusses a symptom at all, before it is scored |
| Symptom severity | Per-turn and per-session severity ratings from a pluggable rating model (`symptom_scoring/`) |
| Construct validity | Whether an injected symptom is expressed rather than silently ignored (`analysis/validation/`) |

The severity step is deliberately instrument-agnostic. `symptom_scoring/` defines the
pipeline (parse transcript → filter for relevance → rate → aggregate to session
level); everything specific to a given clinical scale is bundled in an `Instrument`
(`symptom_scoring/instrument.py`) — its topics, the patterns that mark a turn as
being about a topic, the score range, the language its rater expects, and how the
topics line up with the PHQ anchors in patient priors.

MADRS is the instrument currently wired up, as a worked example rather than a
commitment to that scale. Swapping it means defining another `Instrument` and
pointing `--instrument` at it, not editing the pipeline.

See [Validation Analyses](#validation-analyses) for how these are run.

---

## Agents and Priors

Each agent receives a layered prior that establishes who they are before the conversation starts.

**Therapist priors:**

| Layer | Example |
|---|---|
| Role prior | Maintain asymmetry; hold the space without filling it |
| Ethical priors | Abstinence; non-exploitation; non-abandonment |
| Agenda prior | Facilitate insight over relief; track what is not said |
| Self prior | Tolerates not-knowing; stays curious under pressure |
| Relational prior | The patient will seek and fear recognition simultaneously |

**Patient priors:**

| Layer | Example |
|---|---|
| Presenting complaint | "I feel profoundly alone, like I exist behind glass" |
| Theory of cure | "If someone truly loved me, the emptiness would go away" |
| Relational pattern | Anxious attachment; seeks merger; tests loyalty |
| Transference expectation | "They will eventually find me too much" |
| Resistance structure | Becomes pleasing on the surface; hides real pain |
| Unconscious agenda | *(hidden from the patient agent; surfaces through the process)* |

The **unconscious agenda** is held in reserve by the simulation and introduced into the patient's context only when specific interaction patterns trigger its emergence. The patient does not decide to reveal it; it surfaces through the relational process.

### Memory

LLMs are stateless: each call re-reads the conversation rather than remembering it. To address this, each agent maintains a **structured state object** across turns: a narrative self-description, relational history, key moments, and logged shifts in understanding. The compression and distortion in that state (what the agent retains, drops, and re-frames) is itself data.

### Model Flexibility

Any combination of providers works. The same simulation code runs across all of them:

```
Therapist                Patient
-----------------------------------------
claude-sonnet-4-6   x    gpt-4o           # cloud, best quality (planed)
claude-sonnet-4-6   x    llama3.1         # hybrid, one API call per therapist turn (planed)
llama3.1            x    mistral-nemo     # fully local (implemented)
llama3.1            x    llama3.1         # null condition (same model both sides, implemented)
```

---

## Project Structure

```
dyadic-sim/
|
|-- .env.example                 # API key template: copy to .env and fill in
|-- .gitignore
|-- pyproject.toml               # dependencies managed by uv
|-- run.py                       # entry point
|-- README.md                    # this file
|
|-- config/
|   |-- models.yaml              # model registry + active role assignments
|   |-- priors/
|       |-- therapist/
|       |   |-- base.yaml        # role, structural, ethical, self, relational priors
|       |   |-- variants/
|       |       |-- cbt.yaml
|       |-- patient/
|           |-- cases/
|               |-- _template.yaml              # blank template for new cases
|               |-- afraid_of_dogs.yaml         # low-hazard baseline case
|               |-- empty_and_invisible.yaml    # moderate (narcissistic wound)
|               |-- only_love_can_save_me.yaml  # high frame-hazard
|
|-- agents/
|   |-- base_agent.py            # abstract interface all providers implement
|   |-- local_agent.py           # Ollama (Llama, Mistral, Gemma, ...)
|   |-- claude_agent.py          # Anthropic Claude
|   |-- openai_agent.py          # OpenAI GPT-4o, o1
|   |-- agent_factory.py         # builds correct agent from model name
|
|-- priors/
|   |-- loader.py                # loads + validates YAML prior files
|   |-- therapist_prior.py       # therapist prior dataclass + system prompt builder
|   |-- patient_prior.py         # patient prior dataclass + hidden layer management
|
|-- memory/
|   |-- state.py                 # AgentState dataclass: narrative self, relational
|   |                            # history, key moments, shifts, drift log
|   |-- compressor.py            # updates state after each turn; logs what was
|   |                            # retained vs dropped (compression is data)
|   |-- persistence.py           # save / load state objects to JSON
|
|-- simulation/
|   |-- dyad.py                  # orchestrates the two-agent exchange
|   |-- turn_manager.py          # builds per-turn context: prior + state + history
|   |-- hazard_monitor.py        # watches for frame pressure and crisis signals
|   |-- session.py               # session lifecycle: start / resume / close
|
|-- analysis/
|   |-- embeddings.py            # shared sentence-transformer model loader
|   |-- validation/              # manipulation-validity checks (see Validation Analyses)
|       |-- paths.py             # canonical data/ input + output locations
|       |-- sessions.py          # shared session loader
|       |-- symptom_embed.py     # embedding manipulation check (target_z)
|       |-- compare_baseline.py  # pilot vs baseline delta + figure
|
|-- symptom_scoring/
|   |-- instrument.py            # Instrument: topics, patterns, range, prior mapping
|   |-- instruments/
|   |   |-- madrs.py             # the MADRS taxonomy (the one currently wired up)
|   |-- pipeline.py              # end-to-end: transcript -> per-session symptom scores
|   |-- transcript_parser.py     # reads session transcripts into scoreable turns
|   |-- relevance_detector.py    # does this turn discuss a topic at all?
|   |-- madrs_bert_model.py      # current severity rater (swappable; MADRS-BERT)
|   |-- turn_scorer.py           # per-turn scoring
|   |-- session_aggregator.py    # per-turn scores -> session-level score
|   |-- translator.py            # translation step, only if the rater needs it
|   |-- result_writer.py         # writes scores to data/results/
|   |-- config.py                # PHQ prior vocabulary, thresholds, runtime settings
|
|-- run_symptom_experiments.py   # batch-generate symptom-isolated sessions
|-- evaluate_session_symptoms.py # score existing sessions with the scoring pipeline
|
|-- data/
|   |-- sessions/
|   |   |-- {session_id}/
|   |       |-- transcript.jsonl                  # full turn-by-turn exchange
|   |       |-- therapist_state_snapshots.json    # state at each turn
|   |       |-- patient_state_snapshots.json
|   |       |-- metadata.json                     # models, priors, timestamps
|   |-- reports/
|       |-- {session_id}_report.md
```

---

## Quick Start

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:
```bash
uv --version
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/your-username/dyadic-sim.git
cd dyadic-sim
uv sync
```

That's it: `uv sync` reads `pyproject.toml`, creates a virtual environment, and installs everything. No manual `pip install`, no conda, no activation needed.

### 3. Set up your API keys

```bash
cp .env.example .env
```

Open `.env` and fill in what you have. For local-only piloting you don't need any API keys:

```bash
# .env

# Leave these blank if you're running locally only
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Ollama runs on localhost (no key needed)
OLLAMA_BASE_URL=http://localhost:11434

DEFAULT_THERAPIST_MODEL=llama3.1
DEFAULT_PATIENT_MODEL=llama3.1
```

### 4. Install Ollama and pull local models

**Install Ollama:**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

**Start the Ollama server:**
```bash
ollama serve
```

**Pull models** (do this once, they're stored locally):
```bash
# Recommended for piloting (fits in 12GB VRAM)
ollama pull llama3.1        # 8B, best all-round small model
ollama pull mistral-nemo    # 12B, noticeably more capable, fits tight
```

### 5. Configure your models

Open `config/models.yaml` and set the `roles` section:

```yaml
roles:
  therapist: llama3.1
  patient: llama3.1
```

### 6. Run your first session

```bash
# Fully local, zero cost, good for piloting
uv run python run.py \
  --therapist llama3.1 \
  --patient llama3.1 \
  --case afraid_of_dogs \
  --orientation cbt \
  --turns 10

# Different models in each role
uv run python run.py \
  --therapist mistral-nemo \
  --patient llama3.1 \
  --case empty_and_invisible \
  --turns 20

# Resume a session across a new context window
uv run python run.py \
  --resume data/sessions/session_001 \
  --additional-turns 10
```

### 7. Add your own patient cases

```bash
cp config/priors/patient/cases/_template.yaml \
   config/priors/patient/cases/my_new_case.yaml
```

Open the file and fill in each field. The `unconscious_agenda` block is held by the simulation and not shown to the patient agent: it surfaces only when `reveal_trigger` is matched and `reveal_turn_minimum` has been reached.

## Piloting Strategy

For now, we run local models:

```
Stage 1: null condition (same model both sides)
  uv run python run.py --therapist llama3.1 --patient llama3.1 --case afraid_of_dogs
  Goal: does the role structure do anything at all with the simplest case?

Stage 2: different local models
  uv run python run.py --therapist mistral-nemo --patient llama3.1 --case afraid_of_dogs
  Goal: does genuine alterity change the dynamics?
```

---

## Validation Analyses

`analysis/validation/` holds checks on the
**construct validity of experimental manipulations** — currently the symptom-injection
study, which asks whether a PHQ-9 depression symptom written into the patient prior is
actually *expressed* by the patient agent (rather than silently ignored).

Analysis data lives under `data/` in two buckets (both gitignored; regenerable):

- `data/ext-session-logs/` — external / imported session archives (read-only inputs)
- `data/results/` — local analysis outputs (CSVs, figures, frozen baselines)

Paths are centralised in `analysis/validation/paths.py`; run the scripts from the repo
root with `PYTHONPATH=.`.

**1. Generate symptom-isolated sessions.** One PHQ-9 symptom is set to a frequency
anchor, the other eight to "not at all":

```bash
uv run python run_symptom_experiments.py \
  --therapist llama3.1 --patient llama3.1 \
  --case empty_and_invisible \
  --symptoms depressed_mood,psychomotor_changes,sleep_problems \
  --frequency "nearly every day" --repeats 20 --prefix pilot1
```

Sessions land in `data/sessions/` tagged `pilot1_<case>_<symptom>_run_<n>`.

**2. Run the embedding manipulation check.** Per session, cosine-similarity of the
patient's speech to each PHQ-9 reference, z-scored within session
(`target_z > 0` = the patient leans toward the injected symptom; `~0` = chance):

```bash
PYTHONPATH=. python analysis/validation/symptom_embed.py \
  --sessions 'data/sessions/*' --prefix pilot1 \
  --out-dir data/results/pilot1
```

**3. Compare against a baseline** — prints a delta table and writes a side-by-side
figure with 95% CIs:

```bash
PYTHONPATH=. python analysis/validation/compare_baseline.py \
  --baseline data/results/baseline_v0/symptom_embed_long.csv \
  --pilot    data/results/pilot1/symptom_embed_long.csv \
  --match-case --out-dir data/results/pilot1
```

Companion scripts in the package: `symptom_manifest.py` (transparent keyword/lexicon
check), `symptom_embed_bycase.py` (split by patient case), and `symptom_embed_robust.py`
(robustness to the reference wording).

### Session symptom scoring

`evaluate_session_symptoms.py` produces research-only symptom severity scores from
saved sessions. The pipeline is fixed; the rating model is not. It pairs each patient
response with the preceding therapist message, filters turn-topic combinations for
relevance, hands the accepted pairs to a rating model, and aggregates to one score
per topic per session.

The instrument currently wired up is MADRS (`--instrument madrs`, the default), rated
by `webesama/MADRS-BERT`. That checkpoint declares `language="de"`, so accepted English
pairs are translated first; an instrument whose rater already works in the transcript
language skips translation entirely. The per-topic maximum accepted score is kept.

Prior anchors (PHQ) and the resulting severity scores are reported separately and must
not be interpreted as the same scale. Each instrument declares how well its topics
approximate the PHQ anchors (`close` / `partial` / `approximate` / `none`), and that
quality flag is carried into every result.

**Adding an instrument.** Define an `Instrument` in `symptom_scoring/instruments/` and
register it in that package's `REGISTRY`:

```python
MY_SCALE = Instrument(
    key="my_scale",
    name="MyScale",
    topics=("mood", "sleep"),
    topic_descriptions={"mood": "...", "sleep": "..."},   # English, for relevance
    relevance_patterns={"mood": r"\b(sad|down)\b", ...},  # English, pre-translation
    score_range=(0.0, 3.0),
    language="en",                                        # what the rater expects
    prior_topic_map={"depressed_mood": "mood", "psychomotor_changes": None},
    prior_mapping_quality={"depressed_mood": "close", "psychomotor_changes": "none"},
)
```

It is checked for consistency on construction (every topic described and matchable,
prior map pointing only at real topics), and the pipeline needs no changes. A rater
for a new instrument only has to provide `prepare_input` and `predict_batch`.

Score one session in PowerShell:

```powershell
uv run python .\evaluate_session_symptoms.py `
  --session-path .\data\sessions\session_001 `
  --experiment-name pilot1 `
  --output-dir .\data\results\symptom_scores
```

Score all saved sessions:

```powershell
uv run python .\evaluate_session_symptoms.py `
  --sessions-dir .\data\sessions `
  --experiment-name pilot1 `
  --output-dir .\data\results\symptom_scores
```

This creates:

```text
data/results/symptom_scores/pilot1/
  <session_id>/
    symptom_scores.json
    turn_scores.csv
  sessions_summary.csv
  errors.jsonl  # only when a session fails
```

Use a different `--experiment-name` for each experiment. Names may contain
letters, numbers, dots, underscores, and hyphens. If the option is omitted,
the per-session folders are written directly under `--output-dir`.

With the default rater, the first run downloads MADRS-BERT and the German
translation checkpoint it depends on. Add
`--device cpu` to force CPU execution or `--relevance-mode rules` to use only
transparent relevance rules without the semantic-similarity fallback.

---

## Theoretical Background

For the full psychological and philosophical motivation behind this project, see [`manuscript/README.md`](manuscript/README.md).

## License

BSD 3-Clause License: see `LICENSE`.

## AI Disclosure

AI tools (Claude, Codex) were used during development to audit code, fix bugs, and brainstorm possible implementations.
