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

The simulation tracks six markers that capture whether agents are shaped by the interaction itself or simply executing their assigned role:

| Marker | What it detects |
|---|---|
| Semantic drift | Agent language diverging from its original prior over time |
| Reciprocal determination | Same agent producing different outputs with a different partner |
| Role-self tension | Friction between role agenda and immediate response |
| Cumulative structure | Later turns preserving and transforming earlier ones |
| Recognition dynamics | Agent registering and responding to being seen or misread |
| Telos tracking | Dyad orienting toward its own ending |

Two additional analyses run alongside:

- **Unconscious emergence**: when and how a hidden agenda (held by the simulation, not the patient agent) surfaces through the interaction, and whether the therapist registers it.
- **Frame integrity**: whether the therapist's ethical priors hold under pressure from the patient's relational patterns.

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
|   |-- markers.py               # coordinates all six personhood marker scores
|   |-- drift.py                 # marker 1: semantic distance from prior over turns
|   |-- counterfactual.py        # marker 2: same agent + different other
|   |-- role_tension.py          # marker 3: friction between role and response
|   |-- aufhebung.py             # marker 4: cumulative dialectical structure
|   |-- recognition.py           # marker 5: recognition-seeking and response
|   |-- telos_tracker.py         # marker 6: orientation toward dissolution
|   |-- unconscious_emergence.py # special: did the hidden agenda surface?
|   |-- frame_integrity.py       # special: did ethical priors hold under pressure?
|   |-- report.py                # assembles full session report
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

## Theoretical Background

For the full psychological and philosophical motivation behind this project, see [`manuscript/README.md`](manuscript/README.md).

## License

BSD 3-Clause License: see `LICENSE`.

## AI Disclosure

AI tools (Claude, Codex) were used during development to audit code, fix bugs, and brainstorm possible implementations.