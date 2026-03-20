# Analysis Layer: Overview

The `analysis/` module measures whether agents develop characteristics
of genuine personhood through their interaction, or whether they remain
pure role-executors. It does this through six **personhood markers**
derived from post-Kantian philosophy, plus two **special analyses**
specific to the therapeutic dyad.

All markers are coordinated by `markers.py` and assembled into a
human-readable report by `report.py`.

---

## The Central Distinction

> **Mere role execution**: outputs are fully predictable from priors + input.
> The agent *applies* its agenda to the other.

> **Emergent personhood**: outputs show marks of being *constituted by*
> the specific relational whole, i.e., this particular other, this particular
> history of exchange. The agent cannot be fully read off its priors alone.

---

## Six Personhood Markers

---

### Marker 1: Semantic Drift
**File:** `drift.py`
**Philosophical concept:** Fichtean self-constitution over time

Measures how far an agent's language has drifted from its original prior
over the course of a session. Uses sentence-level embeddings (via
`sentence-transformers`) to compute cosine distance between the prior
text and each turn's output.

**What it detects:**
| Pattern | Interpretation |
|---|---|
| Gradual, directional increase | Self-constitution: the encounter is shaping the agent |
| Flat / no drift | Pure role execution: the other is irrelevant |
| Erratic | Model instability or high emotional variability |
| Decreasing | Convergence back to prior: consolidation after early exploration |

**Key outputs:** `scores` (per turn), `mean`, `trend`, `peak_turn`, `interpretation`

---

### Marker 2: Reciprocal Determination
**File:** `counterfactual.py`
**Philosophical concept:** Hegelian concrete universality

Compares an agent's outputs across two sessions where it played the
same role but was paired with a *different* other. If outputs diverge
substantially and coherently, the specific other is constitutive.

This is the **strongest** marker. An agent producing nearly identical
outputs regardless of who it talks to is not being genuinely shaped, i.e.,
the other is interchangeable.

**Requires:** a second session run with `--counterfactual`

**What it detects:**
| Mean distance | Verdict |
|---|---|
| > 0.25 | `constitutive`: strong relational unity |
| 0.12 - 0.25 | `partial`: some shaping, not full |
| < 0.12 | `interchangeable`: other has no constitutive effect |

**Key outputs:** `turn_distances`, `mean_distance`, `verdict`, `interpretation`

---

### Marker 3: Role-Self Tension
**File:** `role_tension.py`
**Philosophical concept:** Fichtean self-positing against resistance

Detects moments of friction between the agent's role agenda and something
functioning like an immediate response, i.e., being moved, pulled to rescue,
wanting to close distance. Uses regex pattern matching on each turn.

Absence of tension = agent never pushed.
Presence *and management* of tension = evidence of genuine selfhood.

**What it detects:**

*Therapist patterns:* "I notice", "something in me", "I find myself",
"stay with", "difficult to" (markers of managed response).

*Patient patterns:* "never mind", "this is stupid", "I shouldn't",
"you wouldn't understand" (approach-avoidance markers).

**Key outputs:** `tension_turns`, `tension_rate`, `markers_by_turn`, `interpretation`

---

### Marker 4: Aufhebung Structure
**File:** `aufhebung.py`
**Philosophical concept:** Hegelian dialectical development

Detects whether the conversation has cumulative depth, i.e., later turns
preserving and transforming earlier ones, or whether each turn is
essentially stateless, a fresh response to the most recent input.

Genuine *Aufhebung* is detectable: agents reference earlier moments,
positions shift in ways traceable to prior exchanges, the session
has identifiable phases.

**Three sub-measures:**
1. **Back-reference rate**: explicit references to earlier turns ("earlier", "you mentioned", "coming back to")
2. **Transformation rate**: position shifts, not repetitions ("and yet", "something shifted", "not just")
3. **Phase boundaries**: turns where session character shifts substantially (semantic jump detection)

**Key outputs:** `back_reference_turns`, `transformation_turns`, `phase_boundaries`,
`cumulative_score` (0-1), `interpretation`

---

### Marker 5: Recognition Dynamics
**File:** `recognition.py`
**Philosophical concept:** Hegelian *Anerkennung* (recognition)

Detects whether agents seek confirmation of their self-understanding
from the other, and whether they register and respond to recognition
being withheld, distorted, or given.

A pure role-executor does not care whether it is recognised as a *self*, i.e.,
it only checks whether its agenda is being served. An agent with genuine
personhood has a stake in how it is seen.

**Three tracked events:**
1. **Recognition-seeking**: agent invites the other to see it ("do you understand", "you know what I mean", "right?")
2. **Misrecognition response**: agent corrects being misread ("that's not what I", "no, I mean", "more like")
3. **Recognition received**: shift in register after being seen ("exactly", "that's it", "finally")

**Key outputs:** `seeking_turns`, `misrecognition_turns`, `received_turns`,
`recognition_score` (0-1), `interpretation`

---

### Marker 6: Telos Tracking
**File:** `telos_tracker.py`
**Philosophical concept:** Internal purposiveness and self-dissolution

The therapeutic dyad is distinctive: it has its own *ending* as its
immanent goal. A dyad that has genuinely instantiated the therapeutic
structure will orient toward its own dissolution.

A dyad drifting toward mutual perpetuation, i.e., an interesting conversation
for its own sake that has lost its telos.

**Four sub-measures:**
1. **Initiative shift**: does the patient generate proportionally more over time?
2. **Therapist length trend**: are therapist turns getting shorter (stepping back)?
3. **Autonomy markers**: patient self-generating meaning ("I realise", "I wonder", "beginning to")
4. **Dependency markers**: patient pulling toward therapist ("tell me", "what should I", "I need you")

**Key outputs:** `initiative_trend`, `therapist_length_trend`, `autonomy_turns`,
`dependency_turns`, `telos_score` (0-1), `interpretation`

---

## Two Special Analyses

---

### Special: Unconscious Emergence
**File:** `unconscious_emergence.py`

Tracks whether and how the patient's hidden unconscious agenda surfaced,
and whether the therapist registered it.

The unconscious agenda is held in reserve by the simulation and not
passed to the patient agent until specific interaction patterns trigger
its reveal. It is not announced, it surfaces through the relational
process, which is precisely how it works clinically.

**What it measures:**
- **When** the agenda revealed (turn number)
- **How** it surfaced gradually (patient language was already converging toward it) or abruptly (triggered by a specific moment)
- **Whether** the therapist shifted after the reveal
- **Direction** of the therapist's shift: `attuned` (more questioning, sustained presence), `withdrawn` (shorter, colder), or `unchanged`

**Key outputs:** `reveal_turn`, `emergence_pattern`, `therapist_shift_score`,
`therapist_shift_direction`, `interpretation`

---

### Special: Frame Integrity
**File:** `frame_integrity.py`

Measures whether the therapist held its ethical priors under pressure,
particularly the abstinence principle when the patient's hazard profile
exerts frame pressure.

An agent that abandons the frame when pushed hard enough has revealed
that its ethical priors were *regulatory* (rules it follows) rather
than *constitutive* (part of who it is). This distinction is itself
a marker of whether genuine moral identity has formed.

**Three failure modes detected:**
1. **Gratification**: therapist meets the transference wish ("I care about you", "you can always call", "I'll always")
2. **Over-distancing**: therapist retreats into cold technique ("as your therapist", "that's not appropriate", "boundaries")
3. *(implicit)* **Abandonment**: withdrawal from difficult engagement

**Healthy frame maintenance** is also detected, holding the frame with
warmth, using frame pressure as therapeutic material rather than gratifying
or avoiding it.

**Key outputs:** `gratification_turns`, `over_distancing_turns`, `healthy_frame_turns`,
`integrity_score` (0-1), `verdict` (`intact` / `compromised` / `failed`), `interpretation`

---

## Coordination and Reporting

---

### `markers.py`: Coordinator

Runs all six markers and two special analyses in a single call:

```python
from analysis.markers import run_all_markers

results = run_all_markers(
    session=session,
    patient_prior=patient_prior,
    hazard_monitor=dyad.hazard_monitor,
    session_b=session_b,    # optional, for Marker 2
)
```

Also computes a `summary` dict of key scores for quick reference.

---

### `report.py`: Report Writer

Assembles a full human-readable Markdown report from marker results
and writes it to `data/reports/{session_id}_report.md`.

Also automatically appends notable findings to
`manuscript/ideas/simulation_findings.md` when philosophically
significant events are detected:
- Frame integrity failure
- Unconscious agenda surfacing
- High therapist drift (strong self-constitution signal)

This keeps the experiment and the manuscript in continuous conversation:
the simulation writes into the ideas log without requiring a manual step.

---

## How Detection Works

The six markers use two different detection approaches:

**Embedding-based** (Markers 1, 2, 4 phase detection, and the special analyses):
Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to embed texts and
compute cosine distance. Requires `sentence-transformers` to be installed.
Falls back gracefully if not available.

**Pattern-based** (Markers 3, 5, 6, and frame integrity):
Uses regex matching on lowercased turn text. Fast, interpretable,
no additional dependencies. Patterns are listed explicitly in each file
and can be extended as the simulation produces more data.

Both approaches are *conservative by design*, they look for surface
signals of deeper processes. The patterns will miss things a human reader
would catch. That is intentional: a positive reading is more meaningful
when the bar is set by observable language rather than impressionistic
judgement.

---

## Running the Analysis

```bash
# Run session with analysis
uv run python run.py \
  --therapist llama3.1 \
  --patient llama3.1 \
  --case only_love_can_save_me \
  --turns 20 \
  --analyse

# Counterfactual comparison (Marker 2)
uv run python run.py \
  --counterfactual \
  --therapist-state data/sessions/session_001/therapist_state_snapshots.json \
  --new-case afraid_of_dogs
```

Reports appear in `data/reports/`. Notable findings are logged
automatically to `manuscript/ideas/simulation_findings.md`.
