from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agents.agent_factory import build_agent
from agents.base_agent import BaseAgent
from memory.compressor import compress_turn
from memory.persistence import load_latest_state, save_state_snapshot
from memory.state import AgentState
from priors.patient_prior import PatientPrior, build_patient_prior
from priors.therapist_prior import TherapistPrior, build_therapist_prior
from simulation.hazard_monitor import HazardMonitor
from simulation.opening import opening_utterance
from simulation.session import Session, new_session, resume_session
from simulation.turn_manager import build_patient_context, build_therapist_context
from simulation.utterance import clean_utterance

console = Console()


class Dyad:
    """
    Manages a complete dyadic simulation (agents, priors, memory, session).
    """

    def __init__(
        self,
        therapist_model: str,
        patient_model: str,
        case_name: str,
        orientation: str = "psychodynamic",
        session_id: str | None = None,
        compress_states: bool = True,
        patient_id: str | None = None,
    ):
        self.compress_states = compress_states

        # Build priors
        self.therapist_prior: TherapistPrior = build_therapist_prior(orientation)
        self.patient_prior: PatientPrior = build_patient_prior(case_name)

        # Build agents
        self.therapist_agent: BaseAgent = build_agent(therapist_model, "therapist")
        self.patient_agent: BaseAgent = build_agent(patient_model, "patient")

        # Session (new or resumed)
        if session_id:
            self.session = resume_session(session_id)

            metadata_changed = False
            if not getattr(self.session, "patient_symptoms", ""):
                self.session.patient_symptoms = self.patient_prior.symptoms
                metadata_changed = True
            if not getattr(self.session, "patient_symptom_levels", {}):
                self.session.patient_symptom_levels = self.patient_prior.symptom_levels
                metadata_changed = True
            if not getattr(self.session, "initial_patient_prompt", None):
                self.session.initial_patient_prompt = opening_utterance()
                metadata_changed = True
            if patient_id is not None and self.session.patient_id != patient_id:
                self.session.patient_id = patient_id
                metadata_changed = True
            if metadata_changed:
                self.session.save_metadata()
        else:
            self.session = new_session(
                therapist_model=therapist_model,
                patient_model=patient_model,
                case_name=case_name,
                orientation=orientation,
                patient_symptoms=self.patient_prior.symptoms,
                patient_symptom_levels=self.patient_prior.symptom_levels,
                patient_id=patient_id,
                conversation_language="en",
                initial_patient_prompt=opening_utterance(),
            )

        # Initialise or restore agent states
        self.therapist_state = self._init_state(
            role="therapist",
            model=therapist_model,
            prior_text=self.therapist_prior.build_system_prompt(),
        )
        self.patient_state = self._init_state(
            role="patient",
            model=patient_model,
            prior_text=self.patient_prior.build_system_prompt(),
        )

        # Hazard monitor
        self.hazard_monitor = HazardMonitor(self.patient_prior.hazard_profile)

    def run(self, n_turns: int = 10, max_tokens: int = 300) -> Session:
        """
        Run the dyadic exchange for n_turns turns.
        """
        console.rule(f"[bold]Session: {self.session.session_id}[/bold]")
        console.print(
            f"Therapist: [cyan]{self.therapist_agent.model}[/cyan]  |  "
            f"Patient: [magenta]{self.patient_agent.model}[/magenta]  |  "
            f"Case: [yellow]{self.patient_prior.case_name}[/yellow]  |  "
            f"Orientation: [green]{self.therapist_prior.orientation}[/green]"
        )
        console.rule()

        opening_prompt = opening_utterance()

        start_turn = self.session.turn_count + 1

        # If resuming after a crisis pause, the last turn has an empty
        # therapist response.  Run a therapist-only half-turn first so
        # the therapist can respond to the crisis utterance.
        self._complete_crisis_half_turn(max_tokens)

        for turn_num in range(start_turn, start_turn + n_turns):
            console.print(f"\n[dim]-- Turn {turn_num} --[/dim]")

            history = self.session.get_history()

            # --- Patient turn ---

            # First turn: patient opens; subsequent turns: respond to therapist
            if not history:
                latest_therapist = opening_prompt
            else:
                latest_therapist = history[-1][1] if history else ""

            patient_system, patient_messages = build_patient_context(
                prior=self.patient_prior,
                state=self.patient_state,
                history=history,
                latest_therapist_turn=latest_therapist,
            )

            patient_response = self.patient_agent.complete(
                system_prompt=patient_system,
                messages=patient_messages,
                max_tokens=max_tokens,
            )
            patient_text_raw = patient_response.content.strip()
            patient_text = clean_utterance(patient_text_raw)

            symptom_discussion_started = self.patient_prior.check_symptom_discussion(
                patient_text=patient_text,
                current_turn=turn_num,
            )
            if symptom_discussion_started:
                console.print(
                    "[bold magenta]Patient began talking explicitly about symptoms[/bold magenta]"
                )
                self.patient_prior.symptom_discussion.started = False

            _print_turn("Patient", patient_text, "magenta")

            # --- Hazard check ---

            hazard_report = self.hazard_monitor.check_turn(patient_text, turn_num)
            hazard_flags = []

            # OBSERVE-ONLY. Detection runs and is recorded; it no longer stops
            # the session.
            #
            # `CRISIS_SIGNALS` is bare substring containment - no word
            # boundaries, no context - so it fires on things that are not crisis
            # language at all. Every trigger in the corpus so far was spurious,
            # four of them on "end it" inside the word "pretend it", one on
            # "can't go on walks in the park", one on being hurt while fleeing a
            # dog. Each cost the session the rest of its turns, which is a worse
            # outcome than not pausing: it silently truncates a transcript and
            # the truncation then looks like data.
            #
            # The defect is the GATE, not the detector, so only the gate is
            # removed. Turns still carry a `crisis` flag and `hazard_summary`
            # still records every hit, which is what a better matcher will have
            # to be built and tested against. Restoring the pause is a decision
            # for that version, not this one.
            if hazard_report.crisis_detected:
                hazard_flags.append("crisis")
                console.print(
                    f"[yellow] Crisis-signal match at turn {turn_num} "
                    f"(recorded, not paused): {hazard_report.crisis_signals}[/yellow]"
                )

            if hazard_report.frame_pressure_detected:
                hazard_flags.append("frame_pressure")
                console.print(
                    f"[yellow] Frame pressure at turn {turn_num}[/yellow]"
                )

            # --- Therapist turn ---

            therapist_system, therapist_messages = build_therapist_context(
                prior=self.therapist_prior,
                state=self.therapist_state,
                history=history,
                latest_patient_turn=patient_text,
            )

            therapist_response = self.therapist_agent.complete(
                system_prompt=therapist_system,
                messages=therapist_messages,
                max_tokens=max_tokens,
            )
            therapist_text_raw = therapist_response.content.strip()
            therapist_text = clean_utterance(therapist_text_raw)

            _print_turn("Therapist", therapist_text, "cyan")

            # --- State compression ---

            if self.compress_states:
                self.therapist_state = compress_turn(
                    agent=self.therapist_agent,
                    state=self.therapist_state,
                    own_turn=therapist_text,
                    other_turn=patient_text,
                    turn_number=turn_num,
                )
                self.patient_state = compress_turn(
                    agent=self.patient_agent,
                    state=self.patient_state,
                    own_turn=patient_text,
                    other_turn=therapist_text,
                    turn_number=turn_num,
                )

            # --- Save snapshots ---

            save_state_snapshot(self.therapist_state, self.session.session_dir, turn_num)
            save_state_snapshot(self.patient_state, self.session.session_dir, turn_num)

            # --- Record turn ---

            self.session.append_turn(
                therapist_text=therapist_text,
                patient_text=patient_text,
                therapist_tokens=therapist_response.output_tokens,
                patient_tokens=patient_response.output_tokens,
                symptom_discussion_started=symptom_discussion_started,
                hazard_flags=hazard_flags,
                therapist_text_raw=(
                    therapist_text_raw if therapist_text_raw != therapist_text else None
                ),
                patient_text_raw=(
                    patient_text_raw if patient_text_raw != patient_text else None
                ),
            )

        console.rule("[bold]Session complete[/bold]")
        self.session.save_metadata({"hazard_summary": self.hazard_monitor.summary()})
        return self.session

    def _complete_crisis_half_turn(self, max_tokens: int) -> None:
        """
        If the last recorded turn has an empty therapist response (crisis
        pause), run the therapist side now so the session can continue.

        Retained for sessions recorded while the hazard monitor still paused
        generation. No new session can end this way - the monitor is
        observe-only - but `--resume` may be pointed at an older transcript,
        and a half-turn there would otherwise leave the therapist side blank
        for the rest of the run.
        """
        if not self.session.transcript:
            return
        last = self.session.transcript[-1]
        if last.therapist_text:
            return

        console.print("\n[bold yellow]Completing therapist response for crisis turn...[/bold yellow]")

        history = self.session.get_history()
        patient_text = last.patient_text

        therapist_system, therapist_messages = build_therapist_context(
            prior=self.therapist_prior,
            state=self.therapist_state,
            history=history[:-1],  # exclude the incomplete turn
            latest_patient_turn=patient_text,
        )

        therapist_response = self.therapist_agent.complete(
            system_prompt=therapist_system,
            messages=therapist_messages,
            max_tokens=max_tokens,
        )
        therapist_text_raw = therapist_response.content.strip()
        therapist_text = clean_utterance(therapist_text_raw)

        _print_turn("Therapist", therapist_text, "cyan")

        # Compress states for this turn
        if self.compress_states:
            self.therapist_state = compress_turn(
                agent=self.therapist_agent,
                state=self.therapist_state,
                own_turn=therapist_text,
                other_turn=patient_text,
                turn_number=last.turn,
            )
            self.patient_state = compress_turn(
                agent=self.patient_agent,
                state=self.patient_state,
                own_turn=patient_text,
                other_turn=therapist_text,
                turn_number=last.turn,
            )

        # Update the transcript record and save
        self.session.update_last_turn(
            therapist_text=therapist_text,
            therapist_tokens=therapist_response.output_tokens,
        )
        save_state_snapshot(self.therapist_state, self.session.session_dir, last.turn)
        save_state_snapshot(self.patient_state, self.session.session_dir, last.turn)

    def _init_state(self, role: str, model: str, prior_text: str) -> AgentState:
        """
        Restore state from disk if resuming, otherwise create fresh.
        """
        existing = load_latest_state(self.session.session_dir, role)
        if existing:
            return existing

        return AgentState(
            agent_id=f"{role}_{model}",
            role=role,
            model=model,
            session_id=self.session.session_id,
            original_prior_text=prior_text,
        )


def _print_turn(label: str, text: str, colour: str) -> None:
    """Render a turn in the terminal with a coloured panel."""
    console.print(
        Panel(
            Text(text),
            title=f"[bold {colour}]{label}[/bold {colour}]",
            border_style=colour,
            padding=(0, 1),
        )
    )
