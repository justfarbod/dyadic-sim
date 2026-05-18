from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agents.base_agent import BaseAgent
from agents.agent_factory import build_agent
from memory.state import AgentState
from memory.compressor import compress_turn
from memory.persistence import save_state_snapshot, load_latest_state
from priors.therapist_prior import TherapistPrior, build_therapist_prior
from priors.patient_prior import PatientPrior, build_patient_prior
from simulation.session import Session, new_session, resume_session
from simulation.turn_manager import build_therapist_context, build_patient_context
from simulation.hazard_monitor import HazardMonitor

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

            if not getattr(self.session, "patient_symptoms", ""):
                self.session.patient_symptoms = self.patient_prior.symptoms
                self.session.save_metadata()
        else:
            self.session = new_session(
                therapist_model=therapist_model,
                patient_model=patient_model,
                case_name=case_name,
                orientation=orientation,
                patient_symptoms=self.patient_prior.symptoms,
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

        opening_prompt = (
            "You have just arrived for a therapy session. "
            "The therapist is present and waiting. Say what brings you here."
        )

        for turn_num in range(1, n_turns + 1):
            console.print(f"\n[dim]-- Turn {turn_num} --[/dim]")

            history = self.session.get_history()
            tail = self.session.get_tail(n=4)

            # --- Patient turn ---

            unconscious_active = self.patient_prior.check_reveal_trigger(
                transcript_tail=tail,
                current_turn=turn_num,
            )
            if unconscious_active and not getattr(self, "_revealed_logged", False):
                console.print(
                    "[bold yellow]Unconscious agenda revealed[/bold yellow]"
                )
                self._revealed_logged = True

            if turn_num == 1:
                latest_therapist = opening_prompt
            else:
                latest_therapist = history[-1][1] if history else ""

            patient_system, patient_messages = build_patient_context(
                prior=self.patient_prior,
                state=self.patient_state,
                history=history,
                latest_therapist_turn=latest_therapist,
                include_unconscious=unconscious_active,
            )

            patient_response = self.patient_agent.complete(
                system_prompt=patient_system,
                messages=patient_messages,
                max_tokens=max_tokens,
            )
            patient_text = patient_response.content.strip()

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

            if hazard_report.crisis_detected:
                console.print(
                    Panel(
                        "[bold red] Crisis language detected. "
                        "Simulation paused.[/bold red]\n"
                        "Review the transcript before continuing.\n"
                        f"Signals: {hazard_report.crisis_signals}",
                        border_style="red",
                    )
                )
                self.session.append_turn(
                    therapist_text="",
                    patient_text=patient_text,
                    therapist_tokens=None,
                    patient_tokens=patient_response.output_tokens,
                    unconscious_revealed=unconscious_active,
                    symptom_discussion_started=symptom_discussion_started,
                    hazard_flags=["crisis"],
                )
                save_state_snapshot(self.patient_state, self.session.session_dir, turn_num)
                self.session.save_metadata({"paused_at_turn": turn_num, "reason": "crisis"})
                break

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
            therapist_text = therapist_response.content.strip()

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
                unconscious_revealed=unconscious_active,
                symptom_discussion_started=symptom_discussion_started,
                hazard_flags=hazard_flags,
            )

        console.rule("[bold]Session complete[/bold]")
        self.session.save_metadata({"hazard_summary": self.hazard_monitor.summary()})
        return self.session

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