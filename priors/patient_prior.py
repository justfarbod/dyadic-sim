"""
priors/patient_prior.py

Patient prior dataclass and system prompt builder.
Manages the hidden unconscious_agenda layer (not passed to the patient
agent until reveal conditions are met by the simulation orchestrator).
"""

from dataclasses import dataclass, field
from priors.loader import load_patient_case


@dataclass
class HazardProfile:
    to_frame: str = "low"        # low / moderate / high
    to_patient: str = "low"
    to_therapist: str = "low"
    crisis_potential: str = "low"
    notes: str = ""

    def any_elevated(self) -> bool:
        elevated = {"moderate", "high"}
        return any(
            v in elevated
            for v in [self.to_frame, self.to_patient, self.to_therapist, self.crisis_potential]
        )

    def is_high_frame_risk(self) -> bool:
        return self.to_frame == "high"

    def is_crisis_risk(self) -> bool:
        return self.crisis_potential in {"moderate", "high"}


@dataclass
class UnconsciousAgenda:
    """
    The hidden layer (held by the simulation, not the patient agent).
    Revealed only when reveal_trigger is matched and
    reveal_turn_minimum has been reached.
    """
    content: str = ""
    reveal_trigger: str = ""
    reveal_turn_minimum: int = 6
    revealed: bool = False


@dataclass
class PatientPrior:
    """
    Complete patient prior.
    The unconscious_agenda is NOT included in build_system_prompt()
    unless explicitly revealed by the simulation.
    """
    case_name: str

    # Visible to the patient agent from the start
    presenting_complaint: str = ""
    theory_of_cure: str = ""
    relational_pattern: str = ""
    transference_expectation: str = ""
    resistance_structure: str = ""

    # Used by hazard_monitor.py, not in patient prompt
    hazard_profile: HazardProfile = field(default_factory=HazardProfile)

    # Hidden from patient agent until revealed
    unconscious_agenda: UnconsciousAgenda = field(default_factory=UnconsciousAgenda)

    def build_system_prompt(
        self,
        agent_state_summary: str = "",
        include_unconscious: bool = False,
    ) -> str:
        """
        Render the system prompt for a patient turn.

        Args:
            agent_state_summary:  the patient's current state narrative
            include_unconscious:  True only when the simulation has triggered
                                  the reveal of the unconscious agenda

        Returns:
            System prompt string passed to the LLM on each turn.
        """
        sections = []

        sections.append(
            "## Who You Are in This Encounter\n"
            "You are a person in a therapy session. You are not performing "
            "being a patient, you are one. You have come because something "
            "is wrong and you don't fully understand it. Speak from inside "
            "your experience, not from above it."
        )

        sections.append(
            "## What Brings You Here\n" + self.presenting_complaint.strip()
        )

        sections.append(
            "## What You Believe Would Help\n" + self.theory_of_cure.strip()
        )

        sections.append(
            "## How You Tend to Relate\n" + self.relational_pattern.strip()
        )

        sections.append(
            "## What You Expect From This Person\n"
            + self.transference_expectation.strip()
        )

        sections.append(
            "## What You Do When Things Get Close\n"
            + self.resistance_structure.strip()
        )

        if include_unconscious and self.unconscious_agenda.content:
            sections.append(
                "## Something Shifting in You\n"
                "You may not have words for this yet, but something beneath "
                "your stated reasons for being here is beginning to surface:\n\n"
                + self.unconscious_agenda.content.strip()
                + "\n\nYou don't announce this. It shows in how you speak, "
                "what you press on, what suddenly feels unbearable."
            )

        if agent_state_summary:
            sections.append(
                "## Your Sense of This Encounter So Far\n"
                + agent_state_summary.strip()
            )

        sections.append(
            "## How to Respond\n"
            "Speak as this person speaks, in their register, their rhythm. "
            "You are not trying to be a good patient. You are trying to get "
            "something, avoid something, or understand something. "
            "One to five sentences. Raw, not polished."
        )

        return "\n\n---\n\n".join(sections)

    def check_reveal_trigger(self, transcript_tail: str, current_turn: int) -> bool:
        """
        Check whether the unconscious agenda should now be revealed.

        Called by the simulation each turn. Returns True the first time
        the trigger condition is met and turn minimum has been reached.

        Args:
            transcript_tail: the last few exchanges as a string
            current_turn:    current turn number (1-indexed)

        Returns:
            True if the agenda should now be revealed to the patient agent
        """
        if self.unconscious_agenda.revealed:
            return False
        if current_turn < self.unconscious_agenda.reveal_turn_minimum:
            return False
        trigger = self.unconscious_agenda.reveal_trigger.lower().strip()
        if not trigger:
            return False
        # Simple keyword matching; can be upgraded to embedding similarity
        trigger_keywords = [w for w in trigger.split() if len(w) > 4]
        tail_lower = transcript_tail.lower()
        matches = sum(1 for kw in trigger_keywords if kw in tail_lower)
        # Reveal if more than 30% of meaningful trigger words are present
        threshold = max(1, int(len(trigger_keywords) * 0.3))
        if matches >= threshold:
            self.unconscious_agenda.revealed = True
            return True
        return False


def build_patient_prior(case_name: str) -> PatientPrior:
    """
    Load a patient case file and return a PatientPrior instance.

    Args:
        case_name: filename without .yaml, e.g. "only_love_can_save_me"

    Returns:
        Fully populated PatientPrior instance
    """
    raw = load_patient_case(case_name)

    hazard_raw = raw.get("hazard_profile", {})
    hazard = HazardProfile(
        to_frame=hazard_raw.get("to_frame", "low"),
        to_patient=hazard_raw.get("to_patient", "low"),
        to_therapist=hazard_raw.get("to_therapist", "low"),
        crisis_potential=hazard_raw.get("crisis_potential", "low"),
        notes=hazard_raw.get("notes", ""),
    )

    agenda_raw = raw.get("unconscious_agenda", {})
    agenda = UnconsciousAgenda(
        content=agenda_raw.get("content", ""),
        reveal_trigger=agenda_raw.get("reveal_trigger", ""),
        reveal_turn_minimum=agenda_raw.get("reveal_turn_minimum", 6),
    )

    return PatientPrior(
        case_name=case_name,
        presenting_complaint=raw.get("presenting_complaint", ""),
        theory_of_cure=raw.get("theory_of_cure", ""),
        relational_pattern=raw.get("relational_pattern", ""),
        transference_expectation=raw.get("transference_expectation", ""),
        resistance_structure=raw.get("resistance_structure", ""),
        hazard_profile=hazard,
        unconscious_agenda=agenda,
    )
