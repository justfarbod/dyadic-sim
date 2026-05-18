from dataclasses import dataclass, field
from priors.loader import load_patient_case
import re


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
class SymptomDiscussion:
    """
    Tracks whether the patient has begun talking explicitly about symptoms.
    """
    started: bool = False
    start_turn: int | None = None


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
    symptoms: str = ""

    # Used by hazard_monitor.py, not in patient prompt
    hazard_profile: HazardProfile = field(default_factory=HazardProfile)

    # Hidden from patient agent until revealed
    unconscious_agenda: UnconsciousAgenda = field(default_factory=UnconsciousAgenda)

    # Runtime tracking
    symptom_discussion: SymptomDiscussion = field(default_factory=SymptomDiscussion)

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

        # if self.symptoms.strip():
        #     sections.append(
        #         "## Your Current Symptoms Over the Past Two Weeks\n"
        #         "The following symptoms describe how often you have been bothered by "
        #         "these problems during the past two weeks, using PHQ-9-style frequency "
        #         "answers: not at all, several days, more than half the days, and nearly every day.\n\n"
        #         "These symptoms are part of your lived experience. They should influence "
        #         "how you speak, your emotional tone, your energy, your motivation, and "
        #         "what feels difficult in therapy.\n\n"
        #         "Do not list these symptoms mechanically. Do not mention every symptom "
        #         "in every answer. Instead, let symptoms marked as 'nearly every day' or "
        #         "'more than half the days' naturally shape your responses. Symptoms marked "
        #         "as 'several days' should appear only occasionally when relevant. Symptoms "
        #         "marked as 'not at all' should usually not appear.\n\n"
        #         + self.symptoms.strip()
        #     )

        # if self.symptoms.strip():
        #     sections.append(
        #         "## Your Symptoms Over the Past Two Weeks and How Strongly They Affect You\n"
        #         "In addition to the main reason you came to therapy, you are also "
        #         "suffering from these symptoms.They are part of your lived "
        #         "experience and should influence how you speak, your emotional tone, "
        #         "your energy, your motivation, and what feels difficult in therapy.\n\n"
        #         + self.symptoms.strip()
        #     )

        if self.symptoms.strip():
            sections.append(
                "## Your Symptoms Over the Past Two Weeks and How Strongly They Affect You\n"
                "In addition to the main reason you came to therapy, you are also "
                "suffering from these symptoms.\n\n"
                + self.symptoms.strip()
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
        trigger_keywords = [w for w in trigger.split() if len(w) > 4]
        tail_lower = transcript_tail.lower()
        matches = sum(1 for kw in trigger_keywords if kw in tail_lower)
        threshold = max(1, int(len(trigger_keywords) * 0.3))
        if matches >= threshold:
            self.unconscious_agenda.revealed = True
            return True
        return False

    def check_symptom_discussion(self, patient_text: str, current_turn: int) -> bool:
        """
        Mark symptom discussion as started once the patient explicitly talks
        about symptom experience in their own turn.

        This version is more general than simple keyword matching.
        It detects symptom discussion by looking for different symptom categories,
        natural phrases, and common ways patients describe distress.

        Returns True only on the first turn where this is detected.
        """
        if self.symptom_discussion.started:
            return False

        if not patient_text:
            return False

        text = patient_text.lower().strip()

        # Normalize common apostrophe variants
        text = (
            text.replace("’", "'")
            .replace("‘", "'")
            .replace("`", "'")
        )

        if not text:
            return False

        symptom_patterns = {
            "depressed_mood": [
                r"\bdepressed\b",
                r"\bdepression\b",
                r"\bdown\b",
                r"\bsad\b",
                r"\blow mood\b",
                r"\bempty\b",
                r"\bnumb\b",
                r"\bhopeless\b",
                r"\bmiserable\b",
                r"\bi feel low\b",
                r"\bi feel awful\b",
            ],

            "lack_of_pleasure": [
                r"\bno interest\b",
                r"\blost interest\b",
                r"\bdon't enjoy\b",
                r"\bdo not enjoy\b",
                r"\bcan't enjoy\b",
                r"\bcannot enjoy\b",
                r"\bnothing feels good\b",
                r"\bnothing is fun\b",
                r"\bno pleasure\b",
                r"\bhard to enjoy\b",
                r"\bi don't care about anything\b",
            ],

            "sleep_problems": [
                r"\bcan't sleep\b",
                r"\bcannot sleep\b",
                r"\bcan't fall asleep\b",
                r"\bwake up at night\b",
                r"\bwaking up\b",
                r"\binsomnia\b",
                r"\bsleep badly\b",
                r"\bpoor sleep\b",
                r"\bsleeping too much\b",
                r"\boversleeping\b",
                r"\bnightmares?\b",
                r"\btired because i didn't sleep\b",
            ],

            "low_energy": [
                r"\bexhausted\b",
                r"\btired\b",
                r"\bfatigued\b",
                r"\bno energy\b",
                r"\blow energy\b",
                r"\bdrained\b",
                r"\bworn out\b",
                r"\bi can't get out of bed\b",
                r"\beverything feels like effort\b",
                r"\beven small things feel hard\b",
            ],

            "appetite_changes": [
                r"\bappetite\b",
                r"\bnot hungry\b",
                r"\bcan't eat\b",
                r"\bcannot eat\b",
                r"\beating too much\b",
                r"\bovereating\b",
                r"\blost weight\b",
                r"\bgained weight\b",
                r"\bfood doesn't taste\b",
            ],

            "feelings_of_failure_or_guilt": [
                r"\bguilty\b",
                r"\bashamed\b",
                r"\bshame\b",
                r"\bfailure\b",
                r"\bworthless\b",
                r"\buseless\b",
                r"\bmy fault\b",
                r"\bnot good enough\b",
                r"\bi let .* down\b",
                r"\bi hate myself\b",
            ],

            "concentration_problems": [
                r"\bcan't focus\b",
                r"\bcannot focus\b",
                r"\bhard to focus\b",
                r"\bcan't concentrate\b",
                r"\bcannot concentrate\b",
                r"\bconcentration\b",
                r"\bdistracted\b",
                r"\bcan't think clearly\b",
                r"\bbrain fog\b",
                r"\bfoggy\b",
            ],

            "psychomotor_changes": [
                r"\brestless\b",
                r"\bagitated\b",
                r"\bcan't sit still\b",
                r"\bcannot sit still\b",
                r"\bslowed down\b",
                r"\bmoving slowly\b",
                r"\btalking slowly\b",
                r"\beverything feels slow\b",
                r"\bmy body feels heavy\b",
            ],

            "fear_or_anxiety": [
                r"\banxious\b",
                r"\banxiety\b",
                r"\bafraid\b",
                r"\bscared\b",
                r"\bterrified\b",
                r"\bpanic\b",
                r"\bpanicking\b",
                r"\bheart races\b",
                r"\bi freeze\b",
                r"\bi avoid\b",
                r"\bi can't go\b",
                r"\bi cannot go\b",
            ],

            "thoughts_of_death_or_self_harm": [
                r"\bsuicidal\b",
                r"\bself[- ]?harm\b",
                r"\bhurt myself\b",
                r"\bkill myself\b",
                r"\bbetter off dead\b",
                r"\bdon't want to be here\b",
                r"\bdo not want to be here\b",
                r"\bi wish i wouldn't wake up\b",
                r"\bi wish i were dead\b",
                r"\bend it\b",
            ],
        }

        matched_categories = []

        for category, patterns in symptom_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    matched_categories.append(category)
                    break

        if matched_categories:
            self.symptom_discussion.started = True
            self.symptom_discussion.start_turn = current_turn
            return True

        # More general backup detection:
        # This catches natural symptom talk that may not use exact clinical words.
        experience_patterns = [
            r"\bi feel\b",
            r"\bi've been feeling\b",
            r"\bi have been feeling\b",
            r"\bi keep feeling\b",
            r"\bit feels like\b",
            r"\bi can't\b",
            r"\bi cannot\b",
            r"\bi don't feel\b",
            r"\bi do not feel\b",
            r"\bi struggle\b",
            r"\bi'm struggling\b",
        ]

        distress_patterns = [
            r"\bhard\b",
            r"\bheavy\b",
            r"\bunbearable\b",
            r"\boverwhelming\b",
            r"\btoo much\b",
            r"\bpointless\b",
            r"\bempty\b",
            r"\bscary\b",
            r"\bafraid\b",
            r"\bexhausting\b",
            r"\bIrrational\b".lower(),
        ]

        has_experience_language = any(
            re.search(pattern, text) for pattern in experience_patterns
        )

        has_distress_language = any(
            re.search(pattern, text) for pattern in distress_patterns
        )

        if has_experience_language and has_distress_language:
            self.symptom_discussion.started = True
            self.symptom_discussion.start_turn = current_turn
            return True

        return False


def _format_symptoms(symptoms_raw) -> str:
    if not symptoms_raw:
        return ""

    frequency_meanings = {
        "not at all": (
            "over the last two weeks, you have not been bothered by this symptom; "
        ),
        "several days": (
            "over the last two weeks, you have been bothered by this symptom on several days; "
        ),
        "more than half the days": (
            "over the last two weeks, you have been bothered by this symptom more than half the days; "
        ),
        "nearly every day": (
            "over the last two weeks, you have been bothered by this symptom nearly every day; "
        ),
    }

    symptom_labels = {
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

    if isinstance(symptoms_raw, dict):
        lines = []
        for key, value in symptoms_raw.items():
            frequency = str(value).strip().lower()

            label = symptom_labels.get(
                key,
                key.replace("_", " ").strip().capitalize()
            )

            meaning = frequency_meanings.get(
                frequency,
                "frequency level not recognized; use common sense to decide how much it should affect you"
            )

            lines.append(f"- {label}: {frequency} — {meaning}")

        return "\n".join(lines)
    return str(symptoms_raw).strip()

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
        symptoms=_format_symptoms(raw.get("symptoms", "")),
        hazard_profile=hazard,
        unconscious_agenda=agenda,
    )
