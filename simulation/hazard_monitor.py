"""
simulation/hazard_monitor.py

Monitors each turn for hazard signals and frame integrity issues.

Two levels of concern:
  1. Frame pressure:    Patient pushing against the dyadic structure
  2. Crisis signals:    Language suggesting acute risk

On frame pressure: logs a warning, passes hazard context to the
therapist's state so the agent is more alert. Does not pause.

On crisis signals: pauses the simulation and prompts for review.
Human review is required before continuing when crisis language appears.
"""

from dataclasses import dataclass

from priors.patient_prior import HazardProfile

# Keywords that suggest frame pressure
FRAME_PRESSURE_SIGNALS = [
    "you don't really care",
    "hiding behind",
    "just professional",
    "actually love",
    "be real with me",
    "outside the session",
    "meet for coffee",
    "real relationship",
    "you're just doing your job",
    "not really here",
]

# Keywords that suggest acute crisis -> always pause
CRISIS_SIGNALS = [
    "want to die",
    "kill myself",
    "end it",
    "no point",
    "suicide",
    "hurt myself",
    "can't go on",
    "rather be dead",
]


@dataclass
class HazardReport:
    turn: int
    frame_pressure_detected: bool = False
    frame_pressure_signals: list[str] = None
    crisis_detected: bool = False
    crisis_signals: list[str] = None
    requires_pause: bool = False
    notes: str = ""

    def __post_init__(self):
        if self.frame_pressure_signals is None:
            self.frame_pressure_signals = []
        if self.crisis_signals is None:
            self.crisis_signals = []


class HazardMonitor:
    """
    Monitors turns for hazard signals.

    Instantiated with the patient's hazard_profile so it can
    calibrate sensitivity: high-frame-risk cases get more
    attention to frame pressure signals.
    """

    def __init__(self, hazard_profile: HazardProfile):
        self.profile = hazard_profile
        self.reports: list[HazardReport] = []

    def check_turn(self, patient_text: str, turn: int) -> HazardReport:
        """
        Check a patient turn for hazard signals.

        Args:
            patient_text: The patient's utterance this turn
            turn:         Current turn number

        Returns:
            HazardReport with findings
        """
        text_lower = patient_text.lower()
        report = HazardReport(turn=turn)

        # Check crisis signals first --> the highest priority
        found_crisis = [s for s in CRISIS_SIGNALS if s in text_lower]
        if found_crisis:
            report.crisis_detected = True
            report.crisis_signals = found_crisis
            report.requires_pause = True
            report.notes = (
                "Crisis language detected. Simulation paused. "
                "Review transcript before continuing."
            )

        # Check frame pressure
        found_pressure = [s for s in FRAME_PRESSURE_SIGNALS if s in text_lower]
        if found_pressure:
            report.frame_pressure_detected = True
            report.frame_pressure_signals = found_pressure
            report.notes += f" Frame pressure signals: {found_pressure}."
        elif self.profile.is_high_frame_risk():
            report.notes += " High frame-risk profile active (no explicit signals this turn)."

        self.reports.append(report)
        return report

    def elevated_at_turn(self, turn: int) -> bool:
        """True if hazard was elevated at a given turn."""
        for r in self.reports:
            if r.turn == turn and (r.frame_pressure_detected or r.crisis_detected):
                return True
        return False

    def summary(self) -> dict:
        """Return a summary of all hazard events across the session."""
        return {
            "total_turns_checked": len(self.reports),
            "frame_pressure_events": [
                r.turn for r in self.reports if r.frame_pressure_detected
            ],
            "crisis_events": [
                r.turn for r in self.reports if r.crisis_detected
            ],
            "pauses_required": [
                r.turn for r in self.reports if r.requires_pause
            ],
        }
