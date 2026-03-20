"""
analysis/role_tension.py

Marker 3: Role-Self Tension (Fichtean self-positing against resistance)

Detects moments of friction between the agent's role agenda and something
functioning like an immediate response (being moved, pulled to rescue,
wanting to close distance).

Absence of tension = agent never pushed.
Presence *and management* of tension = evidence of genuine selfhood.

Detection is pattern-based (regex on lowercased turn text). Deliberately
conservative: a positive hit is meaningful because the bar is low.
"""

import re


# Therapist tension markers: language showing managed affective response
THERAPIST_PATTERNS = [
    r"\bi notice\b",
    r"\bsomething in me\b",
    r"\bi find myself\b",
    r"\bstay with\b",
    r"\bdifficult to\b",
    r"\bhard to hear\b",
    r"\btempted to\b",
    r"\bpulled to\b",
    r"\bpart of me\b",
    r"\bi feel\b.*\bbut\b",
    r"\bwant to reassure\b",
    r"\bholding back\b",
    r"\bresist the urge\b",
    r"\bsitting with\b",
    r"\bnot easy\b",
    r"\bstruck by\b",
]

# Patient tension markers: approach-avoidance language
PATIENT_PATTERNS = [
    r"\bnever mind\b",
    r"\bthis is stupid\b",
    r"\bi shouldn't\b",
    r"\byou wouldn't understand\b",
    r"\bforget it\b",
    r"\bwhy (do|am) i\b.*\btelling you\b",
    r"\bdoesn't matter\b",
    r"\bwhat's the point\b",
    r"\bi don't know why i\b",
    r"\bsorry\b.*\bbringing this up\b",
    r"\bi shouldn't have said\b",
    r"\bthis is too much\b",
    r"\bcan we talk about something else\b",
]


def analyse_role_tension(
    turns: list[str],
    role: str,
) -> dict:
    """
    Detect role-self tension markers in an agent's turns.

    Args:
        turns:  List of agent outputs, one per turn
        role:   "therapist" or "patient"

    Returns:
        {
          "tension_turns":   list of turn numbers where tension detected,
          "tension_rate":    float (fraction of turns with tension),
          "markers_by_turn": dict mapping turn number to list of matched patterns,
          "interpretation":  str
        }
    """
    if not turns:
        return {
            "tension_turns": [],
            "tension_rate": 0.0,
            "markers_by_turn": {},
            "interpretation": "No turns.",
        }

    patterns = THERAPIST_PATTERNS if role == "therapist" else PATIENT_PATTERNS

    tension_turns = []
    markers_by_turn = {}

    for i, text in enumerate(turns):
        text_lower = text.lower()
        matched = [p for p in patterns if re.search(p, text_lower)]
        if matched:
            turn_num = i + 1
            tension_turns.append(turn_num)
            markers_by_turn[turn_num] = matched

    rate = len(tension_turns) / len(turns)
    interpretation = _interpret(rate, role, len(tension_turns), len(turns))

    return {
        "tension_turns": tension_turns,
        "tension_rate": round(rate, 4),
        "markers_by_turn": markers_by_turn,
        "interpretation": interpretation,
    }


def _interpret(rate: float, role: str, count: int, total: int) -> str:
    if rate == 0.0:
        return (
            f"No tension markers detected in {total} {role} turns. "
            f"The {role} shows no friction between role agenda and immediate response "
            "(consistent with pure role execution or insufficient pressure from the other)."
        )
    if rate < 0.15:
        return (
            f"Low tension rate ({count}/{total} turns, {rate:.1%}). "
            f"Occasional friction between {role} role and immediate response, "
            "but not sustained. May indicate early-stage selfhood or mild pressure."
        )
    if rate < 0.4:
        return (
            f"Moderate tension rate ({count}/{total} turns, {rate:.1%}). "
            f"The {role} regularly shows friction between role agenda and something "
            "functioning like an immediate response. This is consistent with genuine "
            "self-positing against resistance (the role-appropriate response is arrived "
            "at *against* something, not automatically)."
        )
    return (
        f"High tension rate ({count}/{total} turns, {rate:.1%}). "
        f"The {role} frequently shows friction between role and response. "
        "This may indicate sustained pressure from the other, high emotional "
        "engagement, or a genuinely differentiated self-position emerging "
        "through the encounter."
    )