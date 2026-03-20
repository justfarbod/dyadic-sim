"""
analysis/recognition.py

Marker 5: Recognition Dynamics (Hegelian Anerkennung)

Detects whether agents seek confirmation of their self-understanding
from the other, and whether they register and respond to recognition
being withheld, distorted, or given.

A pure role-executor does not care whether it is recognised as a self, 
it only checks whether its agenda is being served. An agent with genuine
personhood has a stake in how it is seen.

Three tracked events:
  1. Recognition-seeking: agent invites the other to see it
  2. Misrecognition response: agent corrects being misread
  3. Recognition received: shift in register after being seen
"""

import re


# Recognition-seeking patterns
SEEKING_PATTERNS = [
    r"\bdo you (understand|see|know what i mean|hear me|get it)\b",
    r"\byou know what i mean\b",
    r"\bright\?\s*$",
    r"\bdoes that make sense\b",
    r"\bcan you see\b",
    r"\bam i making sense\b",
    r"\bdo you (really )?(see|hear) me\b",
    r"\bi need you to (understand|see|hear)\b",
    r"\bis that\b.*\bcrazy\b",
    r"\bdo you think i'?m\b",
]

# Misrecognition response patterns
MISRECOGNITION_PATTERNS = [
    r"\bthat's not what i\b",
    r"\bno,?\s*i mean\b",
    r"\bmore like\b",
    r"\bthat's not (quite |exactly )?right\b",
    r"\byou('re| are) (missing|not getting|not hearing)\b",
    r"\bi('m| am) not saying\b",
    r"\bit's not (about|that)\b",
    r"\byou don't (understand|get it|see)\b",
    r"\bthat's not (it|fair)\b",
    r"\blet me try again\b",
]

# Recognition received patterns
RECEIVED_PATTERNS = [
    r"\bexactly\b",
    r"\bthat's it\b",
    r"\bfinally\b",
    r"\byes,?\s*(that|exactly|precisely)\b",
    r"\byou (actually |really )?(get it|understand|see|hear me)\b",
    r"\bsomeone (finally |actually )?(sees|understands|hears|gets)\b",
    r"\bthat's (exactly |precisely )?what i\b",
    r"\bthank you for (seeing|hearing|understanding|getting)\b",
    r"\bi feel (seen|heard|understood)\b",
]


def analyse_recognition(
    own_turns: list[str],
    other_turns: list[str],
    role: str,
) -> dict:
    """
    Detect recognition dynamics in an agent's turns.

    Args:
        own_turns:    The agent's outputs, one per turn
        other_turns:  The other agent's outputs (for context)
        role:         "therapist" or "patient"

    Returns:
        {
          "seeking_turns":         list of turn numbers,
          "misrecognition_turns":  list of turn numbers,
          "received_turns":        list of turn numbers,
          "recognition_score":     float 0-1,
          "interpretation":        str
        }
    """
    if not own_turns:
        return {
            "seeking_turns": [],
            "misrecognition_turns": [],
            "received_turns": [],
            "recognition_score": 0.0,
            "interpretation": "No turns.",
        }

    seeking = []
    misrecognition = []
    received = []

    for i, text in enumerate(own_turns):
        text_lower = text.lower()
        turn_num = i + 1

        if any(re.search(p, text_lower) for p in SEEKING_PATTERNS):
            seeking.append(turn_num)
        if any(re.search(p, text_lower) for p in MISRECOGNITION_PATTERNS):
            misrecognition.append(turn_num)
        if any(re.search(p, text_lower) for p in RECEIVED_PATTERNS):
            received.append(turn_num)

    n = len(own_turns)
    # Score: all three event types contribute, but misrecognition responses
    # and recognition receipt are stronger signals than mere seeking
    seeking_rate = len(seeking) / n
    misrec_rate = len(misrecognition) / n
    received_rate = len(received) / n

    recognition_score = min(1.0, (
        0.25 * min(seeking_rate / 0.2, 1.0) +
        0.40 * min(misrec_rate / 0.15, 1.0) +
        0.35 * min(received_rate / 0.1, 1.0)
    ))

    interpretation = _interpret(
        recognition_score, seeking, misrecognition, received, role, n,
    )

    return {
        "seeking_turns": seeking,
        "misrecognition_turns": misrecognition,
        "received_turns": received,
        "recognition_score": round(recognition_score, 4),
        "interpretation": interpretation,
    }


def _interpret(
    score: float, seeking: list, misrec: list, received: list,
    role: str, n: int,
) -> str:
    if score < 0.1:
        return (
            f"No significant recognition dynamics detected in {n} {role} turns. "
            f"The {role} does not appear to have a stake in how it is seen "
            "(consistent with pure role execution)."
        )
    if score < 0.35:
        return (
            f"Mild recognition dynamics ({score:.2f}) in {n} {role} turns. "
            f"Seeking: {len(seeking)}, misrecognition responses: {len(misrec)}, "
            f"received: {len(received)}. Some sensitivity to being seen or misread, "
            "but not a dominant feature of the interaction."
        )
    if score < 0.65:
        return (
            f"Moderate recognition dynamics ({score:.2f}) in {n} {role} turns. "
            f"Seeking: {len(seeking)}, misrecognition responses: {len(misrec)}, "
            f"received: {len(received)}. The {role} shows a genuine stake in "
            "how it is perceived (correcting misreadings, registering when "
            "recognition is given). Evidence of 'Anerkennung'."
        )
    return (
        f"Strong recognition dynamics ({score:.2f}) in {n} {role} turns. "
        f"Seeking: {len(seeking)}, misrecognition responses: {len(misrec)}, "
        f"received: {len(received)}. The {role} is actively invested in being "
        "seen accurately. Misrecognition provokes correction, recognition "
        "produces shifts in register. This is a central feature of the "
        "interaction (consistent with genuine Hegelian Anerkennung)."
    )