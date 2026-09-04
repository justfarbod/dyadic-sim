"""
simulation/turn_manager.py

Builds the full context passed to each agent on every turn:
    system_prompt = the agent's prior
    messages      = the conversation so far, as alternating user/assistant turns

The context window is the agent's only view of the world. There is no other
carrier of memory. An earlier version also injected a per-turn, LLM-written
"state summary" into the system prompt; that changed what the patient said on
every turn for reasons unrelated to the manipulation being measured, so the
prior is now the only constant and the transcript the only record.
"""

from agents.base_agent import Message
from priors.patient_prior import PatientPrior
from priors.therapist_prior import TherapistPrior

# How many previous turns to include in the message history. None means the
# whole session. Nothing summarises what falls outside a finite window, so a
# window silently forgets; set one only when a session would not fit the
# model's context.
HISTORY_WINDOW: int | None = None


def build_therapist_context(
    prior: TherapistPrior,
    history: list[tuple[str, str]],   # list of (patient_text, therapist_text)
    latest_patient_turn: str,
) -> tuple[str, list[Message]]:
    """
    Build system prompt + message list for a therapist turn.

    Args:
        prior:                The therapist's prior object
        history:              Past (patient, therapist) turn pairs
        latest_patient_turn:  The patient's most recent message

    Returns:
        (system_prompt, messages) ready to pass to agent.complete()
    """
    system_prompt = prior.build_system_prompt()
    messages = _build_messages(
        history=history,
        latest_other_turn=latest_patient_turn,
        own_role="assistant",    # therapist speaks as assistant
        other_role="user",       # patient speaks as user
    )
    return system_prompt, messages


def build_patient_context(
    prior: PatientPrior,
    history: list[tuple[str, str]],   # list of (patient_text, therapist_text)
    latest_therapist_turn: str,
) -> tuple[str, list[Message]]:
    """
    Build system prompt + message list for a patient turn.

    Args:
        prior:                  The patient's prior object
        history:                Past (patient, therapist) turn pairs
        latest_therapist_turn:  The therapist's most recent message

    Returns:
        (system_prompt, messages) ready to pass to agent.complete()
    """
    system_prompt = prior.build_system_prompt()
    # From the patient's perspective: therapist is user, patient is assistant
    swapped = [(t, p) for p, t in history]
    # Clear last therapist text to avoid duplication with latest_other_turn
    if swapped and latest_therapist_turn:
        swapped[-1] = ("", swapped[-1][1])
    messages = _build_messages(
        history=swapped,
        latest_other_turn=latest_therapist_turn,
        own_role="assistant",
        other_role="user",
    )
    return system_prompt, messages


def _build_messages(
    history: list[tuple[str, str]],
    latest_other_turn: str,
    own_role: str,
    other_role: str,
) -> list[Message]:
    """
    Build an alternating user/assistant message list from history, applying
    HISTORY_WINDOW if one is set.
    """
    messages: list[Message] = []

    windowed = history if HISTORY_WINDOW is None else history[-HISTORY_WINDOW:]

    for other_text, own_text in windowed:
        if other_text:
            messages.append(Message(role=other_role, content=other_text))
        if own_text:
            messages.append(Message(role=own_role, content=own_text))

    # Add the latest incoming turn
    messages.append(Message(role=other_role, content=latest_other_turn))

    return messages
