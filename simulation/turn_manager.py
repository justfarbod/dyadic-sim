"""
simulation/turn_manager.py

Builds the full context passed to each agent on every turn:
    system_prompt = prior + current state summary
    messages      = conversation history (last N turns)

The context window is the agent's only view of the world.
How we build it determines what the agent can 'remember' and 'be'.
"""

from agents.base_agent import Message
from memory.state import AgentState
from priors.patient_prior import PatientPrior
from priors.therapist_prior import TherapistPrior

# How many previous turns to include in the message history.
# Beyond this, the agent relies on its state summary (compressed memory).
# Increase for richer context at the cost of longer prompts.
HISTORY_WINDOW = 10


def build_therapist_context(
    prior: TherapistPrior,
    state: AgentState,
    history: list[tuple[str, str]],   # list of (patient_text, therapist_text)
    latest_patient_turn: str,
) -> tuple[str, list[Message]]:
    """
    Build system prompt + message list for a therapist turn.

    Args:
        prior:                The therapist's prior object
        state:                The therapist's current state
        history:              Past (patient, therapist) turn pairs
        latest_patient_turn:  The patient's most recent message

    Returns:
        (system_prompt, messages) ready to pass to agent.complete()
    """
    system_prompt = prior.build_system_prompt(
        agent_state_summary=state.to_summary()
    )
    messages = _build_messages(
        history=history,
        latest_other_turn=latest_patient_turn,
        own_role="assistant",    # therapist speaks as assistant
        other_role="user",       # patient speaks as user
    )
    return system_prompt, messages


def build_patient_context(
    prior: PatientPrior,
    state: AgentState,
    history: list[tuple[str, str]],   # list of (patient_text, therapist_text)
    latest_therapist_turn: str,
    include_unconscious: bool = False,
) -> tuple[str, list[Message]]:
    """
    Build system prompt + message list for a patient turn.

    Args:
        prior:                  The patient's prior object
        state:                  The patient's current state
        history:                Past (patient, therapist) turn pairs
        latest_therapist_turn:  The therapist's most recent message
        include_unconscious:    Whether to reveal the unconscious agenda

    Returns:
        (system_prompt, messages) ready to pass to agent.complete()
    """
    system_prompt = prior.build_system_prompt(
        agent_state_summary=state.to_summary(),
        include_unconscious=include_unconscious,
    )
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
    Build alternating user/assistant message list from history.

    Applies the HISTORY_WINDOW limit (older turns are dropped),
    leaving the agent to rely on its state summary for earlier context.
    """
    messages: list[Message] = []

    # Apply window
    windowed = history[-HISTORY_WINDOW:] if len(history) > HISTORY_WINDOW else history

    for other_text, own_text in windowed:
        if other_text:
            messages.append(Message(role=other_role, content=other_text))
        if own_text:
            messages.append(Message(role=own_role, content=own_text))

    # Add the latest incoming turn
    messages.append(Message(role=other_role, content=latest_other_turn))

    return messages
