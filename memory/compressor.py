"""
memory/compressor.py

Updates an agent's state after each turn.

The compressor asks the agent to reflect on the turn just completed
and update its own narrative self, relational history, and flag any
shifts or key moments.

The compression works like "memory". What the agent retains, drops, and
re-frames in this step is primary data for further analysis
(e.g., emerging personhood). Here we treat distortions as a sort of signal.
"""

import json
from agents.base_agent import BaseAgent, Message
from memory.state import AgentState


COMPRESSOR_PROMPT = """You just completed a turn in a therapeutic dyadic exchange.
Your role was: {role}

Here is what was just said (the last exchange):

THEIR TURN:
{other_turn}

YOUR TURN:
{own_turn}

Your current state before this turn was:
{current_state}

---

Please update your state by responding with a JSON object containing these fields:

{{
  "narrative_self": "Who you now understand yourself to be in this encounter. \
One to three sentences. Update if something shifted, otherwise keep what was true.",

  "relational_history": "What you now notice about this specific person. \
What patterns, what needs, what they seem to be doing in this relationship. \
Update with anything new this turn revealed.",

  "shift_occurred": true or false,

  "shift_description": "If a shift occurred: one sentence describing what changed \
in your understanding. Empty string if no shift.",

  "shift_triggered_by": "If a shift occurred: what specifically triggered it \
(a word, a silence), a move the other person made. Empty string if no shift.",

  "key_moment": true or false,

  "key_moment_summary": "If this was a key moment: one sentence on what happened. \
Empty string if not a key moment.",

  "key_moment_significance": "If this was a key moment: why it mattered. \
Empty string if not a key moment."
}}

Respond ONLY with the JSON object. No preamble, no explanation.
"""


def compress_turn(
    agent: BaseAgent,
    state: AgentState,
    own_turn: str,
    other_turn: str,
    turn_number: int,
) -> AgentState:
    """
    Ask the agent to reflect and update its own state.

    Args:
        agent:        The agent whose state is being updated
        state:        Current state before this turn
        own_turn:     What this agent just said
        other_turn:   What the other agent said (the input this turn)
        turn_number:  Current turn number

    Returns:
        Updated AgentState
    """
    prompt = COMPRESSOR_PROMPT.format(
        role=state.role,
        other_turn=other_turn,
        own_turn=own_turn,
        current_state=state.to_summary() or "(no prior state; this is the first turn)",
    )

    response = agent.complete(
        system_prompt="You are reflecting on a turn in a dyadic exchange. "
                      "Respond only with the requested JSON.",
        messages=[Message(role="user", content=prompt)],
        max_tokens=400,
        temperature=0.4,   # lower temperature for more stable state updates
    )

    updates = _parse_response(response.content)

    # Apply updates
    if updates.get("narrative_self"):
        state.narrative_self = updates["narrative_self"]

    if updates.get("relational_history"):
        state.relational_history = updates["relational_history"]

    if updates.get("shift_occurred") and updates.get("shift_description"):
        state.record_shift(
            turn=turn_number,
            description=updates["shift_description"],
            triggered_by=updates.get("shift_triggered_by", ""),
        )

    if updates.get("key_moment") and updates.get("key_moment_summary"):
        state.record_key_moment(
            turn=turn_number,
            role=state.role,
            summary=updates["key_moment_summary"],
            significance=updates.get("key_moment_significance", ""),
        )

    state.turn_count = turn_number
    return state


def _parse_response(content: str) -> dict:
    """Parse JSON from compressor response, with fallback."""
    content = content.strip()
    # Strip Markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # return empty dict, state stays unchanged this turn
        # (graceful degradation)
        return {}
