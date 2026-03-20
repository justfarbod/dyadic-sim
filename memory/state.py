"""
memory/state.py

AgentState: The prosthetic memory that gives agents temporal depth.

A standard LLM call is stateless. This object is maintained externally
by the simulation and injected into each agent's context, giving it
a narrative sense of who/what it has become through this particular encounter.

The compression and distortion in this state (what is retained,
what is dropped, what is re-framed) is primary data.
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC


@dataclass
class StateShift:
    """A recorded moment where the agent's understanding changed."""
    turn: int
    description: str
    triggered_by: str  # brief quote or description of what caused it


@dataclass
class KeyMoment:
    """A turn the agent flagged as significant."""
    turn: int
    role: str          # "therapist" or "patient"
    summary: str
    significance: str  # why this moment mattered


@dataclass
class AgentState:
    """
    The evolving self of an agent across a session.

    Updated after each turn by compressor.py.
    Serialised to JSON by persistence.py.
    Injected into the system prompt by turn_manager.py.
    """
    agent_id: str                              # e.g. "therapist_llama3.1"
    role: str                                  # "therapist" or "patient"
    model: str
    session_id: str
    turn_count: int = 0

    # Narrative self (who the agent understands itself to be in this encounter).
    # Updated each turn. Starts from the self_prior, drifts from there.
    narrative_self: str = ""

    # What this agent has noticed about the specific other.
    # This is where reciprocal determination becomes visible.
    relational_history: str = ""

    # Significant turns (moments the compressor judged worth preserving)
    key_moments: list[KeyMoment] = field(default_factory=list)

    # Recorded shifts (moments when understanding changed)
    shifts: list[StateShift] = field(default_factory=list)

    # Drift log (distance from original prior at each turn; filled by analysis)
    drift_log: list[float] = field(default_factory=list)

    # Raw text of the agent's original prior (stored for drift comparison)
    original_prior_text: str = ""

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_summary(self) -> str:
        """
        Render the state as a natural-language summary for prompt injection.
        This is what the agent actually receives as its 'memory'.
        """
        parts = []

        if self.narrative_self:
            parts.append(f"**Who you are in this encounter:**\n{self.narrative_self}")

        if self.relational_history:
            parts.append(f"**What you have noticed about this person:**\n{self.relational_history}")

        if self.shifts:
            recent_shifts = self.shifts[-3:]  # last three shifts
            shift_text = "\n".join(
                f"- Turn {s.turn}: {s.description}"
                for s in recent_shifts
            )
            parts.append(f"**Moments where your understanding shifted:**\n{shift_text}")

        if self.key_moments:
            recent_moments = self.key_moments[-3:]
            moment_text = "\n".join(
                f"- Turn {m.turn}: {m.summary}"
                for m in recent_moments
            )
            parts.append(f"**Moments that mattered:**\n{moment_text}")

        if not parts:
            return ""

        return "\n\n".join(parts)

    def record_shift(self, turn: int, description: str, triggered_by: str) -> None:
        self.shifts.append(StateShift(
            turn=turn,
            description=description,
            triggered_by=triggered_by,
        ))
        self.updated_at = datetime.utcnow().isoformat()

    def record_key_moment(
        self, turn: int, role: str, summary: str, significance: str
    ) -> None:
        self.key_moments.append(KeyMoment(
            turn=turn,
            role=role,
            summary=summary,
            significance=significance,
        ))
        self.updated_at = datetime.now(UTC).replace(tzinfo=None).isoformat()

    def append_drift(self, score: float) -> None:
        self.drift_log.append(round(score, 4))

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "model": self.model,
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "narrative_self": self.narrative_self,
            "relational_history": self.relational_history,
            "key_moments": [
                {
                    "turn": m.turn,
                    "role": m.role,
                    "summary": m.summary,
                    "significance": m.significance,
                }
                for m in self.key_moments
            ],
            "shifts": [
                {
                    "turn": s.turn,
                    "description": s.description,
                    "triggered_by": s.triggered_by,
                }
                for s in self.shifts
            ],
            "drift_log": self.drift_log,
            "original_prior_text": self.original_prior_text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentState":
        state = cls(
            agent_id=data["agent_id"],
            role=data["role"],
            model=data["model"],
            session_id=data["session_id"],
            turn_count=data.get("turn_count", 0),
            narrative_self=data.get("narrative_self", ""),
            relational_history=data.get("relational_history", ""),
            drift_log=data.get("drift_log", []),
            original_prior_text=data.get("original_prior_text", ""),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
        )
        for m in data.get("key_moments", []):
            state.key_moments.append(KeyMoment(**m))
        for s in data.get("shifts", []):
            state.shifts.append(StateShift(**s))
        return state
