"""
simulation/session.py

Session lifecycle:  start, resume, and close sessions.
Creates the session directory, metadata file, and transcript log.
Provides the append_turn() interface used by dyad.py.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path


SESSIONS_DIR = Path("data/sessions")


@dataclass
class TurnRecord:
    """One complete turn in a session transcript."""
    turn: int
    therapist_text: str
    patient_text: str
    therapist_model: str
    patient_model: str
    therapist_tokens: int | None
    patient_tokens: int | None
    unconscious_revealed: bool
    hazard_flags: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).replace(tzinfo=None).isoformat())

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Session:
    """
    Manages a single simulation session (directory, transcript, metadata).
    """

    def __init__(
        self,
        session_id: str,
        therapist_model: str,
        patient_model: str,
        case_name: str,
        orientation: str,
    ):
        self.session_id = session_id
        self.therapist_model = therapist_model
        self.patient_model = patient_model
        self.case_name = case_name
        self.orientation = orientation
        self.turn_count = 0
        self.transcript: list[TurnRecord] = []
        self.created_at = datetime.now(UTC).replace(tzinfo=None).isoformat()

        self.session_dir = SESSIONS_DIR / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._transcript_path = self.session_dir / "transcript.jsonl"
        self._metadata_path = self.session_dir / "metadata.json"

    def append_turn(
        self,
        therapist_text: str,
        patient_text: str,
        therapist_tokens: int | None = None,
        patient_tokens: int | None = None,
        unconscious_revealed: bool = False,
        hazard_flags: list[str] | None = None,
    ) -> TurnRecord:
        """
        Record a completed turn to the transcript.

        Args:
            therapist_text:      What the therapist said
            patient_text:        What the patient said
            therapist_tokens:    Token count for therapist response
            patient_tokens:      Token count for patient response
            unconscious_revealed: Whether unconscious agenda was active
            hazard_flags:        Any hazard signals detected this turn

        Returns:
            The TurnRecord appended to the transcript
        """
        self.turn_count += 1
        record = TurnRecord(
            turn=self.turn_count,
            therapist_text=therapist_text,
            patient_text=patient_text,
            therapist_model=self.therapist_model,
            patient_model=self.patient_model,
            therapist_tokens=therapist_tokens,
            patient_tokens=patient_tokens,
            unconscious_revealed=unconscious_revealed,
            hazard_flags=hazard_flags or [],
        )
        self.transcript.append(record)

        # Append to JSONL file immediately (survive crashes)
        with open(self._transcript_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        return record

    def save_metadata(self, extra: dict | None = None) -> None:
        """Write / update session metadata."""
        meta = {
            "session_id": self.session_id,
            "therapist_model": self.therapist_model,
            "patient_model": self.patient_model,
            "case_name": self.case_name,
            "orientation": self.orientation,
            "turn_count": self.turn_count,
            "created_at": self.created_at,
            "updated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        }
        if extra:
            meta.update(extra)
        with open(self._metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

    def get_history(self) -> list[tuple[str, str]]:
        """
        Return transcript as list of (patient_text, therapist_text) pairs.
        Used by turn_manager to build message history.
        """
        return [(r.patient_text, r.therapist_text) for r in self.transcript]

    def get_tail(self, n: int = 5) -> str:
        """
        Return the last n turns as a plain text string.
        Used by hazard_monitor and unconscious_agenda trigger checking.
        """
        recent = self.transcript[-n:]
        parts = []
        for r in recent:
            parts.append(f"Patient: {r.patient_text}")
            parts.append(f"Therapist: {r.therapist_text}")
        return "\n".join(parts)


def new_session(
    therapist_model: str,
    patient_model: str,
    case_name: str,
    orientation: str,
) -> Session:
    """Create a new session with a fresh ID."""
    session_id = f"session_{datetime.now(UTC).replace(tzinfo=None).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    session = Session(
        session_id=session_id,
        therapist_model=therapist_model,
        patient_model=patient_model,
        case_name=case_name,
        orientation=orientation,
    )
    session.save_metadata()
    return session


def resume_session(session_id: str) -> Session:
    """
    Resume an existing session from its directory.
    Replays the transcript to restore turn_count and history.
    """
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")

    meta_path = session_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    session = Session(
        session_id=session_id,
        therapist_model=meta["therapist_model"],
        patient_model=meta["patient_model"],
        case_name=meta["case_name"],
        orientation=meta["orientation"],
    )

    # Replay transcript
    transcript_path = session_dir / "transcript.jsonl"
    if transcript_path.exists():
        with open(transcript_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    record = TurnRecord(**data)
                    session.transcript.append(record)
                    session.turn_count = record.turn

    session.created_at = meta.get("created_at", session.created_at)
    return session
