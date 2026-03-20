"""
memory/persistence.py

Save and load AgentState objects to/from JSON.

State objects are saved after every turn so sessions can be
resumed, compared, and used for counterfactual analysis.
The session-crossing test (starting fresh with only the state
object) is run via the --resume flag in run.py.
"""

import json
from pathlib import Path
from memory.state import AgentState


def save_state_snapshot(
    state: AgentState,
    session_dir: Path,
    turn: int,
) -> None:
    """
    Append the current state as a snapshot to the session's state file.

    Args:
        state:       The agent state to save
        session_dir: Path to the session directory
        turn:        Current turn number (used as snapshot key)
    """
    filename = f"{state.role}_state_snapshots.json"
    path = session_dir / filename

    snapshots = _load_snapshots(path)
    snapshots[str(turn)] = state.to_dict()

    with open(path, "w") as f:
        json.dump(snapshots, f, indent=2)


def load_latest_state(session_dir: Path, role: str) -> AgentState | None:
    """
    Load the most recent state snapshot for a given role.

    Args:
        session_dir: Path to the session directory
        role:        "therapist" or "patient"

    Returns:
        The latest AgentState, or None if no snapshots exist
    """
    path = session_dir / f"{role}_state_snapshots.json"
    if not path.exists():
        return None

    snapshots = _load_snapshots(path)
    if not snapshots:
        return None

    latest_turn = max(int(k) for k in snapshots.keys())
    return AgentState.from_dict(snapshots[str(latest_turn)])


def load_all_snapshots(session_dir: Path, role: str) -> dict[int, AgentState]:
    """
    Load all state snapshots for a role. This is used by analysis layer.

    Returns:
        Dict mapping turn number -> AgentState
    """
    path = session_dir / f"{role}_state_snapshots.json"
    if not path.exists():
        return {}

    snapshots = _load_snapshots(path)
    return {
        int(turn): AgentState.from_dict(data)
        for turn, data in snapshots.items()
    }


def _load_snapshots(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
