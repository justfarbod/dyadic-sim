"""
priors/therapist_prior.py

Therapist prior dataclass and system prompt builder.
Merges base priors + orientation variant into a single structured object,
then renders a system prompt to be injected at the start of each turn.
"""

from dataclasses import dataclass, field
from priors.loader import load_therapist_base, load_therapist_variant


@dataclass
class TherapistPrior:
    """
    Complete therapist prior: base + orientation variant merged.
    This is what the therapist agent receives as its constitutive identity.
    """
    orientation: str

    # From base.yaml
    role_prior: str = ""
    structural_priors: list[str] = field(default_factory=list)
    ethical_priors: dict[str, str] = field(default_factory=dict)
    self_prior: str = ""
    relational_prior: str = ""

    # From variant yaml
    agenda_prior: str = ""
    technical_priors: list[str] = field(default_factory=list)

    def build_system_prompt(self, agent_state_summary: str = "") -> str:
        """
        Render the full system prompt for a therapist turn.

        Args:
            agent_state_summary: the agent's current state narrative,
                                 injected so the agent "remembers" who
                                 it has become through this encounter.

        Returns:
            System prompt string passed to the LLM on each turn.
        """
        sections = []

        sections.append("## Your Role\n" + self.role_prior.strip())

        if self.structural_priors:
            lines = "\n".join(f"- {p}" for p in self.structural_priors)
            sections.append("## Structural Priors\n" + lines)

        if self.ethical_priors:
            ethics = "\n\n".join(
                f"**{k.replace('_', ' ').title()}**: {v.strip()}"
                for k, v in self.ethical_priors.items()
            )
            sections.append("## Ethical Priors\n" + ethics)

        sections.append("## Your Orientation\n" + self.agenda_prior.strip())

        if self.technical_priors:
            lines = "\n".join(f"- {p}" for p in self.technical_priors)
            sections.append("## Technical Priors\n" + lines)

        sections.append("## Your Self-Understanding\n" + self.self_prior.strip())

        sections.append("## How You Understand the Patient\n" + self.relational_prior.strip())

        if agent_state_summary:
            sections.append(
                "## Your State in This Encounter\n"
                + agent_state_summary.strip()
                + "\n\n"
                + "This is who you have become through this particular encounter. "
                "It is not a script, it is a living record of how this relationship "
                "has shaped you so far."
            )

        sections.append(
            "## How to Respond\n"
            "Respond as this therapist would, in the moment, in the room. "
            "Do not describe what you are doing. Do not explain your technique. "
            "Simply speak. One to four sentences is usually enough. "
            "Say less than you think you need to."
        )

        return "\n\n---\n\n".join(sections)


def build_therapist_prior(orientation: str = "psychodynamic") -> TherapistPrior:
    """
    Load and merge base + variant priors into a TherapistPrior.

    Args:
        orientation: therapist orientation variant to load

    Returns:
        Fully populated TherapistPrior instance
    """
    base = load_therapist_base()
    variant = load_therapist_variant(orientation)

    return TherapistPrior(
        orientation=orientation,
        role_prior=base.get("role_prior", ""),
        structural_priors=base.get("structural_priors", []),
        ethical_priors=base.get("ethical_priors", {}),
        self_prior=base.get("self_prior", ""),
        relational_prior=base.get("relational_prior", ""),
        agenda_prior=variant.get("agenda_prior", ""),
        technical_priors=variant.get("technical_priors", []),
    )
