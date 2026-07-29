"""Configuration, topic definitions, and PHQ-to-MADRS mappings."""

from __future__ import annotations

from dataclasses import dataclass


MADRS_TOPICS = (
    "Traurigkeit",
    "Anspannung",
    "Schlaf",
    "Appetit",
    "Konzentration",
    "Antriebslosigkeit",
    "Gefühlslosigkeit",
    "Gedanken",
    "Suizid",
)

MADRS_TOPIC_DESCRIPTIONS_EN = {
    "Traurigkeit": (
        "Reported sadness, depressed mood, discouragement, helplessness, "
        "hopelessness, or persistent low mood."
    ),
    "Anspannung": (
        "Inner tension, uneasiness, irritability, restlessness, anxiety, fear, "
        "panic, or feeling keyed up."
    ),
    "Schlaf": (
        "Reduced or disturbed sleep, difficulty falling asleep, waking during "
        "the night, early waking, or sleeping less deeply than usual."
    ),
    "Appetit": (
        "Reduced appetite, loss of interest in food, food not tasting good, "
        "skipping meals, or needing to force oneself to eat."
    ),
    "Konzentration": (
        "Difficulty concentrating, collecting thoughts, reading, following a "
        "conversation, remembering, or sustaining attention."
    ),
    "Antriebslosigkeit": (
        "Lassitude, low drive, difficulty starting or completing ordinary "
        "activities, inertia, or needing great effort to act."
    ),
    "Gefühlslosigkeit": (
        "Reduced interest or pleasure, emotional numbness, inability to feel, "
        "or loss of feelings for activities and other people."
    ),
    "Gedanken": (
        "Pessimistic thoughts, guilt, worthlessness, self-blame, remorse, "
        "feelings of failure, or hopeless expectations about the future."
    ),
    "Suizid": (
        "Thoughts that life is not worth living, wishing to be dead, suicidal "
        "thoughts, self-harm intent, plans, or preparation."
    ),
}

# English relevance patterns are intentionally transparent. They are used before
# translation, where evidence can still be traced to the original transcript.
MADRS_RELEVANCE_PATTERNS = {
    "Traurigkeit": (
        r"\b(sad|sadness|depress\w*|hopeless|despair|down|low mood|miserabl\w*|"
        r"helpless|discourag\w*)\b"
    ),
    "Anspannung": (
        r"\b(anxi\w*|tense|tension|panic\w*|uneasy|irritab\w*|agitat\w*|"
        r"restless\w*|on edge|keyed up|nervous\w*|fear\w*)\b"
    ),
    "Schlaf": (
        r"\b(sleep\w*|asleep|insomnia|awake|waking|wake up|nightmares?|"
        r"restless night|toss(?:ing)? and turn(?:ing)?)\b"
    ),
    "Appetit": (
        r"\b(appetite|eat\w*|food|meal|hungry|hunger|weight|overeat\w*|"
        r"skip(?:ping)? meals?)\b"
    ),
    "Konzentration": (
        r"\b(concentrat\w*|focus\w*|distract\w*|attention|can'?t think|"
        r"cannot think|forget\w*|mind wander\w*|lose track)\b"
    ),
    "Antriebslosigkeit": (
        r"\b(no drive|motivat\w*|can'?t get started|cannot get started|"
        r"hard to start|little energy|no energy|fatigu\w*|exhaust\w*|"
        r"letharg\w*|drained|sluggish|tired)\b"
    ),
    "Gefühlslosigkeit": (
        r"\b(pleasure|enjoy\w*|interest\w*|joy|fun|anhedoni\w*|numb\w*|"
        r"emotionless|don'?t care|cannot feel|can'?t feel|no longer enjoy)\b"
    ),
    "Gedanken": (
        r"\b(guilt\w*|failure|failed|failing|worthless\w*|self[- ]?blame|"
        r"ashamed|shame|not good enough|pessimis\w*|regret\w*)\b"
    ),
    "Suizid": (
        r"\b(death|dead|dying|die|suicid\w*|kill myself|hurt(?:ing)? myself|"
        r"self[- ]harm|better off dead|end my life|no reason to live|"
        r"don'?t want to live|do not want to live|wish I were dead)\b"
    ),
}

PHQ_TO_MADRS = {
    "lack_of_pleasure": "Gefühlslosigkeit",
    "depressed_mood": "Traurigkeit",
    "sleep_problems": "Schlaf",
    "low_energy": "Antriebslosigkeit",
    "appetite_changes": "Appetit",
    "feelings_of_failure_or_guilt": "Gedanken",
    "concentration_problems": "Konzentration",
    "psychomotor_changes": None,
    "thoughts_of_death_or_self_harm": "Suizid",
}

PHQ_MAPPING_QUALITY = {
    "lack_of_pleasure": "approximate",
    "depressed_mood": "close",
    "sleep_problems": "partial",
    "low_energy": "approximate",
    "appetite_changes": "partial",
    "feelings_of_failure_or_guilt": "approximate",
    "concentration_problems": "close",
    "psychomotor_changes": "none",
    "thoughts_of_death_or_self_harm": "close",
}

PHQ_SYMPTOM_LABELS = {
    "lack_of_pleasure": "Little interest or pleasure in doing things",
    "depressed_mood": "Feeling down, depressed, or hopeless",
    "sleep_problems": "Sleep problems",
    "low_energy": "Feeling tired or having little energy",
    "appetite_changes": "Poor appetite or overeating",
    "feelings_of_failure_or_guilt": "Feeling bad about yourself, guilty, or like a failure",
    "concentration_problems": "Trouble concentrating",
    "psychomotor_changes": "Moving or speaking slowly, or feeling restless",
    "thoughts_of_death_or_self_harm": (
        "Thoughts that you would be better off dead or of hurting yourself"
    ),
}

FREQUENCY_VALUES = {
    "not at all": 0,
    "several days": 1,
    "more than half the days": 2,
    "nearly every day": 3,
}

KNOWN_OPENING_PROMPT = (
    "You have just arrived for a therapy session. "
    "The therapist is present and waiting. Say what brings you here."
)


@dataclass(frozen=True)
class ScoringConfig:
    """Runtime settings shared by every session in one evaluation process."""

    model_id: str = "webesama/MADRS-BERT"
    translator_model_id: str = "Helsinki-NLP/opus-mt-en-de"
    source_language: str = "en"
    device: str = "auto"
    batch_size: int | None = None
    max_length: int = 512
    relevance_mode: str = "hybrid"
    combined_similarity_threshold: float = 0.50
    patient_similarity_threshold: float = 0.35
    relevance_acceptance_threshold: float = 0.80
    aggregation_method: str = "maximum_relevant_evidence"

