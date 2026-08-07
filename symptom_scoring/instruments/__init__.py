"""Available rating instruments, keyed by name.

Add an instrument by defining an `Instrument` in a module here and registering
it below; nothing in the pipeline needs to change.
"""

from __future__ import annotations

from symptom_scoring.instrument import Instrument
from symptom_scoring.instruments.madrs import MADRS


REGISTRY: dict[str, Instrument] = {
    MADRS.key: MADRS,
}

DEFAULT_INSTRUMENT = MADRS

__all__ = ["Instrument", "MADRS", "REGISTRY", "DEFAULT_INSTRUMENT", "get_instrument"]


def get_instrument(key: str) -> Instrument:
    try:
        return REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"Unknown instrument {key!r}. Available: {', '.join(sorted(REGISTRY))}"
        ) from None
