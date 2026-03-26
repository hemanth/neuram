"""
Long-Term Potentiation (LTP) — memory strengthening on recall.

Each time an engram is recalled, its synaptic connections are strengthened:
- synaptic_strength resets to 1.0 (memory is "fresh" again)
- stability grows (harder to forget in the future)
- access_count increments (spaced repetition benefit)

Biologically: NMDA receptor activation leads to AMPA receptor insertion,
increasing synaptic efficacy. High-salience events trigger norepinephrine
and dopamine release (amygdala modulation), further boosting consolidation.

The inverse process — Long-Term Depression (LTD) — is modelled by the
Ebbinghaus decay in forgetting.py.
"""
from __future__ import annotations

import math
import time

from neuram.models import Engram

# Stability growth per recall, modulated by salience and access count
_LTP_BASE_GROWTH = 0.3
_SALIENCE_MULTIPLIER = 1.5


def potentiate(engram: Engram) -> Engram:
    """
    Apply LTP to an engram — called on each retrieval (reconsolidation).

    Strengthens synaptic_strength and increases stability so the memory
    persists longer before the next recall is required.
    """
    engram.access_count += 1
    engram.last_accessed = time.time()

    # Stability grows logarithmically with repetition (spacing effect)
    # High-salience engrams benefit more (norepinephrine/dopamine modulation)
    salience_boost = 1.0 + (engram.salience - 0.5) * _SALIENCE_MULTIPLIER
    growth = _LTP_BASE_GROWTH * salience_boost * math.log1p(engram.access_count)
    engram.stability = max(1.0, engram.stability + growth)

    # Reset activation — the memory is vivid again post-recall
    engram.synaptic_strength = 1.0

    return engram


def long_term_depression(engram: Engram, amount: float = 0.2) -> Engram:
    """
    Apply LTD — intentional weakening of an engram (e.g., unlearning).
    Reduces stability and synaptic strength.
    """
    engram.stability = max(0.1, engram.stability - amount)
    engram.synaptic_strength = max(0.0, engram.synaptic_strength - amount)
    return engram
