"""
Forgetting — Ebbinghaus forgetting curve + synaptic homeostasis.

Memory retention decays exponentially over time:

    R(t) = e^(-t / S)

where:
    t = hours elapsed since last access
    S = stability (grows with each Long-Term Potentiation event)

During sleep (synaptic homeostasis hypothesis, Tononi & Cirelli):
weak synaptic connections are downscaled and engrams below a forgetting
threshold are pruned — freeing capacity for new learning.
"""
from __future__ import annotations

import math
import time
from typing import Optional

from neuram.models import Engram


def compute_retention(engram: Engram, now: Optional[float] = None) -> float:
    """
    Compute current memory retention using the Ebbinghaus forgetting curve.

    Returns a value in [0.0, 1.0], where 1.0 = perfectly retained,
    0.0 = completely forgotten.
    """
    if now is None:
        now = time.time()
    hours_elapsed = (now - engram.last_accessed) / 3600.0
    retention = math.exp(-hours_elapsed / engram.stability)
    return max(0.0, min(1.0, retention))


def apply_decay(engram: Engram, now: Optional[float] = None) -> Engram:
    """Update engram's synaptic_strength based on elapsed time since last access."""
    engram.synaptic_strength = compute_retention(engram, now)
    return engram


def is_forgotten(engram: Engram, threshold: float = 0.05) -> bool:
    """Returns True if the engram has decayed below the forgetting threshold."""
    return compute_retention(engram) < threshold


def time_until_forgotten(engram: Engram, threshold: float = 0.05) -> float:
    """
    Returns hours remaining before this engram falls below threshold.
    Useful for scheduling proactive rehearsal.
    """
    # R(t) = e^(-t/S) < threshold  →  t > -S * ln(threshold)
    hours = -engram.stability * math.log(threshold)
    elapsed = (time.time() - engram.last_accessed) / 3600.0
    return max(0.0, hours - elapsed)
