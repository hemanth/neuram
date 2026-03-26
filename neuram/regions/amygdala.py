"""
Amygdala — emotional salience processor.

The amygdala modulates memory encoding strength based on emotional
significance. It tags each engram with a salience score (0–1) and
injects a stability pre-boost for high-salience events — analogous to
the norepinephrine/dopamine surge that enhances hippocampal LTP.

Biologically: the basolateral amygdala communicates with the hippocampus
and prefrontal cortex to modulate consolidation based on arousal/valence.
High-stress or rewarding events are encoded more durably.
"""
from __future__ import annotations

from typing import Optional

from neuram.encoder import score_salience
from neuram.models import Engram


class Amygdala:
    """
    Emotional/salience tagger for incoming engrams.
    """

    def __init__(
        self,
        high_salience_threshold: float = 0.75,
        stability_boost_factor: float = 2.0,
    ):
        self.high_salience_threshold = high_salience_threshold
        self.stability_boost = stability_boost_factor

    def tag(self, engram: Engram, salience: Optional[float] = None) -> Engram:
        """
        Assign salience to engram and apply stability pre-boost if warranted.

        If `salience` is not provided, it is auto-scored from content.
        High-salience engrams receive a stability multiplier (enhanced encoding).
        """
        engram.salience = salience if salience is not None else score_salience(engram.content)

        # Emotional arousal boosts initial stability (norepinephrine effect)
        if engram.salience >= self.high_salience_threshold:
            engram.stability *= self.stability_boost

        return engram

    def is_high_salience(self, engram: Engram) -> bool:
        return engram.salience >= self.high_salience_threshold
