"""
Thalamus — sensory relay station and attention gate.

The thalamus receives all sensory signals and acts as the brain's
gatekeeper: only stimuli that exceed the current attention threshold
are forwarded to the PrefrontalCortex for working-memory encoding.
Low-relevance signals are silently discarded.

Biologically: the thalamic reticular nucleus uses top-down signals from
the prefrontal cortex to selectively gate specific sensory thalamic nuclei,
controlling what reaches conscious awareness (Global Workspace Theory).
"""
from __future__ import annotations

from typing import Optional

from neuram.encoder import score_salience
from neuram.regions.sensory_cortex import SensoryTrace


class Thalamus:
    """
    Routes sensory input to working memory based on attention threshold.

    An optional `context_hint` string can boost relevance of traces whose
    content overlaps with the current cognitive context (top-down attention).
    """

    def __init__(self, attention_threshold: float = 0.2):
        self.attention_threshold = attention_threshold

    def gate(
        self,
        traces: list[SensoryTrace],
        context_hint: Optional[str] = None,
    ) -> list[SensoryTrace]:
        """
        Filter sensory traces by relevance.

        Only traces with computed salience >= attention_threshold pass through.
        `context_hint` (current task/topic) can boost contextually relevant items.
        """
        passed: list[SensoryTrace] = []
        for trace in traces:
            salience = score_salience(trace.content)

            if context_hint:
                # Top-down attention: overlap with current context boosts relevance
                trace_words = set(trace.content.lower().split())
                context_words = set(context_hint.lower().split())
                overlap_ratio = len(trace_words & context_words) / max(1, len(context_words))
                salience = min(1.0, salience + overlap_ratio * 0.3)

            if salience >= self.attention_threshold:
                passed.append(trace)

        return passed
