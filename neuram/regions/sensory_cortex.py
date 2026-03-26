"""
SensoryCortex — ultra-short-term raw input buffer.

Analogous to sensory memory (iconic memory for vision, echoic for audio):
holds raw unprocessed stimuli for a few seconds before they are filtered
by the Thalamus for relevance. Items that aren't attended to are lost.

Implemented as a TTL-expiring deque — the oldest un-attended traces
simply fall off and are never encoded into working memory.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SensoryTrace:
    """A raw unprocessed sensory impression."""

    content: str
    modality: str = "text"               # text | audio | visual | other
    received_at: float = field(default_factory=time.time)


class SensoryCortex:
    """
    Ultra-short-term raw input buffer.

    Items expire after `ttl_seconds` (default: 3 s).
    Capacity is soft-limited by TTL, not count — just like sensory memory.
    """

    def __init__(self, ttl_seconds: float = 3.0):
        self.ttl = ttl_seconds
        self._buffer: deque[SensoryTrace] = deque()

    def perceive(self, content: str, modality: str = "text") -> SensoryTrace:
        """Register new raw sensory input."""
        trace = SensoryTrace(content=content, modality=modality)
        self._buffer.append(trace)
        return trace

    def flush(self) -> list[SensoryTrace]:
        """Return all non-expired traces; silently drop expired ones."""
        now = time.time()
        fresh = [t for t in self._buffer if (now - t.received_at) < self.ttl]
        self._buffer = deque(fresh)
        return list(self._buffer)

    def drain(self) -> list[SensoryTrace]:
        """
        Consume and return all fresh traces.
        Used by the Thalamus during its attention scan.
        """
        traces = self.flush()
        self._buffer.clear()
        return traces

    def __len__(self) -> int:
        return len(self.flush())
