"""
Hippocampus — episodic encoder, consolidation coordinator, pattern completer.

The hippocampus is the brain's memory relay and spatial/temporal index:

- Encodes new episodic memories and binds context (when, where, what)
- Coordinates consolidation from working → long-term cortical memory
- Enables pattern completion: retrieve a full memory from a partial cue
  (CA3 recurrent collaterals)
- Enforces pattern separation: similar but distinct events stored apart
  (dentate gyrus sparse coding)
- Temporal contiguity: events close in time are automatically associated

Biologically: hippocampus is critical for forming new declarative memories
(Henry Molaison / HM case study). After consolidation, memories become
hippocampus-independent and reside in neocortex (multiple trace theory).
"""
from __future__ import annotations

import time
from typing import Optional

from neuram.encoder import encode, cosine_similarity
from neuram.models import Engram, MemoryType
from neuram.processes.ltp import potentiate
from neuram.processes.spreading_activation import activate

# Minimum similarity to form a temporal contiguity association
_ASSOCIATION_SIM_THRESHOLD = 0.35
# How many recent engrams to consider for temporal linking
_TEMPORAL_WINDOW = 5


class Hippocampus:
    """
    Episodic memory index, consolidation hub, and retrieval coordinator.

    Maintains an in-memory index of all engrams (across layers) for fast
    pattern completion and association graph traversal.
    """

    def __init__(self):
        # In-memory episodic index: engram_id → Engram
        self._index: dict[str, Engram] = {}

    def encode_episode(self, engram: Engram) -> Engram:
        """
        Register an engram in the episodic index.

        Generates embedding if missing, stamps temporal context,
        and builds temporal contiguity associations with recent engrams.
        """
        if engram.embedding is None:
            engram.embedding = encode(engram.content)

        engram.context.setdefault("encoded_epoch", time.time())
        self._index[engram.engram_id] = engram

        # Temporal contiguity effect: link to semantically related recent engrams
        self._associate_temporally(engram)
        return engram

    def index(self, engram: Engram) -> None:
        """Add or update an engram in the hippocampal index (post-consolidation)."""
        self._index[engram.engram_id] = engram

    def _associate_temporally(self, new_engram: Engram) -> None:
        """
        Link new engram to recent semantically-similar engrams.
        Simulates the temporal contiguity effect in episodic memory.
        """
        recent = sorted(
            self._index.values(),
            key=lambda e: e.encoded_at,
            reverse=True,
        )[: _TEMPORAL_WINDOW + 1]

        for existing in recent:
            if existing.engram_id == new_engram.engram_id:
                continue
            if existing.embedding is None or new_engram.embedding is None:
                continue
            sim = cosine_similarity(new_engram.embedding, existing.embedding)
            if sim >= _ASSOCIATION_SIM_THRESHOLD:
                # Bidirectional association (Hebbian: fire together, wire together)
                new_engram.associations.append((existing.engram_id, round(sim, 3)))
                existing.associations.append((new_engram.engram_id, round(sim, 3)))

    def pattern_complete(
        self,
        partial_cue: str,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
    ) -> list[tuple[Engram, float]]:
        """
        Retrieve full memories from a partial cue (CA3 pattern completion).

        Applies LTP (reconsolidation) to all retrieved engrams.
        """
        cue_embedding = encode(partial_cue)
        candidates = list(self._index.values())

        if memory_type is not None:
            candidates = [e for e in candidates if e.memory_type == memory_type]

        results = activate(cue_embedding, candidates, top_k=top_k)

        # Reconsolidation: retrieved memories become labile, then re-stabilised
        for engram, _ in results:
            potentiate(engram)
            self._index[engram.engram_id] = engram  # update index with strengthened state

        return results

    def remove(self, engram_id: str) -> Optional[Engram]:
        return self._index.pop(engram_id, None)

    def __len__(self) -> int:
        return len(self._index)

    @property
    def engram_ids(self) -> list[str]:
        return list(self._index.keys())
