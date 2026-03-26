"""
PrefrontalCortex — working memory with limited capacity and natural decay.

Mirrors Miller's Law: working memory holds approximately 7 ± 2 chunks
simultaneously. Items decay without active rehearsal and are displaced
via LRU eviction when capacity is exceeded.

Engrams that have been accessed enough times, or carry high salience,
are flagged as consolidation candidates — ready for Hippocampus to
transfer to long-term memory during the next sleep cycle.

Biologically: the dorsolateral PFC maintains active neural representations
through recurrent excitation. Without continued firing, representations
fade (neural decay ≈ Ebbinghaus decay for working memory timescales).
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Optional

from neuram.models import Engram, MemoryLayer
from neuram.processes.forgetting import apply_decay, is_forgotten

WORKING_MEMORY_CAPACITY = 7        # Miller's Law
CONSOLIDATION_MIN_ACCESSES = 3     # Times accessed before LTM candidacy
CONSOLIDATION_SALIENCE_FLOOR = 0.8 # Or high salience → immediate candidacy


class PrefrontalCortex:
    """
    Active working memory: capacity-limited, decaying, LRU-evicting store.
    """

    def __init__(self, capacity: int = WORKING_MEMORY_CAPACITY):
        self.capacity = capacity
        self._slots: OrderedDict[str, Engram] = OrderedDict()

    @property
    def engrams(self) -> list[Engram]:
        return list(self._slots.values())

    def hold(self, engram: Engram) -> Optional[Engram]:
        """
        Add engram to working memory.

        If the same engram is already held, rehearses it (MRU move).
        Returns the displaced (LRU) engram if capacity is exceeded.
        """
        engram.layer = MemoryLayer.WORKING

        if engram.engram_id in self._slots:
            # Rehearsal — refresh position and timestamp
            self._slots.move_to_end(engram.engram_id)
            self._slots[engram.engram_id] = engram
            return None

        displaced: Optional[Engram] = None
        if len(self._slots) >= self.capacity:
            _, displaced = self._slots.popitem(last=False)  # Evict LRU

        self._slots[engram.engram_id] = engram
        return displaced

    def get(self, engram_id: str) -> Optional[Engram]:
        """Retrieve engram by ID; moves it to MRU position."""
        if engram_id in self._slots:
            self._slots.move_to_end(engram_id)
            return self._slots[engram_id]
        return None

    def rehearse(self, engram_id: str) -> Optional[Engram]:
        """Rehearse an engram — refreshes its last_accessed, prevents decay."""
        engram = self.get(engram_id)
        if engram:
            engram.last_accessed = time.time()
        return engram

    def decay_all(self) -> list[Engram]:
        """
        Apply Ebbinghaus decay to all slots.
        Returns list of engrams that have fallen below the forgetting threshold
        and have been removed from working memory.
        """
        forgotten: list[Engram] = []
        for eid in list(self._slots):
            apply_decay(self._slots[eid])
            if is_forgotten(self._slots[eid]):
                forgotten.append(self._slots.pop(eid))
        return forgotten

    def consolidation_candidates(self) -> list[Engram]:
        """
        Return engrams that are ready to be promoted to long-term memory:
        accessed enough times, or carrying high emotional salience.
        """
        return [
            e for e in self._slots.values()
            if (
                e.access_count >= CONSOLIDATION_MIN_ACCESSES
                or e.salience >= CONSOLIDATION_SALIENCE_FLOOR
            )
        ]

    def remove(self, engram_id: str) -> Optional[Engram]:
        return self._slots.pop(engram_id, None)

    def __len__(self) -> int:
        return len(self._slots)
