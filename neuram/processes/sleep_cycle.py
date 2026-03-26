"""
SleepCycle — memory consolidation and synaptic homeostasis.

During biological sleep the brain performs three key operations:

1. NREM / Slow-Wave Sleep (SWS):
   - Hippocampal sharp-wave ripples replay recent memories
   - Systems consolidation: hippocampus → neocortex transfer
   - Weak WM engrams that didn't make the cut are pruned

2. Synaptic Homeostasis (Tononi & Cirelli, 2006):
   - All synaptic weights are globally downscaled
   - Only the strongest connections survive (SNR increase)

3. REM Sleep:
   - Cross-domain memory integration (creative associations)
   - Emotional memory processing (amygdala-hippocampus dialogue)

This async process simulates that full cycle:
  - Promote WM consolidation candidates → appropriate LTM region
  - Apply Ebbinghaus decay to all LTM engrams
  - Delete engrams below forgetting threshold (synaptic pruning)
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from neuram.models import MemoryType
from neuram.processes.forgetting import apply_decay, is_forgotten

if TYPE_CHECKING:
    from neuram.regions.basal_ganglia import BasalGanglia
    from neuram.regions.cerebellum import Cerebellum
    from neuram.regions.hippocampus import Hippocampus
    from neuram.regions.neocortex import Neocortex
    from neuram.regions.prefrontal_cortex import PrefrontalCortex

logger = logging.getLogger(__name__)


class SleepCycle:
    """
    Async background consolidation + pruning cycle.

    Simulates the hippocampal-cortical dialogue during sleep:
    replays working memory, transfers to long-term stores, and prunes
    synaptic connections that have decayed below threshold.
    """

    def __init__(
        self,
        prefrontal: "PrefrontalCortex",
        hippocampus: "Hippocampus",
        neocortex: "Neocortex",
        cerebellum: "Cerebellum",
        basal_ganglia: "BasalGanglia",
        cycle_interval_seconds: float = 60.0,
        forgetting_threshold: float = 0.05,
    ):
        self.prefrontal = prefrontal
        self.hippocampus = hippocampus
        self.neocortex = neocortex
        self.cerebellum = cerebellum
        self.basal_ganglia = basal_ganglia
        self.cycle_interval = cycle_interval_seconds
        self.forgetting_threshold = forgetting_threshold
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def run_once(self) -> dict:
        """
        Run one full sleep cycle.

        Returns stats dict with counts of consolidated/pruned engrams.
        """
        stats = {"consolidated": 0, "pruned_wm": 0, "pruned_ltm": 0}

        # Phase 1: Hippocampal replay — promote WM candidates to LTM
        candidates = self.prefrontal.consolidation_candidates()
        for engram in candidates:
            target = self._route_to_ltm(engram)
            if target is not None:
                await target.store(engram)
                self.prefrontal.remove(engram.engram_id)
                self.hippocampus.index(engram)
                stats["consolidated"] += 1
                logger.debug(
                    "Consolidated %s → LTM (%s)", engram.engram_id, engram.memory_type
                )

        # Phase 2: Synaptic homeostasis — decay and prune forgotten WM engrams
        forgotten_wm = self.prefrontal.decay_all()
        stats["pruned_wm"] = len(forgotten_wm)
        for e in forgotten_wm:
            self.hippocampus.remove(e.engram_id)
            logger.debug("Pruned forgotten WM engram %s", e.engram_id)

        # Phase 3: Decay LTM engrams and remove those below threshold
        for store in (self.neocortex, self.cerebellum, self.basal_ganglia):
            all_engrams = await store.all_engrams()
            for engram in all_engrams:
                apply_decay(engram)
                if is_forgotten(engram, self.forgetting_threshold):
                    await store.delete(engram.engram_id)
                    self.hippocampus.remove(engram.engram_id)
                    stats["pruned_ltm"] += 1
                    logger.debug("Pruned forgotten LTM engram %s", engram.engram_id)
                else:
                    await store.update(engram)

        return stats

    def _route_to_ltm(self, engram):
        """Select LTM region by memory type."""
        return {
            MemoryType.EPISODIC: self.neocortex,
            MemoryType.SEMANTIC: self.neocortex,
            MemoryType.PROCEDURAL: self.cerebellum,
            MemoryType.HABITUAL: self.basal_ganglia,
        }.get(engram.memory_type)

    async def start(self) -> None:
        """Start the background sleep cycle loop."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Sleep cycle started (interval=%.0fs)", self.cycle_interval)

    async def stop(self) -> None:
        """Stop the background sleep cycle."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Sleep cycle stopped")

    async def _loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.cycle_interval)
            try:
                stats = await self.run_once()
                logger.info("Sleep cycle complete: %s", stats)
            except Exception:
                logger.exception("Error during sleep cycle")
