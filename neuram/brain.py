"""
Brain — unified high-level API for neuram.

Wires all brain regions together into a single coherent interface.

    Perception pipeline:
        SensoryCortex → Thalamus → Amygdala → PrefrontalCortex
                                                     ↓
                                               Hippocampus (index)
                                                     ↓ (sleep cycle)
                            Neocortex / Cerebellum / BasalGanglia (Redis)

Usage:
    brain = await Brain.create(redis_url="redis://localhost:6379")

    # Perceive new information
    engram = await brain.perceive("Paris is the capital of France",
                                   memory_type=MemoryType.SEMANTIC)

    # Recall via associative pattern completion
    results = await brain.recall("capital of France")
    for engram, score in results:
        print(f"{score:.2f}  {engram.content}")

    # Manual sleep / consolidation cycle
    stats = await brain.sleep()

    # Background auto-sleep
    await brain.start_sleep_cycle()
"""
from __future__ import annotations

from typing import Optional

import redis.asyncio as aioredis

from neuram.encoder import encode
from neuram.models import Engram, MemoryType
from neuram.processes.sleep_cycle import SleepCycle
from neuram.regions.amygdala import Amygdala
from neuram.regions.basal_ganglia import BasalGanglia
from neuram.regions.cerebellum import Cerebellum
from neuram.regions.hippocampus import Hippocampus
from neuram.regions.neocortex import Neocortex
from neuram.regions.prefrontal_cortex import PrefrontalCortex
from neuram.regions.sensory_cortex import SensoryCortex
from neuram.regions.thalamus import Thalamus


class Brain:
    """
    The complete neural memory system for AI agents.

    Each region mirrors its biological counterpart:
      - SensoryCortex: raw input TTL buffer
      - Thalamus: attention-based gating
      - Amygdala: emotional salience tagging
      - PrefrontalCortex: 7±2 working-memory slots with decay
      - Hippocampus: episodic indexing + pattern completion
      - Neocortex: long-term semantic/episodic store (Redis)
      - Cerebellum: procedural/skill memory (Redis)
      - BasalGanglia: habitual/reward memory (Redis)
      - SleepCycle: background consolidation + synaptic pruning
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        working_memory_capacity: int = 7,
        attention_threshold: float = 0.2,
        sleep_interval_seconds: float = 60.0,
    ):
        self.sensory_cortex = SensoryCortex(ttl_seconds=3.0)
        self.thalamus = Thalamus(attention_threshold=attention_threshold)
        self.amygdala = Amygdala()
        self.prefrontal_cortex = PrefrontalCortex(capacity=working_memory_capacity)
        self.hippocampus = Hippocampus()
        self.neocortex = Neocortex(redis_client)
        self.cerebellum = Cerebellum(redis_client)
        self.basal_ganglia = BasalGanglia(redis_client)
        self.sleep_cycle = SleepCycle(
            prefrontal=self.prefrontal_cortex,
            hippocampus=self.hippocampus,
            neocortex=self.neocortex,
            cerebellum=self.cerebellum,
            basal_ganglia=self.basal_ganglia,
            cycle_interval_seconds=sleep_interval_seconds,
        )

    @classmethod
    async def create(
        cls,
        redis_url: str = "redis://localhost:6379",
        **kwargs,
    ) -> "Brain":
        """Factory: create a Brain connected to Redis."""
        redis_client = aioredis.from_url(redis_url, decode_responses=True)
        return cls(redis_client=redis_client, **kwargs)

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    async def perceive(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        salience: Optional[float] = None,
        modality: str = "text",
        context: Optional[dict] = None,
        reward: float = 0.0,
        context_hint: Optional[str] = None,
    ) -> Engram:
        """
        Full perception pipeline:
            SensoryCortex → Thalamus → Amygdala → PrefrontalCortex → Hippocampus

        Args:
            content: The information to perceive.
            memory_type: Episodic / Semantic / Procedural / Habitual.
            salience: Override auto-computed salience (0–1).
            modality: Sensory channel — "text", "audio", "visual".
            context: Arbitrary metadata to attach to the engram.
            reward: Dopamine signal (0–1); non-zero → BasalGanglia encoding.
            context_hint: Current task/topic for top-down thalamic gating.

        Returns:
            The encoded Engram (even if it didn't pass the attention gate).
        """
        # Stage 1: Sensory registration
        trace = self.sensory_cortex.perceive(content, modality)

        # Stage 2: Thalamic gating (attention filter)
        gated = self.thalamus.gate([trace], context_hint=context_hint)

        # Build the engram regardless — but only fully encode if it passed gating
        engram = Engram(content=content, memory_type=memory_type, context=context or {})

        if not gated:
            # Below attention threshold — engram created but not stored anywhere
            return engram

        # Stage 3: Amygdala salience tagging + stability pre-boost
        engram = self.amygdala.tag(engram, salience=salience)

        # Stage 4: Generate semantic embedding
        engram.embedding = encode(content)

        # Stage 5: Working memory (PrefrontalCortex — 7±2 slots, LRU)
        displaced = self.prefrontal_cortex.hold(engram)
        if displaced:
            # Displaced WM item: opportunistic consolidation check
            await self._opportunistic_consolidate(displaced)

        # Stage 6: Hippocampal episodic indexing + temporal association
        self.hippocampus.encode_episode(engram)

        # Stage 7: High-reward → direct BasalGanglia encoding (dopamine shortcut)
        if reward > 0.3:
            await self.basal_ganglia.store(engram, reward_signal=reward)

        return engram

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
    ) -> list[tuple[Engram, float]]:
        """
        Retrieve memories via hippocampal pattern completion + spreading activation.

        Applies LTP (reconsolidation) to all retrieved engrams:
        retrieved memories are temporarily labile, then re-stabilised stronger.

        Args:
            query: Natural language retrieval cue.
            top_k: Maximum number of results.
            memory_type: Filter by memory type if specified.

        Returns:
            List of (Engram, activation_score) sorted by relevance.
        """
        return self.hippocampus.pattern_complete(
            query, top_k=top_k, memory_type=memory_type
        )

    # ------------------------------------------------------------------
    # Sleep / Consolidation
    # ------------------------------------------------------------------

    async def sleep(self) -> dict:
        """
        Run one manual sleep/consolidation cycle.

        - Promotes WM consolidation candidates → LTM
        - Applies Ebbinghaus decay to all LTM engrams
        - Prunes engrams below forgetting threshold (synaptic homeostasis)

        Returns stats dict: {"consolidated": N, "pruned_wm": N, "pruned_ltm": N}
        """
        return await self.sleep_cycle.run_once()

    async def start_sleep_cycle(self) -> None:
        """Start the background periodic sleep/consolidation loop."""
        await self.sleep_cycle.start()

    async def stop_sleep_cycle(self) -> None:
        """Stop the background sleep cycle."""
        await self.sleep_cycle.stop()

    # ------------------------------------------------------------------
    # Explicit forgetting
    # ------------------------------------------------------------------

    async def forget(self, engram_id: str) -> bool:
        """
        Explicitly remove an engram from all memory stores.
        Simulates intentional forgetting / targeted memory suppression.
        """
        removed = False
        if self.prefrontal_cortex.remove(engram_id):
            removed = True
        if self.hippocampus.remove(engram_id):
            removed = True
        for store in (self.neocortex, self.cerebellum, self.basal_ganglia):
            if await store.delete(engram_id):
                removed = True
        return removed

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def working_memory(self) -> list[Engram]:
        """Current contents of working memory (PrefrontalCortex)."""
        return self.prefrontal_cortex.engrams

    @property
    def hippocampal_index_size(self) -> int:
        """Number of engrams in the hippocampal index."""
        return len(self.hippocampus)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Gracefully shut down background tasks and release resources."""
        await self.stop_sleep_cycle()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _opportunistic_consolidate(self, engram: Engram) -> None:
        """
        Attempt to consolidate a WM-displaced engram to LTM.
        Only promotes if it meets the consolidation criteria.
        """
        if engram.access_count >= 3 or engram.salience >= 0.8:
            target = self.sleep_cycle._route_to_ltm(engram)
            if target is not None:
                await target.store(engram)
                self.hippocampus.index(engram)
