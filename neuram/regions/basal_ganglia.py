"""
BasalGanglia — habitual and reward-based memory (Redis-backed).

The basal ganglia (striatum) encodes stimulus-response associations via
dopaminergic reward signals. Habitual memories are:
- Fast and automatic (stimulus → response, no deliberation)
- Strengthened by reward (dopamine prediction error signal)
- Context-independent (unlike episodic memory)
- Resistant to hippocampal damage

Biologically:
- Caudate/Putamen (dorsal striatum): action-outcome → stimulus-response habits
- Nucleus Accumbens (ventral striatum): reward prediction and motivation
- Dopamine (VTA/SNc): encodes reward prediction error (RPE = actual - predicted)
  - Positive RPE → LTP in striatum (strengthen the association)
  - Negative RPE → LTD in striatum (weaken the association)
"""
from __future__ import annotations

import json
from typing import Optional

import redis.asyncio as aioredis

from neuram.models import Engram, MemoryLayer, MemoryType

_KEY_PREFIX = "basal_ganglia:engram:"


class BasalGanglia:
    """
    Habitual/reward-based memory backed by Redis.

    The `reward_signal` parameter (0–1) simulates dopamine release:
    higher reward → higher stability and salience boost.
    """

    def __init__(self, redis_client: aioredis.Redis):
        self._redis = redis_client

    def _key(self, engram_id: str) -> str:
        return f"{_KEY_PREFIX}{engram_id}"

    async def store(self, engram: Engram, reward_signal: float = 0.5) -> None:
        """
        Store habitual engram, modulated by dopamine reward signal.

        reward_signal in [0, 1]:
            0.0 = no reward / punishment
            0.5 = neutral baseline
            1.0 = maximum reward
        """
        engram.layer = MemoryLayer.LONG_TERM
        engram.memory_type = MemoryType.HABITUAL

        # Dopamine boosts stability (reward prediction error → LTP in striatum)
        dopamine_stability = 1.0 + reward_signal * 4.0
        engram.stability = max(engram.stability, dopamine_stability)

        # Reward also increases salience (motivated attention)
        engram.salience = min(1.0, engram.salience + reward_signal * 0.2)
        engram.context["reward_signal"] = reward_signal

        await self._redis.set(
            self._key(engram.engram_id),
            json.dumps(engram.to_dict()),
        )

    async def retrieve(self, engram_id: str) -> Optional[Engram]:
        data = await self._redis.get(self._key(engram_id))
        if data:
            return Engram.from_dict(json.loads(data))
        return None

    async def delete(self, engram_id: str) -> bool:
        return bool(await self._redis.delete(self._key(engram_id)))

    async def all_engrams(self) -> list[Engram]:
        keys = await self._redis.keys(f"{_KEY_PREFIX}*")
        if not keys:
            return []
        values = await self._redis.mget(*keys)
        return [Engram.from_dict(json.loads(v)) for v in values if v]

    async def update(self, engram: Engram) -> None:
        await self.store(engram, reward_signal=engram.context.get("reward_signal", 0.5))
