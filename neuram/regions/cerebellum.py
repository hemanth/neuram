"""
Cerebellum — procedural and skill memory (Redis-backed).

The cerebellum stores learned patterns, timing, workflows, and implicit
procedural skills. Unlike declarative memory, procedural memory is:
- Acquired gradually through repetition
- Encoded implicitly (without conscious awareness)
- Highly resistant to forgetting once well-consolidated (high base stability)
- Not dependent on the hippocampus

Biologically: the lateral interpositus nucleus stores conditioned motor
responses; Purkinje cells in cerebellar cortex fine-tune execution timing.
McCormick & Thompson (1984) demonstrated the cerebellum as the engram
site for classically conditioned eyeblink responses.
"""
from __future__ import annotations

import json
from typing import Optional

import redis.asyncio as aioredis

from neuram.models import Engram, MemoryLayer, MemoryType

_KEY_PREFIX = "cerebellum:engram:"
_BASE_STABILITY = 5.0  # Procedural memories start with high forgetting resistance


class Cerebellum:
    """
    Procedural/skill memory backed by Redis.
    """

    def __init__(self, redis_client: aioredis.Redis):
        self._redis = redis_client

    def _key(self, engram_id: str) -> str:
        return f"{_KEY_PREFIX}{engram_id}"

    async def store(self, engram: Engram) -> None:
        """Store procedural engram with elevated base stability."""
        engram.layer = MemoryLayer.LONG_TERM
        engram.memory_type = MemoryType.PROCEDURAL
        # Procedural memories are hard to unlearn — boost base stability
        engram.stability = max(engram.stability, _BASE_STABILITY)
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
        await self.store(engram)
