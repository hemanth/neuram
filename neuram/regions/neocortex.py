"""
Neocortex — long-term semantic and episodic memory (Redis-backed).

The neocortex is the final destination for consolidated declarative memories.
Semantic facts and episodic events that survive hippocampal consolidation
are stored here in a distributed fashion across cortical areas.

Biologically: temporal lobe (language, semantic facts), frontal lobe
(autobiographical context), parietal lobe (spatial/body associations).
Over time, cortical representations become hippocampus-independent
(standard model of systems consolidation).
"""
from __future__ import annotations

import json
from typing import Optional

import redis.asyncio as aioredis

from neuram.models import Engram, MemoryLayer

_KEY_PREFIX = "neocortex:engram:"


class Neocortex:
    """
    Long-term semantic / episodic memory backed by Redis.
    """

    def __init__(self, redis_client: aioredis.Redis):
        self._redis = redis_client

    def _key(self, engram_id: str) -> str:
        return f"{_KEY_PREFIX}{engram_id}"

    async def store(self, engram: Engram) -> None:
        """Persist engram to neocortex (long-term declarative memory)."""
        engram.layer = MemoryLayer.LONG_TERM
        await self._redis.set(
            self._key(engram.engram_id),
            json.dumps(engram.to_dict()),
        )

    async def retrieve(self, engram_id: str) -> Optional[Engram]:
        """Retrieve a specific engram by ID."""
        data = await self._redis.get(self._key(engram_id))
        if data:
            return Engram.from_dict(json.loads(data))
        return None

    async def delete(self, engram_id: str) -> bool:
        """Delete an engram (synaptic pruning)."""
        return bool(await self._redis.delete(self._key(engram_id)))

    async def all_engrams(self) -> list[Engram]:
        """Scan all semantic memories (used during sleep cycle decay pass)."""
        keys = await self._redis.keys(f"{_KEY_PREFIX}*")
        if not keys:
            return []
        values = await self._redis.mget(*keys)
        return [Engram.from_dict(json.loads(v)) for v in values if v]

    async def update(self, engram: Engram) -> None:
        """Update an existing engram (post-decay or reconsolidation)."""
        await self.store(engram)
