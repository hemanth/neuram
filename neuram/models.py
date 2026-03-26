"""
Core data model: Engram — a memory trace.

Named after Richard Semon's 'engram': the physical/biophysical substrate
of a memory, imprinted in the brain in response to external stimuli.
Each Engram stores content, its vector embedding, and neurodynamic
properties like synaptic_strength (current activation) and stability
(resistance to forgetting, grows via Long-Term Potentiation).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MemoryType(str, Enum):
    EPISODIC = "episodic"      # Personal events with temporal/spatial context (hippocampus)
    SEMANTIC = "semantic"      # Facts, concepts, general knowledge (neocortex)
    PROCEDURAL = "procedural"  # Skills, patterns, workflows (cerebellum)
    HABITUAL = "habitual"      # Reward-based repetitive behaviors (basal ganglia)


class MemoryLayer(str, Enum):
    SENSORY = "sensory"        # SensoryCortex — raw input, seconds TTL
    WORKING = "working"        # PrefrontalCortex — active, ~7 slots, decays
    LONG_TERM = "long_term"    # Neocortex / Cerebellum / BasalGanglia — Redis


@dataclass
class Engram:
    """
    A memory trace — the fundamental unit of stored memory.

    Neurodynamic properties mirror biological memory:
    - synaptic_strength: current activation level, decays via Ebbinghaus curve
    - stability: resistance to forgetting, grows with each LTP (recall) event
    - salience: emotional significance tagged by the Amygdala (0.0–1.0)
    - associations: spreading activation graph edges [(engram_id, weight)]
    - fragments: sub-engram IDs (memories are reconstructive, not holistic)
    """

    content: str
    memory_type: MemoryType = MemoryType.EPISODIC
    salience: float = 0.5                   # Amygdala-assigned importance (0–1)

    engram_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[list[float]] = None

    # Neurodynamic properties
    synaptic_strength: float = 1.0          # Ebbinghaus retention (0–1), decays over time
    stability: float = 1.0                  # LTP stability (hours); grows with each recall

    # Temporal metadata
    encoded_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    # Associative structure
    fragments: list[str] = field(default_factory=list)              # sub-engram IDs
    associations: list[tuple[str, float]] = field(default_factory=list)  # (id, weight)
    context: dict = field(default_factory=dict)

    layer: MemoryLayer = MemoryLayer.WORKING

    def to_dict(self) -> dict:
        return {
            "engram_id": self.engram_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "salience": self.salience,
            "embedding": self.embedding,
            "synaptic_strength": self.synaptic_strength,
            "stability": self.stability,
            "encoded_at": self.encoded_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "fragments": self.fragments,
            "associations": self.associations,
            "context": self.context,
            "layer": self.layer.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Engram":
        return cls(
            engram_id=data["engram_id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            salience=data["salience"],
            embedding=data.get("embedding"),
            synaptic_strength=data["synaptic_strength"],
            stability=data["stability"],
            encoded_at=data["encoded_at"],
            last_accessed=data["last_accessed"],
            access_count=data["access_count"],
            fragments=data.get("fragments", []),
            associations=[tuple(a) for a in data.get("associations", [])],
            context=data.get("context", {}),
            layer=MemoryLayer(data["layer"]),
        )
