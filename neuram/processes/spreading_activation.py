"""
Spreading Activation — associative recall via semantic network.

When a memory is retrieved, activation spreads outward through the
associative network (Collins & Loftus, 1975). Each hop decays the signal.

Implementation:
1. Compute cosine similarity between query embedding and all candidates
2. Weight by synaptic_strength and salience (current neural state)
3. Spread activation through the engram association graph (N hops)
4. Return top-k engrams sorted by total activation

Biologically: CA3 recurrent collaterals in hippocampus and cortico-cortical
connections in neocortex enable this fan-out associative retrieval.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from neuram.encoder import cosine_similarity

if TYPE_CHECKING:
    from neuram.models import Engram

SPREAD_DECAY = 0.6       # Activation multiplier per association hop
MIN_ACTIVATION = 0.08    # Prune pathways below this (attention cutoff)


def activate(
    query_embedding: list[float],
    candidates: list["Engram"],
    top_k: int = 10,
    spread_hops: int = 2,
) -> list[tuple["Engram", float]]:
    """
    Retrieve top-k engrams via spreading activation.

    Args:
        query_embedding: Vector representation of the retrieval cue.
        candidates: All engrams to search over.
        top_k: Maximum results to return.
        spread_hops: Number of association graph hops to spread through.

    Returns:
        List of (engram, activation_score) sorted descending by activation.
    """
    activation: dict[str, float] = {}
    engram_map: dict[str, "Engram"] = {e.engram_id: e for e in candidates}

    # Phase 1: Initial activation from semantic similarity
    for engram in candidates:
        if engram.embedding is None:
            continue
        sim = cosine_similarity(query_embedding, engram.embedding)
        # Weight by current synaptic state (strength * salience gate)
        score = sim * engram.synaptic_strength * (0.5 + engram.salience * 0.5)
        if score > MIN_ACTIVATION:
            activation[engram.engram_id] = score

    # Phase 2: Spread activation through association graph
    for _ in range(spread_hops):
        spread: dict[str, float] = {}
        for eid, act in activation.items():
            if eid not in engram_map:
                continue
            for assoc_id, weight in engram_map[eid].associations:
                propagated = act * weight * SPREAD_DECAY
                if propagated > MIN_ACTIVATION:
                    spread[assoc_id] = max(spread.get(assoc_id, 0.0), propagated)
        # Merge spread activations (take max — winner-takes-more)
        for eid, val in spread.items():
            activation[eid] = max(activation.get(eid, 0.0), val)

    # Phase 3: Rank and return top-k
    results = [
        (engram_map[eid], score)
        for eid, score in activation.items()
        if eid in engram_map
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
