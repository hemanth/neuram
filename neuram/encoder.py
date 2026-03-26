"""
Encoder — synaptic embedding of new information.

Converts raw content into high-dimensional vector representations using
sentence-transformers, analogous to how neurons encode sensory patterns
into distributed cortical representations.

Importance scoring simulates amygdala-prefrontal circuits that evaluate
emotional and contextual salience before memory encoding.
"""
from __future__ import annotations

import math
import re
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer

    _MODEL: Optional[SentenceTransformer] = None

    def _get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
        global _MODEL
        if _MODEL is None:
            _MODEL = SentenceTransformer(model_name)
        return _MODEL

    def encode(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
        """Encode text into a semantic embedding vector."""
        model = _get_model(model_name)
        return model.encode(text, convert_to_numpy=True).tolist()

except ImportError:

    def encode(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:  # type: ignore[misc]
        raise ImportError(
            "sentence-transformers is required for embedding. "
            "Install with: pip install sentence-transformers"
        )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# Salience patterns (amygdala-like heuristic scoring)
_HIGH_SALIENCE = [
    r"\b(urgent|critical|important|emergency|error|fail|danger|warning|alert)\b",
    r"[!?]{2,}",
    r"\b(remember|never forget|always|must|crucial|key|vital|essential)\b",
    r"\b(love|hate|fear|excited|angry|sad|happy|proud|ashamed)\b",
]
_LOW_SALIENCE = [
    r"\b(maybe|perhaps|possibly|trivial|minor|insignificant|irrelevant)\b",
    r"\b(fyi|btw|just saying|random|whatever)\b",
]


def score_salience(content: str) -> float:
    """
    Estimate emotional/contextual salience (amygdala-like scoring).

    High-salience content: urgency markers, emotional words, imperatives.
    Returns a value in [0.1, 1.0].
    """
    score = 0.5  # baseline
    text = content.lower()

    for pattern in _HIGH_SALIENCE:
        if re.search(pattern, text, re.IGNORECASE):
            score = min(1.0, score + 0.15)

    for pattern in _LOW_SALIENCE:
        if re.search(pattern, text, re.IGNORECASE):
            score = max(0.1, score - 0.1)

    # Longer, richer content tends to carry more information
    if len(content) > 200:
        score = min(1.0, score + 0.05)

    # Questions signal information-seeking (hippocampal novelty signal)
    if "?" in content:
        score = min(1.0, score + 0.05)

    return round(score, 3)
