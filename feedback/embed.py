"""Embeddings for learnings. Gemini in production; a deterministic fake in tests (no network)."""
from __future__ import annotations

import hashlib
import math
import re

from agent.config import settings


class GeminiEmbedder:
    def __init__(self):
        from google import genai
        s = settings()
        self.model, self.dim = s.embed_model, s.embed_dim
        self.client = genai.Client(api_key=s.gemini_api_key)

    def embed(self, text: str) -> list[float]:
        from google.genai import types
        r = self.client.models.embed_content(model=self.model, contents=text,
                                             config=types.EmbedContentConfig(output_dimensionality=self.dim))
        return list(r.embeddings[0].values)


class FakeEmbedder:
    """Bag-of-words hashed into EMBED_DIM buckets, L2-normalised: similar wording → high cosine."""
    def __init__(self, dim: int | None = None):
        self.dim = dim or settings().embed_dim

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if len(w) < 3:
                continue
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


def to_pg(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
