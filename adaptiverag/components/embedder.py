from __future__ import annotations

from typing import List

import numpy as np
from langchain_ollama import OllamaEmbeddings

from ..core.config import EMBED_MODEL


class Embedder:
    def __init__(self, model: str = EMBED_MODEL, batch_size: int = 50):
        self.model      = OllamaEmbeddings(model=model)
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self.model.embed_documents(texts[i:i + self.batch_size]))
        return out

    def cosine(self, a: List[float], b: List[float]) -> float:
        a, b = np.array(a), np.array(b)
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 1e-9 else 0.0
