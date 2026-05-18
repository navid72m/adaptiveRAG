from __future__ import annotations

from typing import List

from ..core.config import RERANKER_MODEL

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self._model = None
        if _CROSS_ENCODER_AVAILABLE:
            try:
                self._model = CrossEncoder(model_name)
            except Exception as e:
                print(f"  ✗   Reranker init failed: {e}")

    @property
    def available(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, docs: List[str]) -> List[str]:
        if not docs:
            return docs
        if self._model is None:
            return sorted(docs, key=len, reverse=True)
        pairs  = [[query, d] for d in docs]
        scores = self._model.predict(pairs)
        return [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
