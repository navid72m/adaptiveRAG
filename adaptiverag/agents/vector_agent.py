from __future__ import annotations
from typing import List, Optional

from .base import BaseRetrievalAgent
from ..core.runtime import RT


class VectorAgent(BaseRetrievalAgent):
    name        = "vector_search"
    description = "Searches the indexed knowledge base using vector similarity. Best for questions answered by your documents."

    def retrieve(self, query: str, top_k: int = 5,
                 source_filter: Optional[str] = None) -> List[str]:
        where = {"source": source_filter} if source_filter else None
        return RT.retriever.retrieve(query, top_k=top_k, where=where)
