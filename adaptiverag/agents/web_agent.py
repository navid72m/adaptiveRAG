from __future__ import annotations
from typing import List, Optional

from .base import BaseRetrievalAgent


class WebSearchAgent(BaseRetrievalAgent):
    name        = "web_search"
    description = "Searches the web for current or real-time information not in the knowledge base."

    def is_available(self) -> bool:
        try:
            import duckduckgo_search  # noqa: F401
            return True
        except ImportError:
            return False

    def retrieve(self, query: str, top_k: int = 5,
                 source_filter: Optional[str] = None) -> List[str]:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=top_k))
            results = []
            for h in hits:
                title = h.get("title", "")
                body  = h.get("body", "")
                url   = h.get("href", "")
                chunk = f"{title}\n{body}\n[Source: {url}]".strip()
                if chunk:
                    results.append(chunk)
            return results
        except Exception as e:
            print(f"  ✗ WebSearchAgent error: {e}")
            return []
