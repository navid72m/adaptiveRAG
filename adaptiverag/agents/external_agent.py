from __future__ import annotations
from typing import List, Optional

from .base import BaseRetrievalAgent


class ExternalDataAgent(BaseRetrievalAgent):
    """
    Pluggable agent for external data sources (Slack, Gmail, Notion, etc.).

    Subclass this and override `retrieve()` to connect a real source:

        class SlackAgent(ExternalDataAgent):
            name = "slack"
            description = "Searches Slack messages and channels."

            def retrieve(self, query, top_k=5, source_filter=None):
                # call Slack API ...
                return [...]

    Then pass it to build_rag():
        build_rag(extra_agents=[SlackAgent()])
    """
    name        = "external_data"
    description = "Retrieves data from external sources such as Slack, Gmail, or other integrations."

    # Connections to real sources go here — set by subclasses or at runtime.
    _sources: List = []

    def is_available(self) -> bool:
        return bool(self._sources)

    def retrieve(self, query: str, top_k: int = 5,
                 source_filter: Optional[str] = None) -> List[str]:
        results = []
        for source in self._sources:
            try:
                chunks = source.search(query, top_k=top_k)
                results.extend(chunks)
            except Exception as e:
                print(f"  ✗ ExternalDataAgent source {source} error: {e}")
        return results[:top_k]
