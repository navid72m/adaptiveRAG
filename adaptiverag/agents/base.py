from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseRetrievalAgent(ABC):
    name: str        # unique identifier used by the orchestrator
    description: str # shown to the routing LLM

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5,
                 source_filter: Optional[str] = None) -> List[str]:
        """Return a list of relevant text chunks for the query."""

    def is_available(self) -> bool:
        """Return False if the agent cannot serve requests (missing deps, no creds, etc.)."""
        return True
