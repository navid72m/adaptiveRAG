from __future__ import annotations

from typing import List, Optional

from .models import KBProfile, PipelineConfig


class Runtime:
    """
    Singleton holding stateful components shared across graph nodes.
    LangGraph state carries serializable payloads; heavy objects (ChromaDB
    clients, models) live here instead.
    """
    embedder:   object = None
    retriever:  object = None
    reranker:   object = None
    llm:        object = None   # decision / routing LLM
    answer_llm: object = None   # answer generation LLM
    profile:    Optional[KBProfile]  = None
    config:     PipelineConfig       = None
    documents:  List[dict]           = None
    agents:     List                 = None  # retrieval agents for multi-agent mode

    def __init__(self):
        self.config    = PipelineConfig()
        self.documents = []
        self.agents    = []


RT = Runtime()
