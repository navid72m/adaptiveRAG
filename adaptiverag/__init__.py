"""
adaptiverag — Agentic RAG Framework built with LangGraph.

Quick start
-----------
>>> from adaptiverag import build_rag
>>> rag = build_rag()          # loads KB, runs setup graph
>>> rag.ask("What is X?")
"""

from adaptiverag.api import build_rag

__all__ = ["build_rag"]
__version__ = "0.1.0"
