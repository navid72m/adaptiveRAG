"""
Public Python API — used when importing adaptiverag as a library.

Example
-------
>>> from adaptiverag import build_rag
>>> rag = build_rag()
>>> print(rag.ask("What is X?"))
>>> print(rag.ask("from:report.pdf Summarize the findings"))
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_ollama import ChatOllama

from .core.config import LLM_MODEL, EMBED_MODEL
from .core.runtime import RT
from .components.embedder import Embedder
from .components.retriever import Retriever
from .components.reranker import Reranker
from .graphs.setup_graph import build_setup_graph
from .graphs.query_graph import build_query_graph

_OLLAMA_BASE = "http://localhost:11434"


def _ollama_models() -> List[str]:
    """Return list of locally available Ollama model names."""
    try:
        with urllib.request.urlopen(f"{_OLLAMA_BASE}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except urllib.error.URLError:
        raise RuntimeError(
            "Cannot connect to Ollama at http://localhost:11434.\n"
            "Make sure Ollama is installed and running:  https://ollama.com/download"
        )


def _model_is_local(model: str, available: List[str]) -> bool:
    tags = set(available)
    bare = {m.split(":")[0] for m in available}
    return model in tags or model.split(":")[0] in bare


def _ensure_model(model: str, available: List[str]) -> None:
    """Pull *model* from the Ollama registry if it is not already local."""
    if _model_is_local(model, available):
        return

    print(f"  Model '{model}' not found locally — pulling from Ollama registry...")
    payload = json.dumps({"name": model}).encode()
    req     = urllib.request.Request(
        f"{_OLLAMA_BASE}/api/pull",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            last_status = ""
            while True:
                line = resp.readline()
                if not line:
                    break
                try:
                    event = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue

                status = event.get("status", "")
                total      = event.get("total", 0)
                completed  = event.get("completed", 0)

                if total and completed:
                    pct = completed / total * 100
                    bar = "#" * int(pct // 5)
                    print(f"\r  [{bar:<20}] {pct:5.1f}%  {status}", end="", flush=True)
                elif status != last_status:
                    print(f"\r  {status:<60}", end="", flush=True)
                    last_status = status

                if status == "success":
                    print(f"\r  ✓ '{model}' pulled successfully.{' ' * 40}")
                    return

        raise RuntimeError(f"Pull of '{model}' ended without a success confirmation.")
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Failed to pull '{model}': HTTP {e.code} — {e.reason}.\n"
            f"Check the model name at https://ollama.com/library"
        )


@dataclass
class QueryResult:
    answer:     str
    confidence: float
    retries:    int
    strategy:   str
    trace:      List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.answer


class AdaptiveRAG:
    """Ready-to-use RAG instance returned by :func:`build_rag`."""

    def __init__(self):
        self._query_graph = build_query_graph()

    def ask(self, question: str,
            source_filter: Optional[str] = None) -> QueryResult:
        """
        Ask a question against the indexed knowledge base.

        Parameters
        ----------
        question:
            Natural-language question.  Prefix with ``from:<filename>`` to
            restrict retrieval to a single source file.
        source_filter:
            Alternative to the prefix syntax — pass the filename directly.
        """
        q = question
        if q.lower().startswith("from:"):
            parts = q.split(" ", 1)
            if len(parts) == 2:
                source_filter, q = parts[0][5:], parts[1]

        result = self._query_graph.invoke(
            {
                "query":         q,
                "source_filter": source_filter,
                "retry_count":   0,
                "trace":         [],
            },
            config={"configurable": {"thread_id": f"query-{int(time.time())}"}},
        )
        return QueryResult(
            answer     = result.get("answer", ""),
            confidence = result.get("confidence", 0.0),
            retries    = result.get("retry_count", 0),
            strategy   = result.get("strategy", {}).get("reason", ""),
            trace      = result.get("trace", []),
        )


def build_rag(
    llm_model:        str = LLM_MODEL,
    embed_model:      str = EMBED_MODEL,
    kb_path:          Optional[str] = None,
    val_queries_path: Optional[str] = None,
) -> AdaptiveRAG:
    """
    Initialize the runtime, run the setup graph (index + optimise), and
    return an :class:`AdaptiveRAG` instance ready to answer questions.

    Parameters
    ----------
    llm_model:
        Ollama model tag for routing and answer generation.
        Defaults to ``gemma4:latest``.  Must be pulled locally first::

            ollama pull gemma4

    embed_model:
        Ollama model tag for embeddings.
        Defaults to ``nomic-embed-text:latest``.  Must be pulled locally::

            ollama pull nomic-embed-text

    kb_path:
        Path to the knowledge-base folder.  Defaults to ``./knowledge_base``
        in the current working directory.
    val_queries_path:
        Path to a JSON file containing validation queries used during
        pipeline auto-tuning.  Each entry must be
        ``{"query": "...", "expected_answer": "..."}``.
        Defaults to ``./validation_queries.json``; auto-created if missing.
    """
    import adaptiverag.core.config as _cfg

    if kb_path:
        _cfg.KNOWLEDGE_BASE_PATH = kb_path
    if val_queries_path:
        _cfg.VAL_QUERIES_PATH = val_queries_path
    _cfg.EMBED_MODEL = embed_model

    # Ensure Ollama is reachable, then pull any missing models automatically.
    available = _ollama_models()
    _ensure_model(llm_model,   available)
    _ensure_model(embed_model, available)

    RT.embedder   = Embedder(model=embed_model)
    RT.retriever  = Retriever(RT.embedder)
    RT.reranker   = Reranker()
    RT.llm        = ChatOllama(model=llm_model, temperature=0.0)
    RT.answer_llm = ChatOllama(model=llm_model, temperature=0.5)

    setup_graph = build_setup_graph()
    setup_graph.invoke(
        {"iteration": 0, "optimizer_log": [], "notes": []},
        config={"configurable": {"thread_id": "setup-1"}},
    )

    return AdaptiveRAG()
