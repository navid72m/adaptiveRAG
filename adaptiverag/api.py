"""
Public Python API — used when importing adaptiverag as a library.

Example (Ollama)
----------------
>>> from adaptiverag import build_rag
>>> rag = build_rag()
>>> print(rag.ask("What is X?"))

Example (Claude / Anthropic)
-----------------------------
>>> from adaptiverag import build_rag
>>> rag = build_rag(api_key="sk-ant-...", llm_model="claude-opus-4-7")
>>> print(rag.ask("What is X?"))
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

_CLAUDE_DEFAULT_MODEL  = "claude-opus-4-7"
_OPENAI_DEFAULT_MODEL  = "gpt-4o"

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


def _build_claude_llm(model: str, api_key: str, temperature: float):
    """Instantiate a ChatAnthropic LLM, raising a helpful error if the package is missing."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic is required to use the Claude provider.\n"
            "Install it with:  pip install 'adaptiverag[claude]'"
        )
    return ChatAnthropic(model=model, api_key=api_key, temperature=temperature)


def _build_openai_llm(model: str, api_key: str, temperature: float):
    """Instantiate a ChatOpenAI LLM, raising a helpful error if the package is missing."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai is required to use the OpenAI provider.\n"
            "Install it with:  pip install 'adaptiverag[openai]'"
        )
    return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)


def build_rag(
    llm_model:        str = LLM_MODEL,
    embed_model:      str = EMBED_MODEL,
    kb_path:          Optional[str] = None,
    val_queries_path: Optional[str] = None,
    api_key:          Optional[str] = None,
    openai_api_key:   Optional[str] = None,
) -> AdaptiveRAG:
    """
    Initialize the runtime, run the setup graph (index + optimise), and
    return an :class:`AdaptiveRAG` instance ready to answer questions.

    Parameters
    ----------
    llm_model:
        Model name for the LLM used for routing and answer generation.

        * **Ollama** (default): an Ollama model tag such as ``"gemma4:latest"``.
          The model is pulled automatically if not already local.
        * **Claude**: a Claude model ID such as ``"claude-opus-4-7"``.
          Requires *api_key*.
        * **OpenAI**: a model ID such as ``"gpt-4o"``.
          Requires *openai_api_key*.

    embed_model:
        Ollama model tag for embeddings (always via Ollama).
        Defaults to ``nomic-embed-text:latest``.

    kb_path:
        Path to the knowledge-base folder.  Defaults to ``./knowledge_base``.

    val_queries_path:
        Path to a JSON file with validation queries for pipeline auto-tuning.
        Each entry: ``{"query": "...", "expected_answer": "..."}``.
        Defaults to ``./validation_queries.json``; auto-generated if missing.

    api_key:
        Anthropic API key (``sk-ant-...``).  When provided, the LLM defaults
        to ``claude-opus-4-7`` and Ollama is used only for embeddings.

    openai_api_key:
        OpenAI API key (``sk-...``).  When provided, the LLM defaults to
        ``gpt-4o`` and Ollama is used only for embeddings.
        If both *api_key* and *openai_api_key* are given, Claude takes priority.
    """
    if api_key and openai_api_key:
        raise ValueError(
            "Pass either api_key (Claude) or openai_api_key (OpenAI), not both."
        )

    import adaptiverag.core.config as _cfg

    if kb_path:
        _cfg.KNOWLEDGE_BASE_PATH = kb_path
    if val_queries_path:
        _cfg.VAL_QUERIES_PATH = val_queries_path
    _cfg.EMBED_MODEL = embed_model

    # Ollama is always needed for embeddings — check connectivity + pull embed model.
    available = _ollama_models()
    _ensure_model(embed_model, available)

    RT.embedder  = Embedder(model=embed_model)
    RT.retriever = Retriever(RT.embedder)
    RT.reranker  = Reranker()

    if api_key:
        if llm_model == LLM_MODEL:
            llm_model = _CLAUDE_DEFAULT_MODEL
        RT.llm        = _build_claude_llm(llm_model, api_key, temperature=0.0)
        RT.answer_llm = _build_claude_llm(llm_model, api_key, temperature=0.5)

    elif openai_api_key:
        if llm_model == LLM_MODEL:
            llm_model = _OPENAI_DEFAULT_MODEL
        RT.llm        = _build_openai_llm(llm_model, openai_api_key, temperature=0.0)
        RT.answer_llm = _build_openai_llm(llm_model, openai_api_key, temperature=0.5)

    else:
        # Ollama provider — auto-pull LLM model too.
        _ensure_model(llm_model, available)
        RT.llm        = ChatOllama(model=llm_model, temperature=0.0)
        RT.answer_llm = ChatOllama(model=llm_model, temperature=0.5)

    setup_graph = build_setup_graph()
    setup_graph.invoke(
        {"iteration": 0, "optimizer_log": [], "notes": []},
        config={"configurable": {"thread_id": "setup-1"}},
    )

    return AdaptiveRAG()
