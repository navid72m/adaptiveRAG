from __future__ import annotations

import json
import os
import re
from typing import Annotated, Dict, List, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..core.config import (
    KNOWLEDGE_BASE_PATH, MAX_OPTIMIZER_TURNS,
    CHUNK_SIZE_OPTIONS, TOP_K_OPTIONS,
)
from ..core.models import KBProfile, PipelineConfig
from ..core.runtime import RT
from ..components.chunker import ContentAwareChunker
from ..pipeline.kb_analysis import profile_documents, initial_config
from ..pipeline.file_loader import load_documents, load_validation_set
from ..pipeline.tools import generate_answer


# ── State ──────────────────────────────────────────────────────────────────
class SetupState(TypedDict, total=False):
    documents:        List[Dict]
    profile:          Dict
    config:           Dict
    baseline_score:   float
    current_score:    float
    best_score:       float
    iteration:        int
    optimizer_action: str
    skip_optimization: bool
    optimizer_log:    Annotated[List[str], lambda a, b: a + b]
    notes:            Annotated[List[str], lambda a, b: a + b]


# ── Nodes ──────────────────────────────────────────────────────────────────
def node_load_kb(state: SetupState) -> Dict:
    print("\n[setup] Loading knowledge base...")
    docs = load_documents(KNOWLEDGE_BASE_PATH)
    if not docs:
        return {"documents": [], "notes": ["No documents found."]}
    RT.documents = docs
    print(f"  ✓ {len(docs)} document(s) loaded")
    return {"documents": docs, "notes": [f"loaded {len(docs)} docs"]}


def node_analyze_kb(state: SetupState) -> Dict:
    print("\n[setup] Profiling knowledge base...")
    p = profile_documents(state["documents"])
    RT.profile = p
    print(f"  ✓ {p.summary()}")
    return {"profile": p.to_dict(), "notes": [p.summary()]}


def node_plan_config(state: SetupState) -> Dict:
    print("\n[setup] Planning initial pipeline config...")
    cfg = initial_config(KBProfile(**state["profile"]))
    RT.config = cfg
    print(f"  ✓ chunk_size={cfg.chunk_size}  top_k={cfg.top_k}  "
          f"strategy={cfg.chunk_strategy}  temp={cfg.temperature}")
    return {"config": cfg.to_dict(), "notes": [f"initial config: {cfg.to_dict()}"]}


def node_index_kb(state: SetupState) -> Dict:
    print("\n[setup] Indexing documents...")
    cfg     = PipelineConfig(**state["config"])
    chunker = ContentAwareChunker(cfg.chunk_strategy, cfg.chunk_size, cfg.chunk_overlap)
    RT.retriever.clear()
    all_chunks, all_meta = [], []
    for doc in state["documents"]:
        chunks = chunker.chunk(doc["content"])
        all_chunks.extend(chunks)
        meta = {"source": doc["filename"], "ext": doc.get("ext", "txt")}
        all_meta.extend([meta] * len(chunks))
    RT.retriever.index(all_chunks, all_meta)
    print(f"  ✓ {len(all_chunks)} chunks indexed")
    return {"notes": [f"indexed {len(all_chunks)} chunks"]}


def _auto_generate_val_queries(documents: List[Dict], n: int = 5) -> List[Dict]:
    """Generate validation Q&A pairs from the loaded KB using the LLM."""
    if not documents:
        return []

    sample = "\n\n---\n\n".join(
        f"[{d['filename']}]\n{d['content'][:2000]}" for d in documents[:3]
    )
    prompt = f"""Generate {n} question-answer pairs for evaluating a RAG system on these documents.
Return ONLY a valid JSON array, no explanation:
[
  {{"query": "...", "expected_answer": "..."}},
  ...
]
Rules:
- Every question must be answerable from the provided text
- Expected answers should be 1-3 concise sentences
- Cover different topics / sections of the content

DOCUMENTS:
{sample}"""

    try:
        resp = RT.llm.invoke(prompt).content
        m    = re.search(r'\[.*\]', resp, re.DOTALL)
        if not m:
            return []
        queries = json.loads(m.group())
        import adaptiverag.core.config as _cfg
        with open(_cfg.VAL_QUERIES_PATH, "w") as fp:
            json.dump(queries, fp, indent=2)
        print(f"  ✓ Auto-generated {len(queries)} validation queries → {_cfg.VAL_QUERIES_PATH}")
        return queries
    except Exception as e:
        print(f"  ✗   Could not auto-generate validation queries: {e}")
        return []


def node_evaluate(state: SetupState) -> Dict:
    val_queries = load_validation_set()

    if not val_queries:
        print("\n[setup] No validation queries found — generating from KB...")
        val_queries = _auto_generate_val_queries(state.get("documents", []))

    if not val_queries:
        print("  ⚠  Skipping optimization (no validation queries available).")
        return {"baseline_score": 0.0, "current_score": 0.0, "best_score": 0.0,
                "skip_optimization": True}

    total_sim, count = 0.0, 0
    for item in val_queries:
        try:
            docs = RT.retriever.retrieve(item["query"], top_k=RT.config.top_k)
            ans  = generate_answer.invoke({
                "query": item["query"], "context": docs, "style": "factual",
            })
            embs = RT.embedder.embed([item["expected_answer"], ans])
            sim  = RT.embedder.cosine(embs[0], embs[1])
            total_sim += sim
            count += 1
        except Exception as e:
            print(f"  ✗   Eval error: {e}")

    score    = total_sim / max(count, 1)
    baseline = state.get("baseline_score", score)
    best     = max(state.get("best_score", 0.0), score)
    print(f"  ✓ Score: {score:.3f}  (baseline={baseline:.3f}  best={best:.3f})")
    return {
        "current_score":  score,
        "baseline_score": baseline,
        "best_score":     best,
        "notes":          [f"eval score={score:.3f}"],
    }


def node_optimizer_orchestrator(state: SetupState) -> Dict:
    iteration = state.get("iteration", 0) + 1
    if state.get("skip_optimization"):
        return {"optimizer_action": "done", "iteration": iteration,
                "notes": ["optimization skipped — no validation queries"]}
    if iteration > MAX_OPTIMIZER_TURNS:
        return {
            "optimizer_action": "done",
            "iteration": iteration,
            "notes": ["max optimizer turns reached"],
        }

    history = "\n".join(state.get("optimizer_log", [])[-5:]) or "(no prior attempts)"
    prompt  = f"""You are a RAG optimization orchestrator. Decide the next action.

KB PROFILE: {json.dumps(state['profile'])}
CURRENT CONFIG: {json.dumps(state['config'])}
CURRENT SCORE: {state.get('current_score', 0.0):.3f}     BEST SO FAR: {state.get('best_score', 0.0):.3f}
ITERATION: {iteration}/{MAX_OPTIMIZER_TURNS}

RECENT OPTIMIZATION HISTORY:
{history}

Choose ONE next action:
  - "tune_chunking"    : try a different chunk_size or strategy (requires re-indexing)
  - "tune_retrieval"   : try a different top_k
  - "tune_generation"  : try a different temperature
  - "tune_reranking"   : toggle reranking on/off
  - "done"             : stop optimizing (if score is good enough or no progress likely)

Return ONLY JSON: {{"action": "<choice>", "reason": "<one sentence>"}}"""

    try:
        resp   = RT.llm.invoke(prompt).content
        m      = re.search(r'\{.*\}', resp, re.DOTALL)
        action = "done"
        reason = "no decision"
        if m:
            data   = json.loads(m.group())
            action = data.get("action", "done")
            reason = data.get("reason", "")
        print(f"\n[orchestrator] iter {iteration}: {action}  → {reason}")
        return {
            "optimizer_action": action,
            "iteration": iteration,
            "optimizer_log": [f"iter {iteration}: {action} ({reason})"],
        }
    except Exception as e:
        print(f"  ✗   Orchestrator failed: {e}")
        return {"optimizer_action": "done", "iteration": iteration}


def _apply_and_score(cfg: PipelineConfig, state: SetupState,
                     reindex: bool = False) -> Dict:
    RT.config = cfg
    extra = {}
    if reindex:
        extra = node_index_kb({"documents": state["documents"], "config": cfg.to_dict()})
    eval_out = node_evaluate(state)
    out = {"config": cfg.to_dict(), **eval_out}
    out["notes"] = extra.get("notes", []) + eval_out.get("notes", [])
    return out


def node_tune_chunking(state: SetupState) -> Dict:
    cfg   = PipelineConfig(**state["config"])
    sizes = [s for s in CHUNK_SIZE_OPTIONS if s != cfg.chunk_size]
    if not sizes:
        return {"notes": ["no chunk_size alternatives"]}
    cfg.chunk_size    = sizes[(state.get("iteration", 1) - 1) % len(sizes)]
    cfg.chunk_overlap = max(cfg.chunk_size // 8, 32)
    print(f"  ✓ tuning chunk_size = {cfg.chunk_size}")
    return _apply_and_score(cfg, state, reindex=True)


def node_tune_retrieval(state: SetupState) -> Dict:
    cfg     = PipelineConfig(**state["config"])
    options = [k for k in TOP_K_OPTIONS if k != cfg.top_k]
    cfg.top_k = options[(state.get("iteration", 1) - 1) % len(options)]
    print(f"  ✓ tuning top_k = {cfg.top_k}")
    return _apply_and_score(cfg, state, reindex=False)


def node_tune_generation(state: SetupState) -> Dict:
    cfg = PipelineConfig(**state["config"])
    cfg.temperature = round(
        max(0.1, min(0.9, cfg.temperature + (0.2 if cfg.temperature < 0.5 else -0.2))), 2
    )
    print(f"  ✓ tuning temperature = {cfg.temperature}")
    return _apply_and_score(cfg, state, reindex=False)


def node_tune_reranking(state: SetupState) -> Dict:
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
        ce_ok = True
    except ImportError:
        ce_ok = False
    cfg = PipelineConfig(**state["config"])
    cfg.rerank_enabled = not cfg.rerank_enabled and ce_ok
    print(f"  ✓ tuning rerank_enabled = {cfg.rerank_enabled}")
    return _apply_and_score(cfg, state, reindex=False)


def node_critique_and_commit(state: SetupState) -> Dict:
    current = state.get("current_score", 0.0)
    best    = state.get("best_score",    0.0)
    if current >= best:
        msg = f"✓ accepted (score {current:.3f} ≥ best {best:.3f})"
        return {"best_score": current, "optimizer_log": [msg], "notes": [msg]}
    msg = f"✗ rejected (score {current:.3f} < best {best:.3f})"
    return {"optimizer_log": [msg], "notes": [msg]}


# ── Router ─────────────────────────────────────────────────────────────────
def route_optimizer(state: SetupState) -> str:
    return state.get("optimizer_action", "done")


# ── Graph builder ──────────────────────────────────────────────────────────
def build_setup_graph():
    g = StateGraph(SetupState)
    g.add_node("load",            node_load_kb)
    g.add_node("analyze",         node_analyze_kb)
    g.add_node("plan",            node_plan_config)
    g.add_node("index",           node_index_kb)
    g.add_node("evaluate",        node_evaluate)
    g.add_node("orchestrate",     node_optimizer_orchestrator)
    g.add_node("tune_chunking",   node_tune_chunking)
    g.add_node("tune_retrieval",  node_tune_retrieval)
    g.add_node("tune_generation", node_tune_generation)
    g.add_node("tune_reranking",  node_tune_reranking)
    g.add_node("critique",        node_critique_and_commit)

    g.add_edge(START,      "load")
    g.add_edge("load",     "analyze")
    g.add_edge("analyze",  "plan")
    g.add_edge("plan",     "index")
    g.add_edge("index",    "evaluate")
    g.add_edge("evaluate", "orchestrate")

    g.add_conditional_edges(
        "orchestrate",
        route_optimizer,
        {
            "tune_chunking":   "tune_chunking",
            "tune_retrieval":  "tune_retrieval",
            "tune_generation": "tune_generation",
            "tune_reranking":  "tune_reranking",
            "done":            END,
        },
    )
    for tune_node in ("tune_chunking", "tune_retrieval",
                      "tune_generation", "tune_reranking"):
        g.add_edge(tune_node, "critique")
    g.add_edge("critique", "orchestrate")

    return g.compile(checkpointer=MemorySaver())
