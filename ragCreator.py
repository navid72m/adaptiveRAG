"""
Agentic RAG Framework — LangGraph Edition
==========================================
A truly agentic RAG system built with LangGraph state machines.

DESIGN PRINCIPLES
-----------------
Unlike the v2 sequential pipeline, this version has:

  * Explicit, inspectable state (TypedDict) flowing through a graph
  * LLM-driven conditional routing — at each decision point, an LLM picks the next node
  * Reflection loops — critic nodes evaluate output and route back if quality is low
  * Tool-based execution — operations are tools the orchestrator can choose to invoke
  * Two composed graphs:
      - SetupGraph  (build-time):  load → analyze → plan → index → optimize (with reflection)
      - QueryGraph  (per-query):   classify → strategize → retrieve → critique → generate → reflect
  * MemorySaver checkpointing — pause, inspect, resume any execution

WHY THIS IS "AGENTIC" AND THE OLD VERSION WASN'T
------------------------------------------------
The old pipeline always ran the same sequence regardless of input.
This version's orchestrator LLM decides AT RUNTIME:

  - Whether to expand the query with HyDE
  - Whether multi-hop retrieval is needed (based on first-pass results)
  - Whether to rerank (based on result quality)
  - Whether the generated answer is good enough or should be retried
  - Which optimizer module to tune next (based on metric deltas)

Different queries take different paths through the graph.

Requirements
  pip install langgraph langchain-core langchain-ollama langchain-community \
              chromadb pypdf python-docx sentence-transformers numpy tqdm
"""

from __future__ import annotations

# ── stdlib ──────────────────────────────────────────────────────────────────
import glob, json, os, re, statistics, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pypdf
import chromadb
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from tqdm import tqdm

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False

try:
    import docx as python_docx
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False


# ======================== CONFIGURATION ========================
KNOWLEDGE_BASE_PATH = "./knowledge_base"
CHROMA_PERSIST_DIR  = "./chroma_store"
VAL_QUERIES_PATH    = "./validation_queries.json"

LLM_MODEL    = "gemma4:latest"        # tool-calling capable
EMBED_MODEL  = "nomic-embed-text:latest"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "agentic_rag_v3"

MAX_OPTIMIZER_TURNS    = 6       # max orchestrator decisions in setup graph
MAX_QUERY_RETRIES      = 2       # max reflective retries per query
CONFIDENCE_THRESHOLD   = 0.65    # below this, the answer-critic forces a retry
RETRIEVAL_THRESHOLD    = 0.50    # below this, the retrieval-critic triggers multihop

CHUNK_SIZE_OPTIONS = [256, 512, 768, 1024]
TOP_K_OPTIONS      = [3, 5, 7, 10]


# ============================================================
# PART 1: SUPPORTING DATA STRUCTURES
# ============================================================
@dataclass
class KBProfile:
    doc_count:      int = 0
    avg_doc_length: int = 0
    domain:         str = "general"
    structure_type: str = "mixed"
    complexity:     str = "moderate"
    has_code:       bool = False
    has_tables:     bool = False
    has_lists:      bool = False
    languages:      List[str] = field(default_factory=lambda: ["english"])

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"domain={self.domain}, structure={self.structure_type}, "
                f"complexity={self.complexity}, docs={self.doc_count}, "
                f"avg_len={self.avg_doc_length}ch, code={self.has_code}")


@dataclass
class PipelineConfig:
    chunk_strategy: str   = "paragraph"
    chunk_size:     int   = 512
    chunk_overlap:  int   = 64
    top_k:          int   = 5
    temperature:    float = 0.5
    rerank_enabled: bool  = False

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================
# PART 2: CORE COMPONENTS (slimmed from v2, unchanged in spirit)
# ============================================================
class ContentAwareChunker:
    def __init__(self, strategy: str = "paragraph", size: int = 512, overlap: int = 64):
        self.strategy, self.size, self.overlap = strategy, size, overlap

    def chunk(self, text: str) -> List[str]:
        return {
            "sentence":  self._sentence, "paragraph": self._paragraph,
            "code":      self._code,
        }.get(self.strategy, self._fixed)(text)

    def _fixed(self, text):
        out, start = [], 0
        while start < len(text):
            out.append(text[start:start + self.size])
            start += self.size - self.overlap
        return [c for c in out if c.strip()]

    def _sentence(self, text):
        sents = re.split(r'(?<=[.!?])\s+', text)
        out, cur = [], ""
        for s in sents:
            if cur and len(cur) + len(s) > self.size:
                out.append(cur.strip())
                cur = (cur[-self.overlap:] + " " + s) if self.overlap else s
            else:
                cur += (" " if cur else "") + s
        if cur.strip(): out.append(cur.strip())
        return out

    def _paragraph(self, text):
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        out, cur = [], ""
        for p in paras:
            if cur and len(cur) + len(p) > self.size:
                out.append(cur); cur = p
            else:
                cur += ("\n\n" if cur else "") + p
        if cur: out.append(cur)
        result = []
        for c in out:
            result.extend(self._fixed(c) if len(c) > self.size * 1.5 else [c])
        return result

    def _code(self, text):
        bounds = list(re.finditer(r'^(?:def |class |function |public )', text, re.MULTILINE))
        if len(bounds) < 2: return self._paragraph(text)
        chunks = []
        for i, m in enumerate(bounds):
            end = bounds[i + 1].start() if i + 1 < len(bounds) else len(text)
            block = text[m.start():end].strip()
            chunks.extend(self._fixed(block) if len(block) > self.size * 2 else [block])
        return [c for c in chunks if c]


class Embedder:
    def __init__(self, model: str = EMBED_MODEL, batch_size: int = 50):
        self.model = OllamaEmbeddings(model=model)
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self.model.embed_documents(texts[i:i + self.batch_size]))
        return out

    def cosine(self, a, b) -> float:
        a, b = np.array(a), np.array(b)
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 1e-9 else 0.0


class Retriever:
    def __init__(self, embedder: Embedder, persist_dir: str = CHROMA_PERSIST_DIR):
        self.embedder = embedder
        self._client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self._client.get_or_create_collection(COLLECTION_NAME)

    @property
    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        name = self.collection.name
        self._client.delete_collection(name)
        self.collection = self._client.get_or_create_collection(name)

    def index(self, chunks: List[str], metadatas: List[Dict]):
        ids = [f"chunk_{i}_{int(time.time()*1000)}" for i in range(len(chunks))]
        embs = []
        for i in tqdm(range(0, len(chunks), self.embedder.batch_size), desc="Embedding"):
            embs.extend(self.embedder.embed(chunks[i:i + self.embedder.batch_size]))
        self.collection.add(embeddings=embs, documents=chunks,
                            metadatas=metadatas, ids=ids)

    def retrieve(self, query: str, top_k: int = 5,
                 where: Optional[Dict] = None) -> List[str]:
        q_emb = self.embedder.embed([query])[0]
        kwargs = {"n_results": top_k}
        if where: kwargs["where"] = where
        res = self.collection.query(q_emb, **kwargs)
        return res["documents"][0] if res["documents"] else []


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self._model = None
        if _CROSS_ENCODER_AVAILABLE:
            try:
                self._model = CrossEncoder(model_name)
            except Exception as e:
                print(f"  ⚠  Reranker init failed: {e}")

    def rerank(self, query: str, docs: List[str]) -> List[str]:
        if not docs or self._model is None:
            return sorted(docs, key=len, reverse=True) if docs else docs
        pairs = [[query, d] for d in docs]
        scores = self._model.predict(pairs)
        return [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]


# ============================================================
# PART 3: GLOBAL RUNTIME — components shared across graph nodes
# ============================================================
class Runtime:
    """
    Singleton holding the stateful components nodes need to call.
    LangGraph state is for control flow + serializable payloads,
    not heavy objects like ChromaDB clients or models.
    """
    embedder:  Embedder
    retriever: Retriever
    reranker:  Reranker
    llm:       ChatOllama          # for tool/decision LLM calls
    answer_llm:ChatOllama          # for the final answer generation
    profile:   Optional[KBProfile] = None
    config:    PipelineConfig      = PipelineConfig()
    documents: List[Dict]          = []


RT = Runtime()   # populated by setup graph


# ============================================================
# PART 4: TOOLS — atomic operations the agents can invoke
# ============================================================
# Tools accept simple types and return simple types so the LLM can
# reason about them as discrete actions.

@tool
def retrieve_documents(query: str, top_k: int = 5,
                       source_filter: Optional[str] = None) -> List[str]:
    """Retrieve the top_k most relevant chunks from the vector store. Optionally filter by source filename."""
    where = {"source": source_filter} if source_filter else None
    return RT.retriever.retrieve(query, top_k=top_k, where=where)


@tool
def rerank_documents(query: str, docs: List[str]) -> List[str]:
    """Rerank documents using a cross-encoder for higher precision. Returns docs reordered by relevance."""
    return RT.reranker.rerank(query, docs)


@tool
def expand_query_hyde(query: str) -> str:
    """Generate a hypothetical answer (HyDE) to use as an additional retrieval query."""
    domain = RT.profile.domain if RT.profile else "general"
    prompt = (
        f"Write one short paragraph that directly answers this question as if "
        f"it appeared in a {domain} document.\n\nQuestion: {query}\n\nParagraph:"
    )
    resp = RT.answer_llm.invoke(prompt)
    return resp.content.strip()


@tool
def generate_answer(query: str, context: List[str], style: str = "factual") -> str:
    """Generate the final answer from the query and retrieved context. Style: factual|analytical|code|comparison|summarization."""
    if not context:
        return "I don't have enough information to answer that."

    instructions = {
        "factual":       "Answer based only on the context. If unsure, say 'I don't know'.",
        "analytical":    "Analyze the context thoroughly and provide a detailed response.",
        "code":          "Answer the technical question with code examples from the context.",
        "comparison":    "Compare and contrast using only the provided context.",
        "summarization": "Summarize the key points from the provided context.",
    }
    ctx = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))
    prompt = (
        f"{instructions.get(style, instructions['factual'])}\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
    )
    return RT.answer_llm.invoke(prompt).content.strip()


# ============================================================
# PART 5: KB ANALYSIS & PLANNING (called by setup-graph nodes)
# ============================================================
def _profile_documents(documents: List[Dict]) -> KBProfile:
    p = KBProfile()
    if not documents:
        return p
    lengths = [len(d["content"]) for d in documents]
    p.doc_count = len(documents)
    p.avg_doc_length = int(statistics.mean(lengths))
    all_text = " ".join(d["content"] for d in documents)

    code_pats = [r'\bdef \w+\(', r'\bclass \w+[:\(]', r'\bimport \w+',
                 r'\bfunction\s+\w+\s*\(', r'```[\w]*\n']
    p.has_code = sum(1 for pat in code_pats if re.search(pat, all_text)) >= 2
    p.has_tables = bool(re.search(r'\|.+\|.+\|', all_text, re.MULTILINE))
    p.has_lists = bool(re.search(r'^\s*[-•*]\s+\w+', all_text, re.MULTILINE))

    # LLM classification
    sample = "\n\n---\n\n".join(
        f"[{d['filename']}]\n{d['content'][:1500]}" for d in documents[:5]
    )
    prompt = f"""Classify these documents. Return ONLY JSON:
{{
  "domain": "<medical|legal|technical|financial|code|scientific|general>",
  "structure_type": "<narrative|qa|reference|code|mixed>",
  "complexity": "<simple|moderate|complex>"
}}

SAMPLES:
{sample}"""
    try:
        resp = RT.llm.invoke(prompt).content
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            data = json.loads(m.group())
            p.domain = data.get("domain", "general")
            p.structure_type = data.get("structure_type", "mixed")
            p.complexity = data.get("complexity", "moderate")
    except Exception as e:
        print(f"  ⚠  KB classification failed: {e}")
    return p


def _initial_config(profile: KBProfile) -> PipelineConfig:
    cfg = PipelineConfig()
    if profile.avg_doc_length < 500:
        cfg.chunk_size, cfg.chunk_overlap, cfg.chunk_strategy = 200, 20, "sentence"
    elif profile.avg_doc_length < 2000:
        cfg.chunk_size, cfg.chunk_overlap = 400, 40
    elif profile.avg_doc_length < 8000:
        cfg.chunk_size, cfg.chunk_overlap = 700, 90
    else:
        cfg.chunk_size, cfg.chunk_overlap = 1024, 128
    if profile.has_code:
        cfg.chunk_strategy = "code"; cfg.chunk_size = 800; cfg.top_k = 7
    if profile.domain in ("medical", "legal", "scientific"):
        cfg.top_k = 8; cfg.temperature = 0.3; cfg.rerank_enabled = _CROSS_ENCODER_AVAILABLE
    elif profile.domain == "financial":
        cfg.temperature = 0.4
    if profile.complexity == "complex":
        cfg.top_k = min(cfg.top_k + 2, 12); cfg.temperature = max(cfg.temperature - 0.1, 0.1)
    return cfg


# ============================================================
# PART 6: SETUP GRAPH STATE & NODES
# ============================================================
class SetupState(TypedDict, total=False):
    """State for the build-time setup graph."""
    documents:       List[Dict]
    profile:         Dict           # serialized KBProfile
    config:          Dict           # serialized PipelineConfig
    baseline_score:  float
    current_score:   float
    best_score:      float
    iteration:       int
    optimizer_action:str            # 'tune_chunking'|'tune_retrieval'|'tune_generation'|'done'
    optimizer_log:   Annotated[List[str], lambda a, b: a + b]  # appended each turn
    notes:           Annotated[List[str], lambda a, b: a + b]


# ── Setup Graph Nodes ───────────────────────────────────────────────────
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
    profile = _profile_documents(state["documents"])
    RT.profile = profile
    print(f"  ✓ {profile.summary()}")
    return {"profile": profile.to_dict(), "notes": [profile.summary()]}


def node_plan_config(state: SetupState) -> Dict:
    print("\n[setup] Planning initial pipeline config...")
    profile = KBProfile(**state["profile"])
    cfg = _initial_config(profile)
    RT.config = cfg
    print(f"  ✓ chunk_size={cfg.chunk_size}  top_k={cfg.top_k}  "
          f"strategy={cfg.chunk_strategy}  temp={cfg.temperature}")
    return {"config": cfg.to_dict(), "notes": [f"initial config: {cfg.to_dict()}"]}


def node_index_kb(state: SetupState) -> Dict:
    print("\n[setup] Indexing documents...")
    cfg = PipelineConfig(**state["config"])
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


def node_evaluate(state: SetupState) -> Dict:
    """Compute semantic-similarity score across validation queries."""
    val_queries = load_validation_set(VAL_QUERIES_PATH)
    if not val_queries:
        return {"baseline_score": 0.0, "current_score": 0.0, "best_score": 0.0}

    total_sim, count = 0.0, 0
    for item in val_queries:
        try:
            docs = RT.retriever.retrieve(item["query"], top_k=RT.config.top_k)
            ans  = generate_answer.invoke({
                "query": item["query"], "context": docs, "style": "factual"
            })
            embs = RT.embedder.embed([item["expected_answer"], ans])
            sim  = RT.embedder.cosine(embs[0], embs[1])
            total_sim += sim
            count += 1
        except Exception as e:
            print(f"  ⚠  Eval error: {e}")

    score = total_sim / max(count, 1)
    baseline = state.get("baseline_score", score)
    best = max(state.get("best_score", 0.0), score)
    print(f"  ✓ Score: {score:.3f}  (baseline={baseline:.3f}  best={best:.3f})")
    return {
        "current_score": score,
        "baseline_score": baseline,
        "best_score": best,
        "notes": [f"eval score={score:.3f}"],
    }


def node_optimizer_orchestrator(state: SetupState) -> Dict:
    """
    THE AGENTIC HEART OF SETUP.
    An LLM looks at the current state and decides which module to tune next.
    Returns a structured action choice.
    """
    iteration = state.get("iteration", 0) + 1
    profile   = state["profile"]
    config    = state["config"]
    current   = state.get("current_score", 0.0)
    best      = state.get("best_score",    0.0)
    log       = state.get("optimizer_log", [])

    if iteration > MAX_OPTIMIZER_TURNS:
        return {"optimizer_action": "done", "iteration": iteration,
                "notes": ["max optimizer turns reached"]}

    history = "\n".join(log[-5:]) if log else "(no prior attempts)"
    prompt = f"""You are a RAG optimization orchestrator. Decide the next action.

KB PROFILE: {json.dumps(profile)}
CURRENT CONFIG: {json.dumps(config)}
CURRENT SCORE: {current:.3f}     BEST SO FAR: {best:.3f}
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
        resp = RT.llm.invoke(prompt).content
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        action = "done"
        reason = "no decision"
        if m:
            data = json.loads(m.group())
            action = data.get("action", "done")
            reason = data.get("reason", "")
        print(f"\n[orchestrator] iter {iteration}: {action}  — {reason}")
        return {
            "optimizer_action": action,
            "iteration": iteration,
            "optimizer_log": [f"iter {iteration}: {action} ({reason})"],
        }
    except Exception as e:
        print(f"  ⚠  Orchestrator failed: {e}")
        return {"optimizer_action": "done", "iteration": iteration}


def _apply_and_score(cfg: PipelineConfig, state: SetupState,
                     reindex: bool = False) -> Dict:
    RT.config = cfg
    extra = {}
    if reindex:
        extra = node_index_kb({"documents": state["documents"], "config": cfg.to_dict()})
    eval_out = node_evaluate(state)
    out = {"config": cfg.to_dict(), **eval_out}
    out["notes"] = (extra.get("notes", []) + eval_out.get("notes", []))
    return out


def node_tune_chunking(state: SetupState) -> Dict:
    cfg = PipelineConfig(**state["config"])
    sizes = [s for s in CHUNK_SIZE_OPTIONS if s != cfg.chunk_size]
    if not sizes: return {"notes": ["no chunk_size alternatives"]}
    cfg.chunk_size = sizes[(state.get("iteration", 1) - 1) % len(sizes)]
    cfg.chunk_overlap = max(cfg.chunk_size // 8, 32)
    print(f"  → tuning chunk_size = {cfg.chunk_size}")
    return _apply_and_score(cfg, state, reindex=True)


def node_tune_retrieval(state: SetupState) -> Dict:
    cfg = PipelineConfig(**state["config"])
    options = [k for k in TOP_K_OPTIONS if k != cfg.top_k]
    cfg.top_k = options[(state.get("iteration", 1) - 1) % len(options)]
    print(f"  → tuning top_k = {cfg.top_k}")
    return _apply_and_score(cfg, state, reindex=False)


def node_tune_generation(state: SetupState) -> Dict:
    cfg = PipelineConfig(**state["config"])
    cfg.temperature = round(max(0.1, min(0.9,
        cfg.temperature + (0.2 if cfg.temperature < 0.5 else -0.2))), 2)
    print(f"  → tuning temperature = {cfg.temperature}")
    return _apply_and_score(cfg, state, reindex=False)


def node_tune_reranking(state: SetupState) -> Dict:
    cfg = PipelineConfig(**state["config"])
    cfg.rerank_enabled = not cfg.rerank_enabled and _CROSS_ENCODER_AVAILABLE
    print(f"  → tuning rerank_enabled = {cfg.rerank_enabled}")
    return _apply_and_score(cfg, state, reindex=False)


def node_critique_and_commit(state: SetupState) -> Dict:
    """
    Reflection node: if score improved, keep the new config; otherwise log regression.
    The conditional edge after this routes back to the orchestrator.
    """
    current = state.get("current_score", 0.0)
    best    = state.get("best_score",    0.0)
    if current >= best:
        msg = f"✅ accepted (score {current:.3f} ≥ best {best:.3f})"
        return {"best_score": current, "optimizer_log": [msg], "notes": [msg]}
    msg = f"❌ rejected (score {current:.3f} < best {best:.3f})"
    return {"optimizer_log": [msg], "notes": [msg]}


# ── Routing ─────────────────────────────────────────────────────────────
def route_optimizer(state: SetupState) -> str:
    return state.get("optimizer_action", "done")


# ── Build Setup Graph ───────────────────────────────────────────────────
def build_setup_graph():
    g = StateGraph(SetupState)
    g.add_node("load",        node_load_kb)
    g.add_node("analyze",     node_analyze_kb)
    g.add_node("plan",        node_plan_config)
    g.add_node("index",       node_index_kb)
    g.add_node("evaluate",    node_evaluate)
    g.add_node("orchestrate", node_optimizer_orchestrator)
    g.add_node("tune_chunking",   node_tune_chunking)
    g.add_node("tune_retrieval",  node_tune_retrieval)
    g.add_node("tune_generation", node_tune_generation)
    g.add_node("tune_reranking",  node_tune_reranking)
    g.add_node("critique",    node_critique_and_commit)

    g.add_edge(START, "load")
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


# ============================================================
# PART 7: QUERY GRAPH STATE & NODES (PER-QUERY AGENT)
# ============================================================
class QueryState(TypedDict, total=False):
    """State for the per-query agent graph."""
    query:           str
    source_filter:   Optional[str]
    query_type:      str
    strategy:        Dict        # {"use_hyde":bool, "use_rerank":bool, "use_multihop":bool}
    expansions:      List[str]
    retrieved_docs:  List[str]
    retrieval_confidence: float
    reranked_docs:   List[str]
    answer:          str
    confidence:      float
    reflection:      str
    retry_count:     int
    trace:           Annotated[List[str], lambda a, b: a + b]


def node_classify_query(state: QueryState) -> Dict:
    """Lightweight regex + LLM-fallback classifier."""
    q = state["query"].lower()
    if any(w in q for w in ["compare", "vs ", "versus", "contrast", "difference"]):
        qt = "comparison"
    elif any(w in q for w in ["how to", "implement", "code", "function", "debug"]):
        qt = "code"
    elif any(w in q for w in ["why", "explain", "analyze", "implications"]):
        qt = "analytical"
    elif any(w in q for w in ["summarize", "summary", "overview", "tldr"]):
        qt = "summarization"
    else:
        qt = "factual"
    return {"query_type": qt, "trace": [f"classified as: {qt}"]}


def node_strategy_orchestrator(state: QueryState) -> Dict:
    """
    THE AGENTIC HEART OF QUERY-TIME.
    LLM decides which tools to use for THIS query based on type + profile + retries.
    Different queries take different paths through the graph.
    """
    retries = state.get("retry_count", 0)
    prev_reflection = state.get("reflection", "")
    qtype = state.get("query_type", "factual")
    profile = RT.profile.to_dict() if RT.profile else {}

    retry_hint = ""
    if retries > 0:
        retry_hint = (f"\nThis is retry #{retries}. Previous attempt feedback:\n"
                      f"  {prev_reflection}\nChange the strategy.")

    prompt = f"""You are a RAG query strategist. Decide which tools to use for this query.

Query: "{state['query']}"
Type: {qtype}
KB Profile: {json.dumps(profile)}{retry_hint}

Return ONLY JSON:
{{
  "use_hyde": <true|false>,        // expand query with hypothetical answer
  "use_multihop": <true|false>,    // do a second retrieval pass with a follow-up question
  "use_rerank": <true|false>,      // apply cross-encoder reranking
  "top_k": <integer 3-15>,         // how many docs to retrieve
  "reason": "<one-sentence justification>"
}}

Guidelines:
- Factual short queries → minimal tools (no hyde, no multihop, no rerank)
- Analytical/comparison queries → use rerank, often multihop
- Vague queries → use hyde to expand
- Code queries → higher top_k, no rerank"""

    try:
        resp = RT.llm.invoke(prompt).content
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            strategy = json.loads(m.group())
        else:
            strategy = {"use_hyde": False, "use_multihop": False, "use_rerank": False,
                        "top_k": RT.config.top_k, "reason": "fallback"}
    except Exception as e:
        strategy = {"use_hyde": False, "use_multihop": False, "use_rerank": False,
                    "top_k": RT.config.top_k, "reason": f"error: {e}"}

    return {
        "strategy": strategy,
        "trace": [f"strategy: {strategy}"],
    }


def node_expand_query(state: QueryState) -> Dict:
    if not state.get("strategy", {}).get("use_hyde"):
        return {"expansions": [state["query"]]}
    hypothetical = expand_query_hyde.invoke({"query": state["query"]})
    expansions = [state["query"], hypothetical]
    return {"expansions": expansions, "trace": [f"hyde expansion added ({len(hypothetical)} chars)"]}


def node_retrieve(state: QueryState) -> Dict:
    queries = state.get("expansions") or [state["query"]]
    top_k   = state.get("strategy", {}).get("top_k", RT.config.top_k)
    src     = state.get("source_filter")

    seen, docs = set(), []
    for q in queries:
        for d in retrieve_documents.invoke({
            "query": q, "top_k": top_k, "source_filter": src,
        }):
            if d not in seen:
                seen.add(d); docs.append(d)
    return {"retrieved_docs": docs, "trace": [f"retrieved {len(docs)} unique docs"]}


def node_retrieval_critic(state: QueryState) -> Dict:
    """
    LLM-based judge: are these docs sufficient to answer the query?
    Score → conditional edge decides whether multihop is needed.
    """
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"retrieval_confidence": 0.0, "trace": ["no docs retrieved"]}

    sample = "\n".join(f"- {d[:200]}..." for d in docs[:3])
    prompt = f"""Rate how sufficient these retrieved documents are for answering the query.

Query: {state['query']}
Sample of retrieved chunks:
{sample}

Return ONLY JSON: {{"score": <0.0-1.0>, "reason": "<brief>"}}
1.0 = clearly contains the answer.  0.0 = totally irrelevant."""

    try:
        resp = RT.llm.invoke(prompt).content
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        score = float(json.loads(m.group()).get("score", 0.5)) if m else 0.5
    except Exception:
        score = 0.5
    return {
        "retrieval_confidence": score,
        "trace": [f"retrieval confidence: {score:.2f}"],
    }


def node_multihop_retrieve(state: QueryState) -> Dict:
    """Generate a follow-up question and retrieve more docs."""
    docs = state.get("retrieved_docs", [])
    preview = "\n".join(docs[:3])[:1500]
    sub_prompt = (
        f"Based on these partial results, what FOLLOW-UP question would "
        f"retrieve the missing information for: '{state['query']}'?\n\n"
        f"Context so far:\n{preview}\n\nFollow-up question (one sentence):"
    )
    try:
        sub_q = RT.llm.invoke(sub_prompt).content.strip()
        more = retrieve_documents.invoke({
            "query": sub_q, "top_k": max(RT.config.top_k // 2, 2),
            "source_filter": state.get("source_filter"),
        })
        existing = set(docs)
        added = [d for d in more if d not in existing]
        return {
            "retrieved_docs": docs + added,
            "trace": [f"multihop added {len(added)} docs via: {sub_q[:80]}..."],
        }
    except Exception as e:
        return {"trace": [f"multihop failed: {e}"]}


def node_rerank(state: QueryState) -> Dict:
    if not state.get("strategy", {}).get("use_rerank"):
        return {"reranked_docs": state.get("retrieved_docs", [])}
    docs = state.get("retrieved_docs", [])
    ranked = rerank_documents.invoke({"query": state["query"], "docs": docs})
    return {"reranked_docs": ranked, "trace": ["reranked via cross-encoder"]}


def node_generate(state: QueryState) -> Dict:
    docs = state.get("reranked_docs") or state.get("retrieved_docs", [])
    answer = generate_answer.invoke({
        "query": state["query"], "context": docs,
        "style": state.get("query_type", "factual"),
    })
    return {"answer": answer, "trace": [f"generated {len(answer)} chars"]}


def node_answer_critic(state: QueryState) -> Dict:
    """
    Reflection: LLM critiques its own answer.
    If confidence is below threshold, the conditional edge routes back to retry.
    """
    answer = state.get("answer", "")
    docs = state.get("reranked_docs") or state.get("retrieved_docs", [])
    if not answer or not docs:
        return {"confidence": 0.0, "reflection": "no answer or no context"}

    ctx = "\n".join(f"[{i+1}] {d[:200]}..." for i, d in enumerate(docs[:5]))
    prompt = f"""Evaluate this RAG answer.

Query: {state['query']}
Answer: {answer}

Available context (truncated):
{ctx}

Return ONLY JSON:
{{
  "confidence": <0.0-1.0>,
  "grounded": <true|false>,
  "complete": <true|false>,
  "issues": "<comma-separated issues, or 'none'>"
}}

confidence < 0.5 means: hallucinated, missing key facts, or off-topic."""
    try:
        resp = RT.llm.invoke(prompt).content
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            data = json.loads(m.group())
            conf = float(data.get("confidence", 0.5))
            reflection = (f"grounded={data.get('grounded')}, "
                          f"complete={data.get('complete')}, "
                          f"issues={data.get('issues')}")
        else:
            conf, reflection = 0.5, "no parse"
    except Exception as e:
        conf, reflection = 0.5, f"error: {e}"
    return {
        "confidence": conf,
        "reflection": reflection,
        "trace": [f"answer confidence: {conf:.2f} ({reflection})"],
    }


# ── Conditional routers ─────────────────────────────────────────────────
def route_after_retrieval(state: QueryState) -> str:
    if state.get("retrieval_confidence", 0.0) < RETRIEVAL_THRESHOLD \
       and state.get("strategy", {}).get("use_multihop"):
        return "multihop"
    return "rerank"


def route_after_reflection(state: QueryState) -> str:
    if state.get("confidence", 1.0) < CONFIDENCE_THRESHOLD \
       and state.get("retry_count", 0) < MAX_QUERY_RETRIES:
        return "retry"
    return "finish"


def node_prepare_retry(state: QueryState) -> Dict:
    """Increment retry counter; preserve reflection for the strategist."""
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "trace": [f"retry #{state.get('retry_count', 0) + 1} (conf was low)"],
    }


# ── Build Query Graph ───────────────────────────────────────────────────
def build_query_graph():
    g = StateGraph(QueryState)
    g.add_node("classify",       node_classify_query)
    g.add_node("strategize",     node_strategy_orchestrator)
    g.add_node("expand",         node_expand_query)
    g.add_node("retrieve",       node_retrieve)
    g.add_node("retrieval_critic", node_retrieval_critic)
    g.add_node("multihop",       node_multihop_retrieve)
    g.add_node("rerank",         node_rerank)
    g.add_node("generate",       node_generate)
    g.add_node("reflect",        node_answer_critic)
    g.add_node("retry",          node_prepare_retry)

    g.add_edge(START, "classify")
    g.add_edge("classify",   "strategize")
    g.add_edge("strategize", "expand")
    g.add_edge("expand",     "retrieve")
    g.add_edge("retrieve",   "retrieval_critic")

    g.add_conditional_edges(
        "retrieval_critic",
        route_after_retrieval,
        {"multihop": "multihop", "rerank": "rerank"},
    )
    g.add_edge("multihop", "rerank")
    g.add_edge("rerank",   "generate")
    g.add_edge("generate", "reflect")

    g.add_conditional_edges(
        "reflect",
        route_after_reflection,
        {"retry": "retry", "finish": END},
    )
    g.add_edge("retry", "strategize")   # loop back to re-plan

    return g.compile(checkpointer=MemorySaver())


# ============================================================
# PART 8: FILE LOADING (extended formats)
# ============================================================
def _read_txt(p): return open(p, "r", encoding="utf-8", errors="ignore").read()
def _read_pdf(p):
    r = pypdf.PdfReader(p)
    return "".join(pg.extract_text() or "" for pg in r.pages)
def _read_md(p):
    t = open(p, "r", encoding="utf-8", errors="ignore").read()
    t = re.sub(r'```[\w]*\n', '', t); t = re.sub(r'```', '', t)
    t = re.sub(r'^#{1,6}\s+', '', t, flags=re.MULTILINE)
    t = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', t)
    return re.sub(r'[*_`~]', '', t)
def _read_docx(p):
    if not _DOCX_AVAILABLE: return ""
    return "\n".join(p.text for p in python_docx.Document(p).paragraphs if p.text.strip())

_READERS = {".txt": _read_txt, ".pdf": _read_pdf, ".md": _read_md, ".docx": _read_docx}


def load_documents(folder: str) -> List[Dict]:
    docs = []
    for f in sorted(glob.glob(os.path.join(folder, "*.*"))):
        ext = Path(f).suffix.lower()
        reader = _READERS.get(ext)
        if not reader: continue
        try:
            content = reader(f)
        except Exception as e:
            print(f"  ⚠  Couldn't read {f}: {e}"); continue
        if content.strip():
            docs.append({"filename": os.path.basename(f), "content": content,
                         "path": f, "ext": ext.lstrip(".")})
    return docs


def load_validation_set(path: str) -> List[Dict]:
    if not os.path.exists(path):
        sample = [{"query": "What is the main topic?", "expected_answer": "unknown"}]
        with open(path, "w") as fp: json.dump(sample, fp, indent=2)
        return sample
    with open(path) as fp: return json.load(fp)


# ============================================================
# PART 9: MAIN — compose the two graphs and serve
# ============================================================
def main():
    print("=" * 70)
    print("  Agentic RAG Framework — LangGraph Edition")
    print("=" * 70)

    # Initialize the global runtime
    RT.embedder  = Embedder()
    RT.retriever = Retriever(RT.embedder)
    RT.reranker  = Reranker()
    RT.llm       = ChatOllama(model=LLM_MODEL, temperature=0.0)        # decisions
    RT.answer_llm= ChatOllama(model=LLM_MODEL, temperature=0.5)        # answers

    # ── Run Setup Graph ──────────────────────────────────────────────────
    print("\n>>> Compiling setup graph...")
    setup_graph = build_setup_graph()
    setup_config = {"configurable": {"thread_id": "setup-1"}}

    print(">>> Invoking setup graph...\n")
    final_setup_state = setup_graph.invoke(
        {"iteration": 0, "optimizer_log": [], "notes": []},
        config=setup_config,
    )

    print("\n" + "=" * 70)
    print(f"  Setup complete.")
    print(f"  Final config : {final_setup_state['config']}")
    print(f"  Final score  : {final_setup_state.get('best_score', 0.0):.3f}")
    print("=" * 70)

    # ── Compile Query Graph and Serve ────────────────────────────────────
    print("\n>>> Compiling query graph...")
    query_graph = build_query_graph()

    print("\n✅  Agentic RAG ready.")
    print("    Commands:")
    print("      Any question              → routed through agentic graph")
    print("      'from:<file> <question>'  → restrict retrieval to one source")
    print("      'trace'                   → show the last query's full trace")
    print("      'exit'                    → quit\n")

    last_trace = []
    while True:
        try:
            q = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!"); break
        if not q: continue
        if q.lower() == "exit": break
        if q.lower() == "trace":
            for step in last_trace: print(f"  - {step}")
            continue

        source_filter = None
        if q.lower().startswith("from:"):
            parts = q.split(" ", 1)
            if len(parts) == 2:
                source_filter, q = parts[0][5:], parts[1]

        thread_id = f"query-{int(time.time())}"
        result = query_graph.invoke(
            {
                "query": q,
                "source_filter": source_filter,
                "retry_count": 0,
                "trace": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        last_trace = result.get("trace", [])
        print(f"\n{result.get('answer', '(no answer)')}\n")
        print(f"  [confidence={result.get('confidence', 0):.2f}  "
              f"retries={result.get('retry_count', 0)}  "
              f"strategy={result.get('strategy', {}).get('reason', '?')}]\n")


if __name__ == "__main__":
    main()