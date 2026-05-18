"""
Adaptive Agentic RAG Framework
================================
Intelligently configures and optimizes the RAG pipeline based on knowledge base
content analysis. Five cooperative agents replace the static default config:

  Agent 1 – KBAnalyzerAgent   : Statistical + LLM-based KB profiling
  Agent 2 – PlannerAgent      : Maps KB profile → optimal initial pipeline config
  Agent 3 – IndexerAgent      : Builds vector index with the planned configuration
  Agent 4 – OptimizerAgent    : Iteratively improves modules with KB-aware feedback
  Agent 5 – QueryRouterAgent  : Routes each query at inference time for best results

Usage
-----
  python adaptive_rag.py

Expects:
  ./knowledge_base/   – .txt and/or .pdf files
  ./validation_queries.json  – [{"query": "...", "expected_answer": "..."}]

Requirements (same as original):
  pip install pypdf chromadb langchain-ollama langchain-community tqdm
  Ollama running with: gemma4:latest  and  nomic-embed-text:latest
"""

import os, json, time, copy, re, glob, statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import pypdf
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from tqdm import tqdm


# ======================== CONFIGURATION ========================
KNOWLEDGE_BASE_PATH       = "./knowledge_base"
VAL_QUERIES_PATH          = "./validation_queries.json"
MAX_ITERATIONS_PER_MODULE = 3
IMPROVEMENT_THRESHOLD     = 0.05      # 5 % improvement needed to accept a change
LLM_MODEL                 = "gemma4:latest"
EMBED_MODEL               = "nomic-embed-text:latest"

# How many documents / chars to feed the analyzer LLM
KB_SAMPLE_DOCS  = 5
KB_SAMPLE_CHARS = 2_000


# ======================== KB PROFILE ========================
@dataclass
class KBProfile:
    """Rich description of the knowledge base, produced by KBAnalyzerAgent."""

    # ── Document statistics ──────────────────────────────────────────────
    doc_count:      int   = 0
    total_chars:    int   = 0
    avg_doc_length: int   = 0
    max_doc_length: int   = 0
    min_doc_length: int   = 0
    std_doc_length: float = 0.0

    # ── Content characteristics (LLM-detected) ──────────────────────────
    domain:         str = "general"   # medical | legal | technical | financial | code | scientific | general
    structure_type: str = "mixed"     # narrative | qa | reference | code | mixed
    complexity:     str = "moderate"  # simple | moderate | complex

    # ── Format flags (regex-detected) ───────────────────────────────────
    content_types: List[str] = field(default_factory=lambda: ["text"])
    languages:     List[str] = field(default_factory=lambda: ["english"])
    has_code:   bool = False
    has_tables: bool = False
    has_lists:  bool = False

    # ── Recommendations set by PlannerAgent ─────────────────────────────
    recommended_chunk_strategy: str   = "fixed"
    recommended_chunk_size:     int   = 512
    recommended_chunk_overlap:  int   = 50
    recommended_top_k:          int   = 5
    recommended_temperature:    float = 0.7

    def summary(self) -> str:
        return (
            f"Domain={self.domain}  Structure={self.structure_type}  "
            f"Complexity={self.complexity}  Docs={self.doc_count}  "
            f"AvgLen={self.avg_doc_length}ch  "
            f"Code={self.has_code}  Tables={self.has_tables}"
        )


# ======================== AGENT 1 – KB ANALYZER ========================
class KBAnalyzerAgent:
    """
    Produces a KBProfile from the raw document list.
    Step 1 – Statistical analysis (fast, no LLM).
    Step 2 – LLM classification of domain / structure / complexity.
    """

    def __init__(self, llm_model: str = LLM_MODEL):
        self.llm = Ollama(model=llm_model)

    def analyze(self, documents: List[Dict]) -> KBProfile:
        print("\n[KBAnalyzerAgent] Profiling knowledge base...")
        profile = KBProfile()
        if not documents:
            print("  ⚠  No documents found.")
            return profile

        # ── 1a. Statistical metrics ───────────────────────────────────
        lengths = [len(d["content"]) for d in documents]
        profile.doc_count      = len(documents)
        profile.total_chars    = sum(lengths)
        profile.avg_doc_length = int(statistics.mean(lengths))
        profile.max_doc_length = max(lengths)
        profile.min_doc_length = min(lengths)
        profile.std_doc_length = statistics.stdev(lengths) if len(lengths) > 1 else 0.0

        # ── 1b. Regex-based content type detection ────────────────────
        all_text = " ".join(d["content"] for d in documents)

        code_patterns = [
            r'\bdef \w+\(',         r'\bclass \w+[:\(]',
            r'\bimport \w+',        r'\bfunction\s+\w+\s*\(',
            r'\bvar \w+\s*=',       r'\bconst \w+\s*=',
            r'\bif __name__\s*==',  r'```[\w]*\n',
            r'\bpublic\s+\w+\s+\w+\s*\(',
        ]
        code_hits = sum(1 for p in code_patterns if re.search(p, all_text))
        profile.has_code = code_hits >= 2
        if profile.has_code:
            profile.content_types.append("code")

        table_patterns = [r'\|.+\|.+\|', r'\+[-+]+\+']
        profile.has_tables = any(re.search(p, all_text, re.MULTILINE) for p in table_patterns)
        if profile.has_tables:
            profile.content_types.append("tables")

        list_patterns = [r'^\s*[-•*]\s+\w+', r'^\s*\d+\.\s+\w+']
        profile.has_lists = any(re.search(p, all_text, re.MULTILINE) for p in list_patterns)
        if profile.has_lists:
            profile.content_types.append("lists")

        # ── 1c. LLM-based domain / structure / complexity detection ───
        sample_text = "\n\n---\n\n".join(
            f"[Doc {i+1}: {d['filename']}]\n{d['content'][:KB_SAMPLE_CHARS]}"
            for i, d in enumerate(documents[:KB_SAMPLE_DOCS])
        )
        llm_data = self._llm_classify(sample_text)
        profile.domain         = llm_data.get("domain",         "general")
        profile.structure_type = llm_data.get("structure_type", "mixed")
        profile.complexity     = llm_data.get("complexity",     "moderate")
        if "languages" in llm_data:
            profile.languages = llm_data["languages"]

        print(f"  ✓ {profile.summary()}")
        return profile

    def _llm_classify(self, sample_text: str) -> Dict:
        prompt = f"""Analyze these document samples and return ONLY a JSON object.

SAMPLES:
{sample_text}

Return exactly:
{{
  "domain": "<medical|legal|technical|financial|code|scientific|historical|general>",
  "structure_type": "<narrative|qa|reference|code|mixed>",
  "complexity": "<simple|moderate|complex>",
  "languages": ["<language>"]
}}

No explanation, only JSON."""
        try:
            response = self.llm.invoke(prompt).strip()
            m = re.search(r'\{.*\}', response, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception as e:
            print(f"  ⚠  LLM classification failed ({e}); using defaults.")
        return {}


# ======================== AGENT 2 – PLANNER ========================
class PlannerAgent:
    """
    Converts a KBProfile into an optimal initial pipeline configuration.
    Uses fast heuristics as a base, then refines with LLM reasoning.
    """

    def __init__(self, llm_model: str = LLM_MODEL):
        self.llm = Ollama(model=llm_model)

    def plan(self, profile: KBProfile) -> Dict[str, Any]:
        print("\n[PlannerAgent] Designing initial pipeline configuration...")

        config = self._heuristic_plan(profile)
        config = self._llm_refine(profile, config)

        # Write recommendations back to profile for downstream agents
        profile.recommended_chunk_strategy = config.get("chunk_strategy", "fixed")
        profile.recommended_chunk_size      = config["chunking"]["size"]
        profile.recommended_chunk_overlap   = config["chunking"]["overlap"]
        profile.recommended_top_k           = config["retrieval"]["top_k"]
        profile.recommended_temperature     = config["generation"]["temperature"]

        print(f"  ✓ Chunk strategy : {profile.recommended_chunk_strategy}")
        print(f"  ✓ Chunk size/ovlp: {profile.recommended_chunk_size} / {profile.recommended_chunk_overlap}")
        print(f"  ✓ Top-K          : {profile.recommended_top_k}")
        print(f"  ✓ Temperature    : {profile.recommended_temperature}")
        return config

    # ── Heuristic rules ──────────────────────────────────────────────────
    def _heuristic_plan(self, p: KBProfile) -> Dict:
        cfg = {
            "chunk_strategy": "fixed",
            "chunking":  {"type": "chunking",  "size": 512, "overlap": 50},
            "embedding": {"type": "embedding", "model": EMBED_MODEL, "batch_size": 50},
            "retrieval": {"type": "retrieval", "top_k": 5,  "similarity": "cosine"},
            "reranking": {"type": "reranking", "enabled": False, "model": None},
            "generation":{"type": "generation","temperature": 0.7, "max_tokens": 256, "llm": LLM_MODEL},
        }

        # Chunk size from avg document length
        if p.avg_doc_length < 500:
            cfg["chunking"].update(size=200, overlap=20)
            cfg["chunk_strategy"] = "sentence"
        elif p.avg_doc_length < 2_000:
            cfg["chunking"].update(size=400, overlap=40)
            cfg["chunk_strategy"] = "paragraph" if p.structure_type == "narrative" else "fixed"
        elif p.avg_doc_length < 8_000:
            cfg["chunking"].update(size=700, overlap=90)
            cfg["chunk_strategy"] = "paragraph"
        else:
            cfg["chunking"].update(size=1024, overlap=128)
            cfg["chunk_strategy"] = "paragraph"

        # Code documents get their own splitter
        if p.has_code or p.domain == "code":
            cfg["chunking"].update(size=800, overlap=100)
            cfg["chunk_strategy"] = "code"
            cfg["retrieval"]["top_k"] = 7

        # High-precision domains → more context + reranking + conservative generation
        if p.domain in ("medical", "legal", "scientific"):
            cfg["retrieval"]["top_k"] = 8
            cfg["reranking"]["enabled"] = True
            cfg["generation"].update(temperature=0.3, max_tokens=400)
        elif p.domain == "financial":
            cfg["retrieval"]["top_k"] = 6
            cfg["generation"]["temperature"] = 0.4

        # Complexity adjustments
        if p.complexity == "complex":
            cfg["retrieval"]["top_k"] = min(cfg["retrieval"]["top_k"] + 2, 12)
            cfg["generation"]["max_tokens"] = 512
            cfg["generation"]["temperature"] = max(cfg["generation"]["temperature"] - 0.1, 0.1)
        elif p.complexity == "simple":
            cfg["retrieval"]["top_k"] = max(cfg["retrieval"]["top_k"] - 2, 3)
            cfg["generation"]["max_tokens"] = 200

        # Structure-type tweaks
        if p.structure_type == "qa":
            cfg["chunking"]["size"] = max(cfg["chunking"]["size"] - 100, 200)
            cfg["chunk_strategy"] = "paragraph"
        elif p.structure_type == "reference":
            cfg["retrieval"]["top_k"] += 1

        return cfg

    # ── LLM refinement of heuristics ────────────────────────────────────
    def _llm_refine(self, profile: KBProfile, heuristic_config: Dict) -> Dict:
        prompt = f"""You are a RAG systems expert. Refine this initial pipeline config for the knowledge base described.

KB PROFILE:
  Domain         : {profile.domain}
  Structure      : {profile.structure_type}
  Complexity     : {profile.complexity}
  Avg doc length : {profile.avg_doc_length} chars  (max={profile.max_doc_length})
  Has code       : {profile.has_code}
  Has tables     : {profile.has_tables}
  Doc count      : {profile.doc_count}

HEURISTIC CONFIG (improve it):
{json.dumps(heuristic_config, indent=2)}

Consider:
- Are chunk size and overlap right for this domain?
- Is top_k appropriate for the number of docs and complexity?
- Is temperature correct (lower for factual domains, higher for creative)?
- Which chunk_strategy is best: "fixed", "sentence", "paragraph", or "code"?

Return ONLY a JSON object with the same top-level keys, values improved where needed."""
        try:
            response = self.llm.invoke(prompt).strip()
            m = re.search(r'\{.*\}', response, re.DOTALL)
            if m:
                refined = json.loads(m.group())
                # Safely merge: only overwrite sub-dicts that exist in both
                for key in heuristic_config:
                    if key in refined:
                        if isinstance(refined[key], dict) and isinstance(heuristic_config[key], dict):
                            heuristic_config[key].update(refined[key])
                        elif type(refined[key]) == type(heuristic_config[key]):
                            heuristic_config[key] = refined[key]
        except Exception as e:
            print(f"  ⚠  LLM refinement failed ({e}); keeping heuristics.")
        return heuristic_config


# ======================== CONTENT-AWARE CHUNKER ========================
class ContentAwareChunker:
    """
    Selects the best splitting strategy based on content type:
      fixed     – character window (original behaviour)
      sentence  – split on sentence boundaries, merge up to target size
      paragraph – split on blank lines, merge small paragraphs
      code      – split on top-level def/class/function declarations
    """

    def __init__(self, config: Dict, strategy: str = "fixed"):
        self.size     = config.get("size", 512)
        self.overlap  = config.get("overlap", 50)
        self.strategy = strategy

    def chunk(self, text: str) -> List[str]:
        dispatch = {
            "sentence":  self._sentence_chunk,
            "paragraph": self._paragraph_chunk,
            "code":      self._code_chunk,
        }
        return dispatch.get(self.strategy, self._fixed_chunk)(text)

    def _fixed_chunk(self, text: str) -> List[str]:
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start : start + self.size])
            start += self.size - self.overlap
        return [c for c in chunks if c.strip()]

    def _sentence_chunk(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for sent in sentences:
            if current and len(current) + len(sent) > self.size:
                chunks.append(current.strip())
                current = current[-self.overlap:] + " " + sent if self.overlap else sent
            else:
                current += (" " if current else "") + sent
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _paragraph_chunk(self, text: str) -> List[str]:
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        chunks, current = [], ""
        for para in paras:
            if current and len(current) + len(para) > self.size:
                chunks.append(current)
                current = para
            else:
                current += ("\n\n" if current else "") + para
        if current:
            chunks.append(current)
        # Oversized paragraphs → fallback fixed-split
        final = []
        for c in chunks:
            final.extend(self._fixed_chunk(c) if len(c) > self.size * 1.5 else [c])
        return final

    def _code_chunk(self, text: str) -> List[str]:
        boundaries = list(re.finditer(
            r'^(?:def |class |function |public |private |protected )',
            text, re.MULTILINE
        ))
        if len(boundaries) < 2:
            return self._paragraph_chunk(text)
        chunks = []
        for i, match in enumerate(boundaries):
            start = match.start()
            end   = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(text)
            block = text[start:end].strip()
            chunks.extend(self._fixed_chunk(block) if len(block) > self.size * 2 else [block])
        return [c for c in chunks if c]


# ======================== CORE COMPONENTS ========================
class Embedder:
    def __init__(self, config: Dict):
        self.model_name = config.get("model", EMBED_MODEL)
        self.batch_size = config.get("batch_size", 50)
        self.embeddings = OllamaEmbeddings(model=self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        all_embeds = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Embedding ({self.model_name})"):
            all_embeds.extend(self.embeddings.embed_documents(texts[i : i + self.batch_size]))
        return all_embeds


class Retriever:
    def __init__(self, config: Dict, embedder: Embedder):
        self.top_k      = config.get("top_k", 5)
        self.embedder   = embedder
        self._client    = chromadb.Client()
        # Unique collection name prevents key conflicts on re-indexing
        self._coll_name = f"rag_{int(time.time() * 1000)}"
        self.collection = self._client.get_or_create_collection(self._coll_name)

    def index(self, chunks: List[str], metadatas: List[Dict] = None):
        ids        = [f"chunk_{i}" for i in range(len(chunks))]
        embeddings = self.embedder.embed(chunks)
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas or [{}] * len(chunks),
            ids=ids,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        k     = top_k or self.top_k
        q_emb = self.embedder.embed([query])[0]
        res   = self.collection.query(q_emb, n_results=k)
        return res["documents"][0] if res["documents"] else []


class Reranker:
    def __init__(self, config: Dict):
        self.enabled = config.get("enabled", False)

    def rerank(self, query: str, docs: List[str]) -> List[str]:
        # Placeholder: replace with a cross-encoder (e.g., sentence-transformers)
        return docs


class Generator:
    def __init__(self, config: Dict):
        self.llm         = Ollama(model=config.get("llm", LLM_MODEL))
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens  = config.get("max_tokens", 256)

    def generate(self, query: str, context: List[str], query_type: str = "factual") -> str:
        if not context:
            return "I don't know."

        instructions = {
            "factual":       "Answer based only on context. If unsure, say 'I don't know'.",
            "analytical":    "Analyze the context thoroughly and provide a detailed response.",
            "code":          "Answer the technical question using code examples from the context.",
            "comparison":    "Compare and contrast using only the provided context.",
            "summarization": "Summarize the key points from the provided context.",
        }
        instruction = instructions.get(query_type, instructions["factual"])

        prompt = f"""{instruction}

Context:
{chr(10).join(f'[{i+1}] {c}' for i, c in enumerate(context))}

Question: {query}

Answer:"""
        return self.llm.invoke(prompt, temperature=self.temperature).strip()


# ======================== AGENT 5 – QUERY ROUTER ========================
class QueryRouterAgent:
    """
    Classifies each incoming query and returns tuned retrieval parameters.
    Uses fast regex heuristics first; falls back to LLM for ambiguous queries.
    """

    TYPES = {"factual", "analytical", "code", "comparison", "summarization"}

    def __init__(self, profile: KBProfile, llm_model: str = LLM_MODEL):
        self.profile = profile
        self.llm     = Ollama(model=llm_model)
        self._cache: Dict[str, str] = {}

    def classify(self, query: str) -> str:
        if query in self._cache:
            return self._cache[query]

        q = query.lower()
        if any(w in q for w in ["compare", "difference", "vs ", "versus", "contrast"]):
            qt = "comparison"
        elif any(w in q for w in ["how to", "implement", "code for", "function", "debug", "syntax"]):
            qt = "code"
        elif any(w in q for w in ["why", "analyze", "implications", "what causes", "explain"]):
            qt = "analytical"
        elif any(w in q for w in ["summarize", "summary", "overview", "briefly", "tldr"]):
            qt = "summarization"
        else:
            # LLM fallback
            try:
                prompt = (
                    f'Classify this query into ONE of: factual, analytical, code, comparison, summarization\n'
                    f'Query: "{query}"\nReturn only one word.'
                )
                result = self.llm.invoke(prompt).strip().lower()
                qt = result if result in self.TYPES else "factual"
            except Exception:
                qt = "factual"

        self._cache[query] = qt
        return qt

    def get_retrieval_params(self, query_type: str) -> Dict:
        base_k = self.profile.recommended_top_k
        table  = {
            "factual":       {"top_k": base_k,              "rerank": False},
            "analytical":    {"top_k": base_k + 3,          "rerank": True},
            "code":          {"top_k": base_k + 2,          "rerank": False},
            "comparison":    {"top_k": base_k + 4,          "rerank": True},
            "summarization": {"top_k": min(base_k + 5, 15), "rerank": False},
        }
        return table.get(query_type, table["factual"])


# ======================== ADAPTIVE RAG PIPELINE ========================
class AdaptiveRAGPipeline:
    """
    Full adaptive pipeline. Combines content-aware chunking with
    query-time routing for best-in-class retrieval and generation.
    """

    def __init__(self, modules: Dict[str, Any], profile: KBProfile):
        self.modules = modules
        self.profile = profile

        strategy        = modules.get("chunk_strategy", profile.recommended_chunk_strategy)
        self.chunker    = ContentAwareChunker(modules["chunking"], strategy)
        self.embedder   = Embedder(modules["embedding"])
        self.retriever  = Retriever(modules["retrieval"], self.embedder)
        self.reranker   = Reranker(modules["reranking"])
        self.generator  = Generator(modules["generation"])
        self.router     = QueryRouterAgent(profile)
        self.is_indexed = False

    # ── Indexer (Agent 3) ────────────────────────────────────────────────
    def index_documents(self, documents: List[Dict]):
        print("\n[IndexerAgent] Chunking and indexing documents...")
        all_chunks, metadata = [], []
        for doc in tqdm(documents, desc="Chunking"):
            chunks = self.chunker.chunk(doc["content"])
            all_chunks.extend(chunks)
            metadata.extend([{"source": doc["filename"]}] * len(chunks))
        print(f"  ✓ {len(all_chunks)} chunks from {len(documents)} document(s)")
        self.retriever.index(all_chunks, metadata)
        self.is_indexed = True

    # ── Inference with routing ───────────────────────────────────────────
    def answer(self, query: str, verbose: bool = False) -> str:
        if not self.is_indexed:
            raise RuntimeError("Pipeline not indexed. Call index_documents() first.")

        query_type = self.router.classify(query)
        params     = self.router.get_retrieval_params(query_type)

        if verbose:
            print(f"  [Router] type={query_type}  top_k={params['top_k']}")

        docs = self.retriever.retrieve(query, top_k=params["top_k"])
        if params["rerank"] and self.reranker.enabled:
            docs = self.reranker.rerank(query, docs)

        return self.generator.generate(query, docs, query_type)

    # ── Hot-swap any module ──────────────────────────────────────────────
    def update_module(self, module_type: str, new_config: Dict):
        self.modules[module_type] = new_config
        if module_type == "chunking":
            strategy        = self.modules.get("chunk_strategy", self.profile.recommended_chunk_strategy)
            self.chunker    = ContentAwareChunker(new_config, strategy)
        elif module_type == "embedding":
            self.embedder   = Embedder(new_config)
            self.retriever  = Retriever(self.modules["retrieval"], self.embedder)
            self.is_indexed = False
        elif module_type == "retrieval":
            self.retriever  = Retriever(new_config, self.embedder)
            self.is_indexed = False
        elif module_type == "reranking":
            self.reranker   = Reranker(new_config)
        elif module_type == "generation":
            self.generator  = Generator(new_config)


# ======================== EVALUATION ========================
@dataclass
class Metrics:
    answer_accuracy:    float = 0.0
    retrieval_precision:float = 0.0
    retrieval_recall:   float = 0.0
    latency_ms:         float = 0.0
    overall_score:      float = 0.0


class Evaluator:
    def __init__(self, val_queries: List[Dict], baseline_pipeline: AdaptiveRAGPipeline):
        self.val_queries      = val_queries
        self.baseline_metrics = (
            self.evaluate(baseline_pipeline) if baseline_pipeline.is_indexed else Metrics()
        )

    def evaluate(self, pipeline: AdaptiveRAGPipeline) -> Metrics:
        acc, prec, rec, lat = 0.0, 0.0, 0.0, 0.0
        for item in tqdm(self.val_queries, desc="Evaluating"):
            query    = item["query"]
            expected = item["expected_answer"].lower()
            t0       = time.time()
            answer   = pipeline.answer(query).lower()
            lat     += (time.time() - t0) * 1000
            acc     += 1.0 if expected in answer or answer in expected else 0.0
            if "relevant_chunks" in item:
                retrieved = pipeline.retriever.retrieve(query)
                rel = set(item["relevant_chunks"])
                ret = set(retrieved)
                tp  = len(rel & ret)
                prec += tp / max(len(ret), 1)
                rec  += tp / max(len(rel), 1)
        n = max(len(self.val_queries), 1)
        m = Metrics(
            answer_accuracy=acc / n,
            retrieval_precision=prec / n,
            retrieval_recall=rec / n,
            latency_ms=lat / n,
        )
        f1 = (2 * m.retrieval_precision * m.retrieval_recall /
              max(m.retrieval_precision + m.retrieval_recall, 1e-9))
        m.overall_score = 0.7 * m.answer_accuracy + 0.3 * f1
        return m

    def is_improvement(self, new_metrics: Metrics) -> bool:
        return (new_metrics.overall_score - self.baseline_metrics.overall_score) >= IMPROVEMENT_THRESHOLD


# ======================== AGENT 4 – OPTIMIZER ========================
class KBAwareLLMModuleGenerator:
    """
    Generates improved module configs using KB profile as context.
    More targeted than the original generic generator.
    """

    def __init__(self, profile: KBProfile, llm_model: str = LLM_MODEL):
        self.profile = profile
        self.llm     = Ollama(model=llm_model)

    def generate_config(self, module_type: str, current_config: Dict,
                        feedback: Optional[str] = None) -> Dict:
        prompt = f"""You are optimizing a RAG system for a {self.profile.domain} knowledge base.

KB CONTEXT:
  Domain={self.profile.domain}  Structure={self.profile.structure_type}
  Complexity={self.profile.complexity}  AvgDocLen={self.profile.avg_doc_length}ch
  HasCode={self.profile.has_code}  HasTables={self.profile.has_tables}

OPTIMIZING: {module_type}
Current config: {json.dumps(current_config, indent=2)}
{"Feedback from last attempt: " + feedback if feedback else "Goal: improve retrieval quality and answer accuracy."}

Generate an improved JSON config for the {module_type} module suited to this {self.profile.domain} domain.
Examples:
  chunking:   {{"size": 400, "overlap": 60}}
  embedding:  {{"model": "nomic-embed-text:latest", "batch_size": 50}}
  retrieval:  {{"top_k": 7, "similarity": "cosine"}}
  generation: {{"temperature": 0.4, "max_tokens": 350, "llm": "{LLM_MODEL}"}}

Return ONLY a JSON object:"""
        try:
            response = self.llm.invoke(prompt).strip()
            m = re.search(r'\{.*\}', response, re.DOTALL)
            if m:
                candidate = json.loads(m.group())
                # Preserve required keys that LLM may have omitted
                for k, v in current_config.items():
                    candidate.setdefault(k, v)
                return candidate
        except Exception as e:
            print(f"  ⚠  Config generation failed ({e}); keeping current.")
        return current_config


class FeedbackEncoder:
    def __init__(self, llm_model: str = LLM_MODEL):
        self.llm = Ollama(model=llm_model)

    def encode(self, module_type: str, old_cfg: Dict, new_cfg: Dict,
               metrics: Metrics, baseline: Metrics) -> str:
        diff   = metrics.overall_score - baseline.overall_score
        prompt = (
            f"RAG optimization attempt for {module_type} module:\n"
            f"Old: {old_cfg}\nNew: {new_cfg}\n"
            f"Score change: {diff:+.3f}  "
            f"(accuracy={metrics.answer_accuracy:.2f}, "
            f"precision={metrics.retrieval_precision:.2f}, "
            f"recall={metrics.retrieval_recall:.2f})\n\n"
            f"Give ONE specific, actionable suggestion for the next configuration attempt."
        )
        try:
            return self.llm.invoke(prompt).strip()
        except Exception:
            return "Try adjusting the parameters in the opposite direction."


class OptimizerAgent:
    """
    Iterates over each module and applies KB-aware LLM-guided search
    for better configurations. Accepts improvements; rolls back regressions.
    """

    # Modules optimized in dependency order (chunking first, generation last)
    MODULE_ORDER = ["chunking", "retrieval", "generation", "reranking"]

    def __init__(self, profile: KBProfile, evaluator: Evaluator, documents: List[Dict]):
        self.profile          = profile
        self.evaluator        = evaluator
        self.documents        = documents
        self.config_generator = KBAwareLLMModuleGenerator(profile)
        self.feedback_encoder = FeedbackEncoder()

    def optimize(self, pipeline: AdaptiveRAGPipeline) -> AdaptiveRAGPipeline:
        for module_type in self.MODULE_ORDER:
            print(f"\n[OptimizerAgent] Optimizing: {module_type}")
            best_config  = copy.deepcopy(pipeline.modules[module_type])
            best_metrics = self.evaluator.baseline_metrics
            feedback     = None

            for it in range(MAX_ITERATIONS_PER_MODULE):
                print(f"  Iteration {it + 1}/{MAX_ITERATIONS_PER_MODULE}")

                new_config = self.config_generator.generate_config(module_type, best_config, feedback)
                pipeline.update_module(module_type, new_config)

                if not pipeline.is_indexed:
                    pipeline.index_documents(self.documents)

                metrics = self.evaluator.evaluate(pipeline)
                print(f"  Score: {metrics.overall_score:.3f}  (best so far: {best_metrics.overall_score:.3f})")

                if metrics.overall_score > best_metrics.overall_score:
                    print(f"  ✅ Improvement accepted.")
                    best_config  = copy.deepcopy(new_config)
                    best_metrics = metrics
                    self.evaluator.baseline_metrics = metrics
                    break
                else:
                    print(f"  ❌ No improvement. Generating feedback...")
                    feedback = self.feedback_encoder.encode(
                        module_type, best_config, new_config, metrics, best_metrics
                    )
                    print(f"     → {feedback}")
                    pipeline.update_module(module_type, best_config)
                    if not pipeline.is_indexed:
                        pipeline.index_documents(self.documents)
            else:
                print(f"  Keeping best config after {MAX_ITERATIONS_PER_MODULE} iterations.")
                pipeline.update_module(module_type, best_config)

        return pipeline


# ======================== DATA LOADING ========================
def load_documents(folder: str) -> List[Dict]:
    docs = []
    for file in glob.glob(os.path.join(folder, "*.*")):
        content = ""
        if file.endswith(".txt"):
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        elif file.endswith(".pdf"):
            reader  = pypdf.PdfReader(file)
            content = "".join(page.extract_text() or "" for page in reader.pages)
        else:
            continue
        if content.strip():
            docs.append({"filename": os.path.basename(file), "content": content})
    return docs


def load_validation_set(path: str) -> List[Dict]:
    if not os.path.exists(path):
        sample = [{"query": "What is the main topic?", "expected_answer": "unknown"}]
        with open(path, "w") as f:
            json.dump(sample, f, indent=2)
        print(f"  Created sample validation set at {path}. Replace with real data.")
        return sample
    with open(path, "r") as f:
        return json.load(f)


# ======================== MAIN ORCHESTRATOR ========================
def run_adaptive_rag():
    """
    Full 5-agent pipeline:
      Load → Analyze → Plan → Index → Baseline → Optimize → Serve
    """
    banner = "=" * 62
    print(f"\n{banner}")
    print("   Adaptive Agentic RAG Framework")
    print(banner)

    # ── Load ─────────────────────────────────────────────────────────────
    print("\n[Loader] Loading knowledge base...")
    documents   = load_documents(KNOWLEDGE_BASE_PATH)
    val_queries = load_validation_set(VAL_QUERIES_PATH)
    if not documents:
        print(f"❌  No .txt or .pdf files found in '{KNOWLEDGE_BASE_PATH}'")
        return
    print(f"  ✓ {len(documents)} document(s) loaded")

    # ── Agent 1: Analyze ─────────────────────────────────────────────────
    profile = KBAnalyzerAgent().analyze(documents)

    # ── Agent 2: Plan ────────────────────────────────────────────────────
    planned_config = PlannerAgent().plan(profile)

    # Build initial pipeline from the plan
    modules = {k: v for k, v in planned_config.items() if k != "chunk_strategy"}
    modules["chunk_strategy"] = planned_config.get("chunk_strategy", "fixed")

    pipeline = AdaptiveRAGPipeline(modules, profile)

    # ── Agent 3: Index ───────────────────────────────────────────────────
    pipeline.index_documents(documents)

    # ── Baseline evaluation ──────────────────────────────────────────────
    evaluator        = Evaluator(val_queries, pipeline)
    baseline_score   = evaluator.baseline_metrics.overall_score
    print(f"\n[Evaluator] Baseline score: {baseline_score:.3f}")

    # ── Agent 4: Optimize ────────────────────────────────────────────────
    optimizer = OptimizerAgent(profile, evaluator, documents)
    pipeline  = optimizer.optimize(pipeline)

    # ── Final report ─────────────────────────────────────────────────────
    final = evaluator.evaluate(pipeline)
    delta = final.overall_score - baseline_score
    print(f"\n{banner}")
    print(f"  Final score : {final.overall_score:.3f}  (Δ {delta:+.3f})")
    print(f"  Accuracy    : {final.answer_accuracy:.3f}")
    print(f"  Avg latency : {final.latency_ms:.0f} ms")
    print(banner)

    # ── Agent 5: Serve with query routing ────────────────────────────────
    print("\n✅  Adaptive RAG ready.")
    print("    Commands:  'profile' → KB summary | 'config' → pipeline config | 'exit'\n")
    while True:
        try:
            q = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() == "exit":
            break
        if q.lower() == "profile":
            print(f"\n{profile.summary()}\n")
            continue
        if q.lower() == "config":
            print(json.dumps(
                {k: v for k, v in pipeline.modules.items() if k != "chunk_strategy"},
                indent=2
            ))
            continue
        answer = pipeline.answer(q, verbose=True)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    run_adaptive_rag()