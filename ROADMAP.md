# Roadmap — Adaptive Agentic RAG Framework

A staged plan to take this from a working prototype to a production-grade open source project. Each phase is a coherent milestone — you can stop after any of them and have something more valuable than where you started.

---

## Phase 0 — Pre-launch hygiene (1 week)

Before announcing the repo publicly, make sure it doesn't look abandoned-in-progress.

- **Pin dependencies** with `requirements.txt` exact versions, plus a `requirements-dev.txt` for tooling
- **Switch to `pyproject.toml`** with `hatch` or `pdm` so the project is installable: `pip install adaptive-agentic-rag`
- **Add `pre-commit`** with `ruff`, `black`, and `mypy` configured to catch obvious issues
- **Type-annotate everything** — the framework has rich state objects (`KBProfile`, `QueryState`) and missing types hurt usability
- **Split the monolith** — one file is fine for a demo, not for contributors. Move to:
  ```
  src/adaptive_rag/
    ├── agents/          # KBAnalyzer, Planner, Optimizer, Router, Strategist
    ├── components/      # Chunker, Embedder, Retriever, Reranker, Generator
    ├── graphs/          # setup_graph, query_graph builders
    ├── state.py         # TypedDict definitions
    ├── tools.py         # @tool decorated functions
    └── config.py        # Settings via pydantic-settings
  ```
- **Write a real README** with: 60-second quickstart, architecture diagram, one screenshot/gif of the REPL, badges (build, coverage, PyPI, license), and a clear "what this is / what this isn't" section
- **Pick a license deliberately** — Apache 2.0 matches the ecosystem (Gemma 4, LangChain, ChromaDB) and is friendliest to enterprise adoption
- **Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates, PR template** — Github expects these, contributors look for them

**Exit criteria:** A new developer can clone the repo, run `pip install -e ".[dev]"`, and have a working setup in under 5 minutes.

---

## Phase 1 — Reliability (2-3 weeks)

The single biggest gap between prototype and production. Right now if anything fails, the user sees a traceback.

### Tests
- **Unit tests** for every component — chunker strategies, embedder cosine, retriever filters, reranker fallback
- **Integration tests** for both graphs end-to-end with a fixture KB and a tiny embedded LLM (e.g. `tinyllama` via Ollama) running in CI
- **Snapshot tests** for graph paths — assert that "factual" queries take 4 nodes, "comparison" queries take 7, etc.
- **Property-based tests** with `hypothesis` for the chunker — chunks should never exceed `size * 1.5`, total chars should equal input, etc.
- **Target: 80% coverage** measured by `pytest-cov`, enforced in CI

### Error handling
- **Replace bare `except Exception`** with specific exceptions and proper logging
- **Add retry logic** with exponential backoff for LLM calls (use `tenacity`) — Ollama can hiccup, OpenAI rate-limits, etc.
- **Graceful degradation** — if HyDE expansion fails, fall back to the original query; if the reranker model can't load, fall back to length sort. Most of this exists but is inconsistent.
- **Circuit breaker** for repeated LLM failures — if 5 consecutive calls fail, return a clear error instead of grinding through retries

### Logging & observability
- **Replace all `print()` with `structlog`** — structured JSON logs are non-negotiable for production
- **Add `langsmith` or `langfuse` integration** — LangGraph has first-class tracing support, take advantage. Every node call, every LLM invocation, every tool call traced.
- **OpenTelemetry hooks** for users who want to plug into their own observability stack

**Exit criteria:** 1000 random queries against a 100-document KB without a single uncaught exception. CI green on every PR.

---

## Phase 2 — Provider abstraction (2 weeks)

Right now the code is tied to Ollama + Gemma 4 + ChromaDB + a specific cross-encoder. That's fine for a demo but kills adoption.

### LLM abstraction
- Define a `BaseLLM` protocol so users can swap in OpenAI, Anthropic, vLLM, llama.cpp, or any LangChain `ChatModel`
- Support **API-based providers** (OpenAI, Anthropic, Gemini, Cohere) and **local** (Ollama, llama.cpp, LM Studio)
- Make tool-calling vs JSON-mode a configuration choice — Gemma 4 has tool calling, smaller models don't

### Vector store abstraction
- Define a `BaseVectorStore` interface — `index`, `query`, `clear`, `count`, `where`
- Add adapters for **Qdrant** (production favorite), **Weaviate**, **pgvector** (for teams already on Postgres), **Pinecone** (managed)
- Keep ChromaDB as the default for the dev experience

### Embedder abstraction
- Support Ollama, OpenAI, Cohere, Voyage, and HuggingFace sentence-transformers
- **Critical for production:** consistent embedding model between indexing and querying. Add a model-hash to the manifest so the index detects when you've switched models and forces a re-index.

### Reranker abstraction
- Cohere Rerank, Jina Reranker, BGE Reranker, and the existing cross-encoder
- Reranking is expensive — make it conditional and configurable per-query (already partially done)

**Exit criteria:** A user can configure the entire stack via YAML or environment variables without touching code.

---

## Phase 3 — Persistence & state management (2 weeks)

The framework already has a checkpointer (`MemorySaver`) but it's in-memory and resets every run. For production:

- **Swap `MemorySaver` for `SqliteSaver` or `PostgresSaver`** — LangGraph supports both natively. Now you can pause/resume long optimization runs across server restarts.
- **Persist `KBProfile` and `PipelineConfig`** to disk — currently rebuilt every run, wasting 30+ seconds of LLM time
- **Conversation memory** — the query graph treats each query in isolation. Add a `thread_id` based memory layer so multi-turn conversations work. LangGraph's checkpointer does this if you organize state right.
- **Cache LLM calls** — the `QueryRewriter` and `QueryRouter` already have ad-hoc caches; replace with a proper cache layer (`langchain.cache` with SQLite or Redis backend)
- **Cache embedding calls** — hash the input text, store the embedding. Huge savings on re-indexing.

**Exit criteria:** A 10,000-document KB can be re-loaded and serving queries in under 5 seconds.

---

## Phase 4 — Concurrency & performance (3 weeks)

Single-user prototype → multi-user service.

- **Async everywhere** — convert all node functions to `async def`, use `asyncio.gather` in retrieval expansion, use async LangChain calls. LangGraph natively supports this.
- **Connection pooling** for vector stores and LLM clients
- **Batch embedding** at index time is already done; batch it at query time too when multiple variants need embeddings
- **Streaming responses** — LangGraph streams node outputs natively (`graph.astream`). Expose this through the API so clients see tokens as they arrive instead of waiting for the full answer.
- **GPU-aware reranker** — load the cross-encoder onto CUDA if available, fall back to CPU. Currently silent CPU-only.
- **Benchmark suite** — `pytest-benchmark` measuring p50/p95/p99 latency for each query type, regression-gated in CI

**Exit criteria:** A single instance handles 50 concurrent queries with p95 latency under 3 seconds (gated on a small KB).

---

## Phase 5 — Deployment & ops (2 weeks)

Make it deployable, not just runnable.

### Container
- **Multi-stage Dockerfile** — slim base, no dev dependencies in the final image, non-root user
- **`docker-compose.yml`** with the full stack: app + Ollama + Qdrant + Postgres for checkpointing + Redis for caching
- **Helm chart** for Kubernetes users

### API server
- **FastAPI wrapper** exposing the query graph as a REST API with proper OpenAPI docs
  - `POST /index` — async indexing job, returns job ID
  - `POST /query` — accepts query + optional thread_id, streams the response
  - `GET /profile` — returns the current KB profile
  - `GET /health` — for liveness/readiness probes
- **WebSocket endpoint** for streaming with backpressure
- **Auth middleware** — JWT or API keys, multi-tenant ready
- **Rate limiting** — per-key, per-IP, configurable

### Configuration
- **`pydantic-settings`** for all config — env vars, `.env` files, config classes with validation
- **Health checks** that probe Ollama, the vector store, and the embedder before reporting ready

**Exit criteria:** `docker compose up` brings up a production-ready service with one command.

---

## Phase 6 — Evaluation & quality (3 weeks)

The framework's "agentic" claim needs evidence. This phase is about proving the system is actually better than alternatives.

- **Benchmark against standard RAG datasets** — MS MARCO, BEIR, RAG-Bench, FinanceBench. Publish numbers in the README.
- **Add `ragas` integration** — faithfulness, answer relevance, context precision/recall as first-class metrics in the Evaluator
- **A/B testing harness** — the OptimizerAgent already chooses between configs based on scores. Generalize this so users can A/B test their own pipeline variants.
- **Failure mode catalog** — systematically find query types where the framework fails (multi-hop reasoning, table extraction, numerical comparison) and document them honestly
- **Reproducibility** — pin random seeds, log full configs, make every reported number reproducible from a single command

**Exit criteria:** Public benchmark page with numbers, methodology, and reproducible scripts.

---

## Phase 7 — Developer experience (2 weeks)

The features that turn passing visitors into committed users.

- **CLI** with `typer` — `arag index`, `arag query "..."`, `arag eval`, `arag inspect-trace <thread_id>`
- **Notebook examples** — at least 5 Jupyter notebooks for common use cases (legal docs, code repo, customer support, financial reports, scientific papers)
- **A web UI** with Streamlit or Gradio — drop documents, ask questions, see the agentic trace visualized. Killer for the README gif.
- **MCP server** — expose the framework as a Model Context Protocol server so Claude/Cursor/Cline can use it as a tool
- **VSCode extension** (optional, much later) — query your code KB from the editor

**Exit criteria:** New users can go from `pip install` to first useful answer in under 2 minutes.

---

## Phase 8 — Documentation (ongoing, but kicked off here)

The single most underrated factor in OSS success.

- **MkDocs Material site** hosted on GitHub Pages or Read the Docs
- **Architecture deep-dive** — explain what each agent does, why LangGraph, how to extend
- **How-to guides** — "Add a custom chunker", "Use OpenAI instead of Ollama", "Plug in pgvector", "Build a custom critic node"
- **Conceptual docs** — what is agentic RAG, what is HyDE, what is reranking, when to use multi-hop. Become an educational resource.
- **API reference** — auto-generated from docstrings with `mkdocstrings`
- **Versioned docs** — `mike` plugin so old docs stay accessible after breaking changes
- **A blog post per phase** completed — these become the marketing engine

**Exit criteria:** Someone can build a custom agent on top of the framework without ever reading the source code.

---

## Phase 9 — Community & governance (ongoing)

- **Discord or GitHub Discussions** — pick one, run it actively for the first 6 months
- **`GOVERNANCE.md`** — how decisions are made, who has merge rights, conflict resolution
- **Public roadmap** as a GitHub Project — this very document, but with cards and statuses
- **Weekly releases** for the first 3 months, then biweekly — momentum matters
- **Changelog** maintained religiously with `release-please` or `changesets`
- **Security policy** — `SECURITY.md`, a `security@` contact, GitHub security advisories enabled
- **Sponsor via GitHub Sponsors** — gives serious users a way to fund development

**Exit criteria:** At least 3 external contributors with merged PRs in a single month.

---

## Phase 10 — Differentiation features (long-term)

What makes this project worth using over LlamaIndex, Haystack, or rolling your own?

- **Self-improving loops** — log every (query, answer, user_feedback) triple, periodically re-run the OptimizerAgent against this real-world data
- **Multi-modal RAG** — Gemma 4 supports images and audio natively. Index PDFs as images alongside text. Index meeting recordings.
- **Agentic chunking** — let an LLM decide chunk boundaries based on semantic units, not character counts (research-grade but achievable)
- **Knowledge graph backbone** — extract entities and relations at index time, use graph traversal alongside vector retrieval
- **Multi-tenancy primitives** — namespace isolation, per-tenant configs, billing hooks
- **Cost tracking** — every LLM call logged with token counts and cost; show users their actual spend
- **Active learning** — when answer confidence is low, flag for human review and fold the human's answer back into the eval set

These are the features that justify the project's existence in a crowded space. Pick 2-3 to be your headline differentiators.

---

## Non-goals (worth being explicit about)

- **Don't build a general LangChain replacement** — LangGraph is enough orchestration. Stay focused on RAG.
- **Don't ship a frontend framework** — the Streamlit demo is a demo. Real frontends are out of scope.
- **Don't optimize for tiny models** — keep Gemma 4 as the floor. Below that, the agentic decisions degrade too much.
- **Don't build training infrastructure** — fine-tuning rerankers or embedders is a different project.

---

## Suggested order & realistic timeline

Solo developer working evenings/weekends:

| Phase | Calendar time |
|-------|---------------|
| 0 — hygiene | 1 week |
| 1 — reliability | 4 weeks |
| 2 — abstractions | 3 weeks |
| 3 — persistence | 3 weeks |
| 4 — concurrency | 5 weeks |
| 5 — deployment | 3 weeks |
| 6 — evaluation | 4 weeks |
| 7 — DX | 3 weeks |
| 8 — docs | overlapping with all phases |
| 9 — community | from phase 1 onward |
| 10 — differentiation | ongoing after launch |

**v1.0 milestone:** Phases 0-5 complete. ~5 months solo.
**Public launch:** End of phase 6, with a benchmark post. ~6 months.

Small team of 2-3 cuts this roughly in half.

---

## What to do this week

If this list feels overwhelming, the minimum viable set to start is:

1. Split the monolith file
2. Switch to `pyproject.toml`
3. Add 10 unit tests for the chunker and retriever
4. Set up GitHub Actions to run the tests
5. Replace `print` with `structlog` everywhere
6. Write a `CONTRIBUTING.md`

That's a weekend's work and the project will look serious to anyone who lands on the repo.