# AdaptiveRAG — Agentic RAG Framework

<p align="center">
  <img src="architecture.png" alt="AdaptiveRAG architecture diagram showing setup and query graphs" width="720"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/adaptiverag/"><img src="https://img.shields.io/pypi/v/adaptiverag?color=blue&label=PyPI" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/adaptiverag/"><img src="https://img.shields.io/pypi/pyversions/adaptiverag" alt="Python versions"/></a>
  <a href="https://github.com/navid72m/adaptiveRAG/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"/></a>
  <a href="https://ollama.com"><img src="https://img.shields.io/badge/embeddings-Ollama-black" alt="Embeddings via Ollama"/></a>
  <a href="https://www.anthropic.com"><img src="https://img.shields.io/badge/LLM-Claude-blueviolet" alt="Claude support"/></a>
  <a href="https://openai.com"><img src="https://img.shields.io/badge/LLM-OpenAI-412991" alt="OpenAI support"/></a>
  <a href="https://pepy.tech/projects/adaptiverag"><img src="https://static.pepy.tech/personalized-badge/adaptiverag?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="PyPI Downloads"/></a>
</p>

> **Self-optimising Retrieval-Augmented Generation built with LangGraph.**
> AdaptiveRAG analyses your knowledge base, auto-tunes the pipeline, and routes every query through the best retrieval strategy — works fully locally with Ollama or with Claude / OpenAI cloud APIs.

---

## Table of Contents

- [What makes it agentic](#what-makes-it-agentic)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [LLM providers](#llm-providers)
- [Multi-agent mode](#multi-agent-mode)
- [Configuration](#configuration)
- [Supported document formats](#supported-document-formats)
- [Validation queries](#validation-queries)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## What makes it agentic

Most RAG pipelines execute the same fixed sequence regardless of what you ask. AdaptiveRAG uses LLM-driven decision nodes at every step so the path through the graph changes per query and per knowledge base.

| Capability | Fixed RAG pipeline | AdaptiveRAG |
|---|---|---|
| Chunking strategy | Hard-coded | Chosen per document type (sentence / paragraph / code) |
| Chunk size | Fixed | Auto-tuned against your actual documents |
| Query expansion | None | HyDE (hypothetical document embedding) for vague queries |
| Retrieval passes | Single | Multi-hop follow-up when first pass is insufficient |
| Result reranking | None | Cross-encoder reranking for analytical / comparison queries |
| Answer quality | Not checked | Critic node scores the answer; retries with a new strategy if confidence is low |
| Parameter tuning | Manual | Optimizer agent tunes chunk size, top-k, temperature, and reranking automatically |
| LLM provider | Single hard-coded | Ollama (local), Claude, or OpenAI — swap with one parameter |

---

## How it works

AdaptiveRAG is composed of two LangGraph state machines.

### Setup graph — runs once at startup

```
load docs ──► profile KB ──► plan config ──► index ──► evaluate ──► orchestrate ──┐
                                                                         ▲          │
                                                                    critique ◄── tune_*
                                                                              (chunk / retrieval /
                                                                               generation / reranking)
```

1. **Profile** — the LLM classifies domain, structure type, and complexity of your documents
2. **Plan** — heuristic config is derived from the profile (chunk size, strategy, top-k, temperature)
3. **Index** — documents are chunked and embedded into ChromaDB
4. **Evaluate** — answers are scored against validation queries using cosine similarity
5. **Orchestrate** — the LLM picks which parameter to tune next and loops until scores plateau

### Query graph — runs for every question

```
classify ──► strategize ──► expand ──► retrieve ──► retrieval critic ──┐
    ▲                                                        │           │
    │                                                    multihop ◄──── ┘
    │                                                        │
    └──── retry ◄──── reflect ◄──── generate ◄──── rerank ◄─┘
```

1. **Classify** — query type detected (factual / analytical / code / comparison / summarisation)
2. **Strategize** — LLM decides which tools to use (HyDE, rerank, multihop, top-k)
3. **Retrieve** — vector search, optionally with expanded queries
4. **Critic** — retrieval quality is scored; if too low, a follow-up multi-hop query is issued
5. **Generate** — answer produced using the style matching the query type
6. **Reflect** — answer critic checks groundedness and completeness; retries if below threshold

---

## Installation

```bash
pip install adaptiverag
```

**With cross-encoder reranking** (recommended for analytical or comparison queries):

```bash
pip install "adaptiverag[reranker]"
```

**With Claude (Anthropic) support:**

```bash
pip install "adaptiverag[claude]"
```

**With OpenAI support:**

```bash
pip install "adaptiverag[openai]"
```

**With web search agent:**

```bash
pip install "adaptiverag[websearch]"
```

**Everything at once:**

```bash
pip install "adaptiverag[reranker,claude,openai,websearch]"
```

> **Prerequisite:** [Ollama](https://ollama.com/download) must be running locally — it is used for embeddings regardless of which LLM provider you choose.
> Any missing Ollama models are **pulled automatically** the first time `build_rag()` is called.

---

## Quick start

### 1. Add documents

Create a `knowledge_base/` folder and drop in your files (`.txt`, `.pdf`, `.md`, `.docx`):

```
knowledge_base/
├── report.pdf
├── notes.md
└── spec.txt
```

### 2. Run

```python
from adaptiverag import build_rag

# Indexes knowledge_base/, auto-tunes the pipeline, returns a ready instance
rag = build_rag()

result = rag.ask("What are the main findings?")
print(result)                # the answer (str(result) also works)
print(result.confidence)     # 0.0 – 1.0 self-assessed confidence
print(result.strategy)       # why the agent chose this retrieval path
print(result.trace)          # full step-by-step reasoning log
```

### 3. CLI

```bash
adaptiverag
```

Interactive prompt with the same agentic graph — type `trace` to see the last query's reasoning.

---

## LLM providers

AdaptiveRAG supports three LLM providers. **Embeddings always run locally via Ollama** regardless of which provider you pick.

### Ollama (default — fully local)

No API key needed. Any model from [ollama.com/library](https://ollama.com/library) works and is auto-pulled on first use.

```python
from adaptiverag import build_rag

rag = build_rag(
    llm_model   = "gemma4:latest",           # auto-pulled if not local
    embed_model = "nomic-embed-text:latest",
)
```

### Claude (Anthropic)

Get an API key at [console.anthropic.com](https://console.anthropic.com). Install the extra first:

```bash
pip install "adaptiverag[claude]"
```

```python
from adaptiverag import build_rag

rag = build_rag(
    api_key   = "sk-ant-...",          # defaults to claude-opus-4-7
    # llm_model = "claude-sonnet-4-6" # override the model if needed
)
```

### OpenAI

Get an API key at [platform.openai.com](https://platform.openai.com/api-keys). Install the extra first:

```bash
pip install "adaptiverag[openai]"
```

```python
from adaptiverag import build_rag

rag = build_rag(
    openai_api_key = "sk-...",    # defaults to gpt-4o
    # llm_model    = "gpt-4-turbo" # override the model if needed
)
```

### Provider comparison

| | Ollama | Claude | OpenAI |
|---|---|---|---|
| Install extra | — | `adaptiverag[claude]` | `adaptiverag[openai]` |
| Default model | `gemma4:latest` | `claude-opus-4-7` | `gpt-4o` |
| Internet required | No | Yes | Yes |
| Data leaves machine | No | Yes | Yes |
| Cost | Free | Pay-per-token | Pay-per-token |

---

## Multi-agent mode

AdaptiveRAG can operate as a **multi-agent system** where an LLM orchestrator dispatches each query to one or more specialised retrieval agents in parallel, merges their results, and synthesises a final answer.

```
Query
  │
  ▼
Orchestrator Agent  ──── decides which agents to activate ────►
  │                                                            │
  ├── Vector Agent A  ──►  ChromaDB (your knowledge base)     │
  ├── Web Agent B     ──►  DuckDuckGo web search              │
  └── External Agent C ──► Slack / Gmail / custom source      │
                                                               │
  ◄──────────────────── merge + rerank + synthesise ──────────┘
  │
  ▼
Response
```

### Enable web search

```bash
pip install "adaptiverag[websearch]"
```

```python
from adaptiverag import build_rag

rag = build_rag(enable_web_search=True)

# The orchestrator will automatically add web search for queries about
# current events or information unlikely to be in your documents.
result = rag.ask("What are the latest developments in this field?")
```

### Add a custom agent (Slack, Gmail, Notion, …)

Subclass `ExternalDataAgent` (or `BaseRetrievalAgent`) and pass it via `extra_agents`:

```python
from adaptiverag import build_rag
from adaptiverag.agents import ExternalDataAgent

class SlackAgent(ExternalDataAgent):
    name        = "slack"
    description = "Searches Slack messages and channels."

    def retrieve(self, query, top_k=5, source_filter=None):
        # call your Slack API here
        return ["message snippet 1 ...", "message snippet 2 ..."]

rag = build_rag(extra_agents=[SlackAgent()])
result = rag.ask("Any Slack messages about the deployment issue?")
```

### Combine everything

```python
rag = build_rag(
    api_key           = "sk-ant-...",   # Claude as the LLM
    enable_web_search = True,           # DuckDuckGo agent
    extra_agents      = [SlackAgent()], # your custom agent
)
```

### How the orchestrator decides

The routing LLM receives the query, its classified type, and a description of every available agent. It returns a list of agent names to activate — for example:

| Query | Agents activated |
|---|---|
| "What does the spec say about X?" | `vector_search` |
| "What happened with X last week?" | `vector_search`, `web_search` |
| "Any emails about the X deadline?" | `vector_search`, `external_data` |

On retry (low answer confidence), the orchestrator automatically broadens its agent selection.

---

## Configuration

```python
rag = build_rag(
    llm_model        = "gemma4:latest",              # Ollama model (auto-pulled)
    embed_model      = "nomic-embed-text:latest",    # Ollama embedding model (auto-pulled)
    kb_path          = "./knowledge_base",           # path to your documents
    val_queries_path = "./validation_queries.json",  # optional — auto-generated if omitted
    api_key          = None,                         # Anthropic key → uses Claude
    openai_api_key   = None,                         # OpenAI key → uses OpenAI
)
```

### Restrict retrieval to a single source file

```python
# keyword prefix
result = rag.ask("from:report.pdf Summarise the methodology")

# or the parameter
result = rag.ask("Summarise the methodology", source_filter="report.pdf")
```

### Supported Ollama models

| Role | Recommended models |
|---|---|
| LLM (routing + answers) | `gemma4`, `llama3.2`, `mistral`, `qwen2.5` |
| Embeddings | `nomic-embed-text`, `mxbai-embed-large` |

---

## Supported document formats

| Format | Extension | Notes |
|---|---|---|
| Plain text | `.txt` | UTF-8 |
| PDF | `.pdf` | Text-based; scanned PDFs not supported |
| Markdown | `.md` | Code blocks, headings, and links stripped cleanly |
| Word | `.docx` | Requires `python-docx` (included) |

Mixed formats in the same folder are fully supported.

---

## Validation queries

The setup graph tunes pipeline parameters by scoring generated answers against expected answers. Provide your own queries for best results:

```json
[
  {
    "query": "What problem does this research solve?",
    "expected_answer": "The research addresses the challenge of ..."
  },
  {
    "query": "What method is used for data collection?",
    "expected_answer": "Data was collected through ..."
  }
]
```

Pass the path via `val_queries_path`. If you omit it:
- AdaptiveRAG checks for `./validation_queries.json`
- If not found, the LLM **auto-generates** queries from your documents and saves them to that path
- You can then open the file, edit or extend the queries, and they will be used on the next run

---

## API reference

### `build_rag(...) → AdaptiveRAG`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `llm_model` | `str` | `"gemma4:latest"` | Model for routing and answer generation |
| `embed_model` | `str` | `"nomic-embed-text:latest"` | Ollama model for embeddings (always local) |
| `kb_path` | `str \| None` | `"./knowledge_base"` | Folder containing your documents |
| `val_queries_path` | `str \| None` | `"./validation_queries.json"` | Validation Q&A file (auto-generated if missing) |
| `api_key` | `str \| None` | `None` | Anthropic API key — enables Claude as the LLM |
| `openai_api_key` | `str \| None` | `None` | OpenAI API key — enables OpenAI as the LLM |
| `enable_web_search` | `bool` | `False` | Add DuckDuckGo web-search agent (requires `adaptiverag[websearch]`) |
| `extra_agents` | `list \| None` | `None` | Additional `BaseRetrievalAgent` instances (Slack, Gmail, etc.) |

### `AdaptiveRAG.ask(question, source_filter=None) → QueryResult`

| Parameter | Type | Description |
|---|---|---|
| `question` | `str` | Natural-language question. Prefix with `from:<file>` to filter by source. |
| `source_filter` | `str \| None` | Restrict retrieval to a single filename |

### `QueryResult` fields

| Field | Type | Description |
|---|---|---|
| `answer` | `str` | The generated answer (`str(result)` also works) |
| `confidence` | `float` | Self-assessed confidence, 0.0 – 1.0 |
| `retries` | `int` | Number of reflection retries used |
| `strategy` | `str` | One-line explanation of the retrieval strategy chosen |
| `trace` | `list[str]` | Complete step-by-step decision log |

---

## Project structure

```
adaptiverag/
├── core/
│   ├── config.py              # constants and defaults
│   ├── models.py              # KBProfile, PipelineConfig dataclasses
│   └── runtime.py             # shared runtime singleton (RT)
├── components/
│   ├── chunker.py             # content-aware chunking strategies
│   ├── embedder.py            # Ollama embedding wrapper
│   ├── retriever.py           # ChromaDB retrieval
│   └── reranker.py            # cross-encoder reranking (optional)
├── pipeline/
│   ├── tools.py               # LangChain tools (retrieve, rerank, HyDE, decompose, synthesize)
│   ├── kb_analysis.py         # KB profiling and heuristic config planning
│   └── file_loader.py         # document loading (.txt, .pdf, .md, .docx)
├── agents/
│   ├── base.py                # BaseRetrievalAgent abstract class
│   ├── vector_agent.py        # Agent A — ChromaDB vector search
│   ├── web_agent.py           # Agent B — DuckDuckGo web search
│   └── external_agent.py      # Agent C — pluggable stub (Slack, Gmail, …)
├── graphs/
│   ├── setup_graph.py         # build-time LangGraph agent (index + optimise)
│   ├── query_graph.py         # single-agent query graph (default)
│   └── multi_agent_graph.py   # multi-agent orchestrator graph
├── api.py                     # public Python API (build_rag, AdaptiveRAG, QueryResult)
└── main.py                    # CLI entry point
```

---

## Requirements

- Python ≥ 3.10
- [Ollama](https://ollama.com/download) running at `http://localhost:11434` (for embeddings)
- Core dependencies (installed automatically via pip):

```
langgraph>=0.2          langchain-core>=0.3
langchain-ollama>=0.2   langchain-community>=0.3
chromadb>=0.5           pypdf>=4.0
python-docx>=1.1        numpy>=1.26
tqdm>=4.66
```

- Optional extras:

| Extra | Installs | Enables |
|---|---|---|
| `adaptiverag[reranker]` | `sentence-transformers` | Cross-encoder reranking |
| `adaptiverag[claude]` | `langchain-anthropic` | Claude LLM provider |
| `adaptiverag[openai]` | `langchain-openai` | OpenAI LLM provider |
| `adaptiverag[websearch]` | `duckduckgo-search` | Web search agent |
| `adaptiverag[dev]` | `pytest`, `ruff` | Development tools |

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

```bash
git clone https://github.com/navid72m/adaptiveRAG.git
cd adaptiveRAG
pip install -e ".[dev]"
```

---

## License

[MIT](LICENSE) © navid72m
