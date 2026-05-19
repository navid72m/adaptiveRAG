# adaptiveRAG

![PyPI](https://img.shields.io/pypi/v/adaptiverag)
![Python Versions](https://img.shields.io/pypi/pyversions/adaptiverag)
![Downloads](https://static.pepy.tech/badge/adaptiverag)
![License](https://img.shields.io/github/license/navid72m/adaptiveRAG)
![Stars](https://img.shields.io/github/stars/navid72m/adaptiveRAG)

**Agentic RAG that thinks before it retrieves.**

`adaptiverag` is a fully local, self-optimising Retrieval-Augmented Generation framework built on [LangGraph](https://github.com/langchain-ai/langgraph) and [Ollama](https://ollama.com). It runs two autonomous agent graphs — one that analyses and indexes your knowledge base at startup, and one that routes every query through the best possible retrieval strategy at runtime.

---

## Why adaptiverag?

Most RAG pipelines run the same fixed sequence for every query. `adaptiverag` treats retrieval as a decision problem:

| Fixed pipeline | adaptiverag |
|---|---|
| Same chunk size for all docs | Analyses doc structure, picks chunk strategy automatically |
| Fixed top-k for every query | LLM-chosen top-k per query based on type and complexity |
| No query expansion | Uses HyDE expansion for vague queries |
| Single-pass retrieval | Multi-hop follow-up retrieval when first pass is insufficient |
| No quality check | Critic node scores the answer; retries with a new strategy if confidence is low |
| Manual parameter tuning | Optimizer agent tunes chunk size, top-k, temperature, and reranking automatically |

---

## How it works

### Setup graph (runs once at startup)

```
load docs → profile KB → plan config → index → evaluate → orchestrate ──┐
                                                              ↑           │
                                                         critique ←── tune_*
```

The orchestrator LLM analyses the knowledge base profile and current scores, then decides which parameter to tune next. It loops until scores stop improving or the iteration budget runs out.

### Query graph (runs per query)

```
classify → strategize → expand → retrieve → retrieval_critic ──┐
                ↑                                    ↓          │
              retry ← reflect ← generate ← rerank ← multihop ──┘
```

The strategist LLM picks tools (HyDE, multi-hop, reranker) based on query type. The answer critic scores the result and routes back for a retry if confidence is below the threshold.

---

## Installation

```bash
pip install adaptiverag

# Optional: cross-encoder reranking (improves precision on complex queries)
pip install "adaptiverag[reranker]"
```

Requires [Ollama](https://ollama.com/download) running locally. Any missing models are **pulled automatically** on first run — no manual `ollama pull` needed.

---

## Quick start

```python
from adaptiverag import build_rag

# Indexes ./knowledge_base, auto-tunes the pipeline, returns a ready instance
rag = build_rag()

result = rag.ask("What are the main findings?")
print(result)                  # prints the answer
print(result.confidence)       # 0.0 – 1.0
print(result.strategy)         # one-line explanation of what the agent chose
print(result.trace)            # full step-by-step reasoning trace
```

### Custom paths and models

```python
rag = build_rag(
    llm_model        = "llama3.2:latest",           # any Ollama model
    embed_model      = "nomic-embed-text:latest",
    kb_path          = "/path/to/your/documents",
    val_queries_path = "/path/to/validation.json",  # optional — auto-generated if omitted
)
```

### Restrict retrieval to one file

```python
result = rag.ask("Summarise the methodology", source_filter="paper.pdf")
# or using the inline prefix:
result = rag.ask("from:paper.pdf Summarise the methodology")
```

### CLI

```bash
adaptiverag
```

---

## Supported document formats

| Format | Extension |
|---|---|
| Plain text | `.txt` |
| PDF | `.pdf` |
| Markdown | `.md` |
| Word | `.docx` |

Drop files into your `knowledge_base/` folder. Mixed formats are supported.

---

## Validation queries

The optimizer tunes pipeline parameters by scoring answers against expected answers. You can provide your own:

```json
[
  {
    "query": "What problem does this paper solve?",
    "expected_answer": "The paper addresses the challenge of ..."
  }
]
```

Pass the path via `val_queries_path`. If you omit it, `adaptiverag` generates queries automatically from your documents using the LLM and saves them to `./validation_queries.json` for you to review and edit.

---

## QueryResult fields

| Field | Type | Description |
|---|---|---|
| `answer` | `str` | The generated answer (`str(result)` also works) |
| `confidence` | `float` | Self-assessed confidence, 0.0 – 1.0 |
| `retries` | `int` | Number of reflection retries needed |
| `strategy` | `str` | One-line explanation of the agent's retrieval strategy |
| `trace` | `list[str]` | Step-by-step log of every decision made |

---

## Requirements

- Python ≥ 3.10
- [Ollama](https://ollama.com/download) running locally (`http://localhost:11434`)
- At least one chat model and one embedding model available in Ollama (auto-pulled if missing)

---

## License

MIT
