from __future__ import annotations

import json
import re
from typing import Annotated, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..core.config import CONFIDENCE_THRESHOLD, RETRIEVAL_THRESHOLD, MAX_QUERY_RETRIES
from ..core.runtime import RT
from ..pipeline.tools import (
    retrieve_documents, rerank_documents, expand_query_hyde,
    generate_answer, decompose_query, synthesize_answers,
)


# ── State ──────────────────────────────────────────────────────────────────
class QueryState(TypedDict, total=False):
    query:                str
    source_filter:        Optional[str]
    query_type:           str
    strategy:             Dict
    expansions:           List[str]
    sub_queries:          List[str]
    sub_answers:          List[Dict]
    retrieved_docs:       List[str]
    retrieval_confidence: float
    reranked_docs:        List[str]
    answer:               str
    confidence:           float
    reflection:           str
    retry_count:          int
    trace:                Annotated[List[str], lambda a, b: a + b]


# ── Nodes ──────────────────────────────────────────────────────────────────
def node_classify_query(state: QueryState) -> Dict:
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
    retries         = state.get("retry_count", 0)
    prev_reflection = state.get("reflection", "")
    qtype           = state.get("query_type", "factual")
    profile         = RT.profile.to_dict() if RT.profile else {}

    retry_hint = ""
    if retries > 0:
        retry_hint = (
            f"\nThis is retry #{retries}. Previous attempt feedback:\n"
            f"  {prev_reflection}\nChange the strategy."
        )

    prompt = f"""You are a RAG query strategist. Decide which tools to use for this query.

Query: "{state['query']}"
Type: {qtype}
KB Profile: {json.dumps(profile)}{retry_hint}

Return ONLY JSON:
{{
  "use_hyde": <true|false>,
  "use_decompose": <true|false>,
  "use_multihop": <true|false>,
  "use_rerank": <true|false>,
  "top_k": <integer 3-15>,
  "reason": "<one-sentence justification>"
}}

Guidelines:
- Factual short queries → minimal tools, no decompose
- Multi-part or compound questions (contains "and", "also", "as well as", multiple "?") → use_decompose
- Analytical/comparison queries → use rerank, often multihop
- Vague queries → use hyde to expand
- Code queries → higher top_k, no rerank
- use_decompose and use_hyde are mutually exclusive — pick one or neither"""

    try:
        resp     = RT.llm.invoke(prompt).content
        m        = re.search(r'\{.*\}', resp, re.DOTALL)
        strategy = json.loads(m.group()) if m else {
            "use_hyde": False, "use_multihop": False, "use_rerank": False,
            "top_k": RT.config.top_k, "reason": "fallback",
        }
    except Exception as e:
        strategy = {
            "use_hyde": False, "use_multihop": False, "use_rerank": False,
            "top_k": RT.config.top_k, "reason": f"error: {e}",
        }

    return {"strategy": strategy, "trace": [f"strategy: {strategy}"]}


def node_decompose_query(state: QueryState) -> Dict:
    if not state.get("strategy", {}).get("use_decompose"):
        return {"sub_queries": [], "sub_answers": []}

    sub_qs = decompose_query.invoke({"query": state["query"]})
    if len(sub_qs) <= 1:
        return {"sub_queries": [], "sub_answers": [],
                "trace": ["decompose: query not complex enough, skipped"]}

    src    = state.get("source_filter")
    top_k  = max(state.get("strategy", {}).get("top_k", 5) // 2, 2)
    pairs  = []
    for q in sub_qs:
        docs = retrieve_documents.invoke({"query": q, "top_k": top_k, "source_filter": src})
        ans  = generate_answer.invoke({"query": q, "context": docs, "style": "factual"})
        pairs.append({"question": q, "answer": ans})

    return {
        "sub_queries": sub_qs,
        "sub_answers": pairs,
        "trace": [f"decomposed into {len(sub_qs)} sub-questions: {sub_qs}"],
    }


def node_expand_query(state: QueryState) -> Dict:
    if not state.get("strategy", {}).get("use_hyde"):
        return {"expansions": [state["query"]]}
    hypothetical = expand_query_hyde.invoke({"query": state["query"]})
    return {
        "expansions": [state["query"], hypothetical],
        "trace": [f"hyde expansion added ({len(hypothetical)} chars)"],
    }


def node_retrieve(state: QueryState) -> Dict:
    queries = state.get("expansions") or [state["query"]]
    top_k   = state.get("strategy", {}).get("top_k", RT.config.top_k)
    src     = state.get("source_filter")

    seen, docs = set(), []
    for q in queries:
        for d in retrieve_documents.invoke(
            {"query": q, "top_k": top_k, "source_filter": src}
        ):
            if d not in seen:
                seen.add(d)
                docs.append(d)
    return {"retrieved_docs": docs, "trace": [f"retrieved {len(docs)} unique docs"]}


def node_retrieval_critic(state: QueryState) -> Dict:
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
        resp  = RT.llm.invoke(prompt).content
        m     = re.search(r'\{.*\}', resp, re.DOTALL)
        score = float(json.loads(m.group()).get("score", 0.5)) if m else 0.5
    except Exception:
        score = 0.5
    return {
        "retrieval_confidence": score,
        "trace": [f"retrieval confidence: {score:.2f}"],
    }


def node_multihop_retrieve(state: QueryState) -> Dict:
    docs    = state.get("retrieved_docs", [])
    preview = "\n".join(docs[:3])[:1500]
    sub_prompt = (
        f"Based on these partial results, what FOLLOW-UP question would "
        f"retrieve the missing information for: '{state['query']}'?\n\n"
        f"Context so far:\n{preview}\n\nFollow-up question (one sentence):"
    )
    try:
        sub_q = RT.llm.invoke(sub_prompt).content.strip()
        more  = retrieve_documents.invoke({
            "query": sub_q,
            "top_k": max(RT.config.top_k // 2, 2),
            "source_filter": state.get("source_filter"),
        })
        existing = set(docs)
        added    = [d for d in more if d not in existing]
        return {
            "retrieved_docs": docs + added,
            "trace": [f"multihop added {len(added)} docs via: {sub_q[:80]}..."],
        }
    except Exception as e:
        return {"trace": [f"multihop failed: {e}"]}


def node_rerank(state: QueryState) -> Dict:
    if not state.get("strategy", {}).get("use_rerank"):
        return {"reranked_docs": state.get("retrieved_docs", [])}
    docs   = state.get("retrieved_docs", [])
    ranked = rerank_documents.invoke({"query": state["query"], "docs": docs})
    return {"reranked_docs": ranked, "trace": ["reranked via cross-encoder"]}


def node_generate(state: QueryState) -> Dict:
    sub_answers = state.get("sub_answers")
    if sub_answers:
        answer = synthesize_answers.invoke({
            "original_query": state["query"],
            "sub_answers":    sub_answers,
        })
        return {"answer": answer, "trace": [f"synthesized from {len(sub_answers)} sub-answers ({len(answer)} chars)"]}

    docs   = state.get("reranked_docs") or state.get("retrieved_docs", [])
    answer = generate_answer.invoke({
        "query":   state["query"],
        "context": docs,
        "style":   state.get("query_type", "factual"),
    })
    return {"answer": answer, "trace": [f"generated {len(answer)} chars"]}


def node_answer_critic(state: QueryState) -> Dict:
    answer = state.get("answer", "")
    docs   = state.get("reranked_docs") or state.get("retrieved_docs", [])
    if not answer or not docs:
        return {"confidence": 0.0, "reflection": "no answer or no context"}

    ctx    = "\n".join(f"[{i+1}] {d[:200]}..." for i, d in enumerate(docs[:5]))
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
        m    = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            data       = json.loads(m.group())
            conf       = float(data.get("confidence", 0.5))
            reflection = (
                f"grounded={data.get('grounded')}, "
                f"complete={data.get('complete')}, "
                f"issues={data.get('issues')}"
            )
        else:
            conf, reflection = 0.5, "no parse"
    except Exception as e:
        conf, reflection = 0.5, f"error: {e}"

    return {
        "confidence": conf,
        "reflection": reflection,
        "trace": [f"answer confidence: {conf:.2f} ({reflection})"],
    }


def node_prepare_retry(state: QueryState) -> Dict:
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "trace": [f"retry #{state.get('retry_count', 0) + 1} (conf was low)"],
    }


# ── Routers ────────────────────────────────────────────────────────────────
def route_after_retrieval(state: QueryState) -> str:
    if (state.get("retrieval_confidence", 0.0) < RETRIEVAL_THRESHOLD
            and state.get("strategy", {}).get("use_multihop")):
        return "multihop"
    return "rerank"


def route_after_reflection(state: QueryState) -> str:
    if (state.get("confidence", 1.0) < CONFIDENCE_THRESHOLD
            and state.get("retry_count", 0) < MAX_QUERY_RETRIES):
        return "retry"
    return "finish"


# ── Graph builder ──────────────────────────────────────────────────────────
def build_query_graph():
    g = StateGraph(QueryState)
    g.add_node("classify",         node_classify_query)
    g.add_node("strategize",       node_strategy_orchestrator)
    g.add_node("decompose",        node_decompose_query)
    g.add_node("expand",           node_expand_query)
    g.add_node("retrieve",         node_retrieve)
    g.add_node("retrieval_critic", node_retrieval_critic)
    g.add_node("multihop",         node_multihop_retrieve)
    g.add_node("rerank",           node_rerank)
    g.add_node("generate",         node_generate)
    g.add_node("reflect",          node_answer_critic)
    g.add_node("retry",            node_prepare_retry)

    g.add_edge(START,        "classify")
    g.add_edge("classify",   "strategize")
    g.add_edge("strategize", "decompose")
    g.add_edge("decompose",  "expand")
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
    g.add_edge("retry", "strategize")

    return g.compile(checkpointer=MemorySaver())
