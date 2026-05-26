"""
Multi-agent query graph.

Each incoming query is routed by an LLM orchestrator to one or more
specialised retrieval agents (vector, web, external).  Results from all
active agents are merged, optionally reranked, then passed to the answer
LLM.  The same reflect / retry loop as the single-agent graph applies.

Graph
-----
classify → route_agents → dispatch → merge → rerank → generate → reflect
    ▲                                                       |
    └──────────── retry ◄──── (confidence < threshold) ◄───┘
"""
from __future__ import annotations

import json
import re
import time
from typing import Annotated, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..core.config import CONFIDENCE_THRESHOLD, MAX_QUERY_RETRIES
from ..core.runtime import RT
from ..pipeline.tools import generate_answer, rerank_documents, synthesize_answers


# ── State ──────────────────────────────────────────────────────────────────
class MultiAgentState(TypedDict, total=False):
    query:          str
    source_filter:  Optional[str]
    query_type:     str
    active_agents:  List[str]
    agent_results:  Dict[str, List[str]]   # agent_name → docs
    merged_docs:    List[str]
    reranked_docs:  List[str]
    answer:         str
    confidence:     float
    reflection:     str
    retry_count:    int
    trace:          Annotated[List[str], lambda a, b: a + b]


# ── Nodes ──────────────────────────────────────────────────────────────────
def node_classify_query(state: MultiAgentState) -> Dict:
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
    return {"query_type": qt, "trace": [f"[orchestrator] classified: {qt}"]}


def node_route_agents(state: MultiAgentState) -> Dict:
    available  = [a for a in RT.agents if a.is_available()]
    agent_desc = "\n".join(f"  - {a.name}: {a.description}" for a in available)

    retries      = state.get("retry_count", 0)
    retry_hint   = ""
    if retries > 0:
        retry_hint = (
            f"\nPrevious attempt had low confidence: {state.get('reflection', '')}."
            f"\nTry activating additional agents to broaden the search."
        )

    prompt = f"""You are a multi-agent RAG orchestrator. Decide which retrieval agents to activate.

Query: "{state['query']}"
Type: {state.get('query_type', 'factual')}{retry_hint}

Available agents:
{agent_desc}

Return ONLY JSON:
{{
  "agents": ["agent_name_1", "agent_name_2"],
  "reason": "<one sentence>"
}}

Guidelines:
- Always include "vector_search" — it is the primary knowledge base.
- Add "web_search" when the query asks about current events, recent news, real-time data,
  or anything unlikely to be in a static document collection.
- Add "external_data" when the query explicitly mentions emails, Slack messages,
  notifications, or recent communications.
- For straightforward factual queries answered by the KB, use only "vector_search"."""

    available_names = {a.name for a in available}
    try:
        resp   = RT.llm.invoke(prompt).content
        m      = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            data   = json.loads(m.group())
            agents = [a for a in data.get("agents", []) if a in available_names]
            reason = data.get("reason", "")
        else:
            agents, reason = ["vector_search"], "fallback"
    except Exception as e:
        agents, reason = ["vector_search"], f"error: {e}"

    if not agents:
        agents = ["vector_search"]

    return {
        "active_agents": agents,
        "trace": [f"[orchestrator] routing to: {agents} — {reason}"],
    }


def node_dispatch_agents(state: MultiAgentState) -> Dict:
    active   = state.get("active_agents", ["vector_search"])
    top_k    = RT.config.top_k if RT.config else 5
    src      = state.get("source_filter")
    agent_map = {a.name: a for a in RT.agents}

    results: Dict[str, List[str]] = {}
    for name in active:
        agent = agent_map.get(name)
        if not agent:
            continue
        try:
            docs = agent.retrieve(state["query"], top_k=top_k, source_filter=src)
            results[name] = docs
            print(f"  [{name}] retrieved {len(docs)} doc(s)")
        except Exception as e:
            results[name] = []
            print(f"  ✗ [{name}] failed: {e}")

    return {
        "agent_results": results,
        "trace": [
            f"[dispatch] " +
            ", ".join(f"{k}={len(v)} docs" for k, v in results.items())
        ],
    }


def node_merge_results(state: MultiAgentState) -> Dict:
    all_results = state.get("agent_results", {})
    seen: set   = set()
    merged: List[str] = []

    # Interleave results from all agents (round-robin) so no single agent dominates.
    agent_lists = list(all_results.values())
    max_len = max((len(lst) for lst in agent_lists), default=0)
    for i in range(max_len):
        for lst in agent_lists:
            if i < len(lst) and lst[i] not in seen:
                seen.add(lst[i])
                merged.append(lst[i])

    return {
        "merged_docs": merged,
        "trace": [f"[merge] {len(merged)} unique docs from {len(all_results)} agent(s)"],
    }


def node_rerank(state: MultiAgentState) -> Dict:
    docs = state.get("merged_docs", [])
    if not docs:
        return {"reranked_docs": []}
    try:
        ranked = rerank_documents.invoke({"query": state["query"], "docs": docs})
        return {"reranked_docs": ranked, "trace": ["[rerank] cross-encoder applied"]}
    except Exception:
        return {"reranked_docs": docs, "trace": ["[rerank] skipped (reranker unavailable)"]}


def node_generate(state: MultiAgentState) -> Dict:
    docs   = state.get("reranked_docs") or state.get("merged_docs", [])
    answer = generate_answer.invoke({
        "query":   state["query"],
        "context": docs,
        "style":   state.get("query_type", "factual"),
    })
    return {"answer": answer, "trace": [f"[generate] {len(answer)} chars"]}


def node_answer_critic(state: MultiAgentState) -> Dict:
    answer = state.get("answer", "")
    docs   = state.get("reranked_docs") or state.get("merged_docs", [])
    if not answer or not docs:
        return {"confidence": 0.0, "reflection": "no answer or no context"}

    ctx    = "\n".join(f"[{i+1}] {d[:200]}..." for i, d in enumerate(docs[:5]))
    prompt = f"""Evaluate this RAG answer.

Query: {state['query']}
Answer: {answer}
Context (truncated):
{ctx}

Return ONLY JSON:
{{
  "confidence": <0.0-1.0>,
  "grounded": <true|false>,
  "complete": <true|false>,
  "issues": "<comma-separated or 'none'>"
}}"""

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
        "trace": [f"[critic] confidence={conf:.2f} ({reflection})"],
    }


def node_prepare_retry(state: MultiAgentState) -> Dict:
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "trace": [f"[retry] attempt #{state.get('retry_count', 0) + 1}"],
    }


# ── Router ─────────────────────────────────────────────────────────────────
def route_after_reflection(state: MultiAgentState) -> str:
    if (state.get("confidence", 1.0) < CONFIDENCE_THRESHOLD
            and state.get("retry_count", 0) < MAX_QUERY_RETRIES):
        return "retry"
    return "finish"


# ── Graph builder ──────────────────────────────────────────────────────────
def build_multi_agent_graph():
    g = StateGraph(MultiAgentState)

    g.add_node("classify",     node_classify_query)
    g.add_node("route_agents", node_route_agents)
    g.add_node("dispatch",     node_dispatch_agents)
    g.add_node("merge",        node_merge_results)
    g.add_node("rerank",       node_rerank)
    g.add_node("generate",     node_generate)
    g.add_node("reflect",      node_answer_critic)
    g.add_node("retry",        node_prepare_retry)

    g.add_edge(START,          "classify")
    g.add_edge("classify",     "route_agents")
    g.add_edge("route_agents", "dispatch")
    g.add_edge("dispatch",     "merge")
    g.add_edge("merge",        "rerank")
    g.add_edge("rerank",       "generate")
    g.add_edge("generate",     "reflect")

    g.add_conditional_edges(
        "reflect",
        route_after_reflection,
        {"retry": "retry", "finish": END},
    )
    g.add_edge("retry", "route_agents")  # re-route on retry (may activate more agents)

    return g.compile(checkpointer=MemorySaver())
