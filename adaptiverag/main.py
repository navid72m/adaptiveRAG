"""CLI entry point — invoked by `adaptiverag` after pip install."""
from __future__ import annotations

import time

from langchain_ollama import ChatOllama

from .core.config import LLM_MODEL
from .core.runtime import RT
from .components.embedder import Embedder
from .components.retriever import Retriever
from .components.reranker import Reranker
from .graphs.setup_graph import build_setup_graph
from .graphs.query_graph import build_query_graph


def main():
    print("=" * 70)
    print("  Agentic RAG Framework — LangGraph Edition")
    print("=" * 70)

    RT.embedder   = Embedder()
    RT.retriever  = Retriever(RT.embedder)
    RT.reranker   = Reranker()
    RT.llm        = ChatOllama(model=LLM_MODEL, temperature=0.0)
    RT.answer_llm = ChatOllama(model=LLM_MODEL, temperature=0.5)

    print("\n>>> Compiling setup graph...")
    setup_graph  = build_setup_graph()

    print(">>> Invoking setup graph...\n")
    final_state = setup_graph.invoke(
        {"iteration": 0, "optimizer_log": [], "notes": []},
        config={"configurable": {"thread_id": "setup-1"}},
    )

    print("\n" + "=" * 70)
    print(f"  Setup complete.")
    print(f"  Final config : {final_state['config']}")
    print(f"  Final score  : {final_state.get('best_score', 0.0):.3f}")
    print("=" * 70)

    print("\n>>> Compiling query graph...")
    query_graph = build_query_graph()

    print("\n✓  Agentic RAG ready.")
    print("    Commands:")
    print("      Any question              → routed through agentic graph")
    print("      'from:<file> <question>'  → restrict retrieval to one source")
    print("      'trace'                   → show the last query's full trace")
    print("      'exit'                    → quit\n")

    last_trace: list[str] = []
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
        if q.lower() == "trace":
            for step in last_trace:
                print(f"  - {step}")
            continue

        source_filter = None
        if q.lower().startswith("from:"):
            parts = q.split(" ", 1)
            if len(parts) == 2:
                source_filter, q = parts[0][5:], parts[1]

        result = query_graph.invoke(
            {
                "query":         q,
                "source_filter": source_filter,
                "retry_count":   0,
                "trace":         [],
            },
            config={"configurable": {"thread_id": f"query-{int(time.time())}"}},
        )
        last_trace = result.get("trace", [])
        print(f"\n{result.get('answer', '(no answer)')}\n")
        print(
            f"  [confidence={result.get('confidence', 0):.2f}  "
            f"retries={result.get('retry_count', 0)}  "
            f"strategy={result.get('strategy', {}).get('reason', '?')}]\n"
        )


if __name__ == "__main__":
    main()
