from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import tool

from ..core.runtime import RT


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
    return RT.answer_llm.invoke(prompt).content.strip()


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
    ctx    = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))
    prompt = (
        f"{instructions.get(style, instructions['factual'])}\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
    )
    return RT.answer_llm.invoke(prompt).content.strip()


@tool
def decompose_query(query: str) -> List[str]:
    """Break a complex query into 2-4 simpler, focused sub-questions."""
    prompt = (
        f"Break this complex question into 2-4 simpler sub-questions that can each "
        f"be answered independently. Each sub-question should cover one specific aspect.\n\n"
        f"Question: {query}\n\n"
        f"Return ONLY a JSON array of strings:\n"
        f'["sub-question 1", "sub-question 2", ...]'
    )
    try:
        import json, re
        resp = RT.llm.invoke(prompt).content
        m    = re.search(r'\[.*\]', resp, re.DOTALL)
        if m:
            parts = json.loads(m.group())
            if isinstance(parts, list) and len(parts) >= 2:
                return [str(q).strip() for q in parts if str(q).strip()]
    except Exception:
        pass
    return [query]


@tool
def synthesize_answers(original_query: str, sub_answers: List[dict]) -> str:
    """Synthesize multiple sub-question answers into one coherent final answer."""
    if not sub_answers:
        return "I don't have enough information to answer that."

    qa_block = "\n\n".join(
        f"Q: {item['question']}\nA: {item['answer']}"
        for item in sub_answers
    )
    prompt = (
        f"You have answered several sub-questions to address a complex query. "
        f"Now write a single, cohesive answer to the original question by integrating "
        f"all the sub-answers below. Do not repeat each sub-question — just synthesize.\n\n"
        f"Original question: {original_query}\n\n"
        f"Sub-answers:\n{qa_block}\n\n"
        f"Synthesized answer:"
    )
    return RT.answer_llm.invoke(prompt).content.strip()
