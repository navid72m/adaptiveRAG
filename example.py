"""
example.py — demonstrates the adaptiverag public API.

Prerequisites
-------------
1. Ollama running locally with the model pulled:
       ollama pull gemma4
       ollama pull nomic-embed-text

2. A knowledge_base/ folder in the same directory as this script,
   containing at least one .txt / .pdf / .md / .docx file.

Run
---
    python example.py
"""

from adaptiverag import build_rag

# ── 1. Build — loads KB, indexes, and auto-tunes the pipeline ─────────────
print("Initialising AdaptiveRAG (this may take a minute on first run)...\n")
rag = build_rag(
    llm_model="gemma4:latest",                        # any Ollama model tag
    kb_path="./knowledge_base",                       # folder with your documents
    val_queries_path="./validation_queries.json",     # optional — auto-created if missing
)

# ── 2. Simple factual question ────────────────────────────────────────────
result = rag.ask("What is the main topic covered in the documents?")
print("Question : What is the main topic covered in the documents?")
print(f"Answer   : {result.answer}")
print(f"Confidence: {result.confidence:.2f}  |  Retries: {result.retries}")
print(f"Strategy : {result.strategy}\n")

# ── 3. Analytical question (triggers rerank + multihop) ──────────────────
result = rag.ask("Why is this topic important? Explain the key implications.")
print("Question : Why is this topic important?")
print(f"Answer   : {result.answer}")
print(f"Confidence: {result.confidence:.2f}  |  Retries: {result.retries}\n")

# ── 4. Source-filtered question ───────────────────────────────────────────
# Replace "report.pdf" with an actual filename in your knowledge_base/
result = rag.ask("Summarize the findings", source_filter="report.pdf")
print("Question : Summarize the findings  [source: report.pdf]")
print(f"Answer   : {result.answer}\n")

# Equivalent using the inline prefix syntax:
# result = rag.ask("from:report.pdf Summarize the findings")

# ── 5. Inspect the reasoning trace ───────────────────────────────────────
result = rag.ask("Compare and contrast the approaches described.")
print("Question : Compare and contrast the approaches described.")
print(f"Answer   : {result.answer}\n")
print("Trace:")
for step in result.trace:
    print(f"  • {step}")
