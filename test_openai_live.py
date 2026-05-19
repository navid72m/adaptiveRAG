"""
Live test — AdaptiveRAG with OpenAI as the LLM provider.

Usage:
    python test_openai_live.py
    OPENAI_API_KEY=sk-... python test_openai_live.py  # override key
"""
import os
import sys

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    print("ERROR: OPENAI_API_KEY not set.")
    print("  Add it to .env or run:  export OPENAI_API_KEY=sk-...")
    sys.exit(1)

print("Provider : OpenAI (gpt-4o)")
print("Embedder : Ollama nomic-embed-text (local)")
print("Building RAG pipeline...\n")

from adaptiverag import build_rag

rag = build_rag(
    openai_api_key=openai_key,
    llm_model="gpt-4o",
    embed_model="nomic-embed-text:latest",
    kb_path="./knowledge_base",
)

questions = [
    "What is the main topic covered in the documents?",
    "Summarize the key points.",
]

for q in questions:
    print(f"Q: {q}")
    result = rag.ask(q)
    print(f"A: {result.answer}")
    print(f"   Confidence: {result.confidence:.2f}  |  Retries: {result.retries}  |  Strategy: {result.strategy}")
    print()
