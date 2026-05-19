"""
Live test — AdaptiveRAG with Claude or OpenAI as the LLM provider.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... python test_claude_live.py
    OPENAI_API_KEY=sk-...        python test_claude_live.py
"""
import os
import sys

from adaptiverag import build_rag

anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
openai_key    = os.environ.get("OPENAI_API_KEY")

if not anthropic_key and not openai_key:
    print("ERROR: set one of:")
    print("  export ANTHROPIC_API_KEY=sk-ant-...")
    print("  export OPENAI_API_KEY=sk-...")
    sys.exit(1)

if anthropic_key:
    print("Provider : Claude (claude-opus-4-7)")
    rag = build_rag(
        api_key=anthropic_key,
        llm_model="claude-opus-4-7",
        embed_model="nomic-embed-text:latest",
        kb_path="./knowledge_base",
    )
else:
    print("Provider : OpenAI (gpt-4o)")
    rag = build_rag(
        openai_api_key=openai_key,
        llm_model="gpt-4o",
        embed_model="nomic-embed-text:latest",
        kb_path="./knowledge_base",
    )

print("\nAsking a question...")
result = rag.ask("What is the main topic covered in the documents?")
print(f"\nAnswer     : {result.answer}")
print(f"Confidence : {result.confidence:.2f}")
print(f"Retries    : {result.retries}")
print(f"Strategy   : {result.strategy}")
if result.trace:
    print("Trace:")
    for step in result.trace:
        print(f"  • {step}")
