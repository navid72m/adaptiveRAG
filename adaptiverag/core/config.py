KNOWLEDGE_BASE_PATH = "./knowledge_base"
CHROMA_PERSIST_DIR  = "./chroma_store"
VAL_QUERIES_PATH    = "./validation_queries.json"

LLM_MODEL       = "gemma4:latest"
EMBED_MODEL     = "nomic-embed-text:latest"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "agentic_rag_v3"

MAX_OPTIMIZER_TURNS  = 6
MAX_QUERY_RETRIES    = 2
CONFIDENCE_THRESHOLD = 0.65
RETRIEVAL_THRESHOLD  = 0.50

CHUNK_SIZE_OPTIONS = [256, 512, 768, 1024]
TOP_K_OPTIONS      = [3, 5, 7, 10]
