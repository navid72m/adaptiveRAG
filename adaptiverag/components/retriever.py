from __future__ import annotations

import time
from typing import Dict, List, Optional

import chromadb
from tqdm import tqdm

from ..core.config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from .embedder import Embedder


class Retriever:
    def __init__(self, embedder: Embedder, persist_dir: str = CHROMA_PERSIST_DIR):
        self.embedder   = embedder
        self._client    = chromadb.PersistentClient(path=persist_dir)
        self.collection = self._client.get_or_create_collection(COLLECTION_NAME)

    @property
    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        name = self.collection.name
        self._client.delete_collection(name)
        self.collection = self._client.get_or_create_collection(name)

    def index(self, chunks: List[str], metadatas: List[Dict]):
        ids  = [f"chunk_{i}_{int(time.time() * 1000)}" for i in range(len(chunks))]
        embs = []
        for i in tqdm(range(0, len(chunks), self.embedder.batch_size), desc="Embedding"):
            embs.extend(self.embedder.embed(chunks[i:i + self.embedder.batch_size]))
        self.collection.add(
            embeddings=embs, documents=chunks, metadatas=metadatas, ids=ids
        )

    def retrieve(self, query: str, top_k: int = 5,
                 where: Optional[Dict] = None) -> List[str]:
        q_emb  = self.embedder.embed([query])[0]
        kwargs: Dict = {"n_results": top_k}
        if where:
            kwargs["where"] = where
        res = self.collection.query(q_emb, **kwargs)
        return res["documents"][0] if res["documents"] else []
