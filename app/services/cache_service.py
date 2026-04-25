from __future__ import annotations
#perkuat cache metadata + TTL + freshness
# app/services/cache_service.py
import json
import time
from datetime import datetime, timedelta, timezone
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import (
    CHROMA_DB_PATH,
    SEMANTIC_CACHE_COLLECTION,
    EMBEDDING_MODEL_NAME,
    SIMILARITY_THRESHOLD,
    CACHE_TTL_HOURS,
)

class CacheService:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.collection = self.client.get_or_create_collection(
            name=SEMANTIC_CACHE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, text: str) -> list[float]:
        return self.embedder.encode(text).tolist()

    def lookup(self, normalized_query: str) -> dict:
        vec = self._embed(normalized_query)
        result = self.collection.query(
            query_embeddings=[vec],
            n_results=1,
            include=["metadatas", "distances", "documents"],
        )

        if not result["ids"] or not result["ids"][0]:
            return {"hit": False, "status": "miss", "score": 0.0}

        meta = result["metadatas"][0][0]
        distance = result["distances"][0][0]
        similarity = 1.0 - distance

        cached_at = datetime.fromisoformat(meta["timestamp"])
        ttl_hours = int(meta.get("ttl", CACHE_TTL_HOURS))
        is_fresh = datetime.now(timezone.utc) - cached_at < timedelta(hours=ttl_hours)

        if similarity >= SIMILARITY_THRESHOLD and is_fresh:
            return {
                "hit": True,
                "status": "hit",
                "score": round(similarity, 4),
                "answer": meta["answer"],
                "intent": meta.get("intent", ""),
                "evidence_summary": json.loads(meta.get("evidence_summary", "[]")),
                "source_metadata": json.loads(meta.get("source_metadata", "[]")),
                "timestamp": meta["timestamp"],
            }

        return {
            "hit": False,
            "status": "stale" if similarity >= SIMILARITY_THRESHOLD else "miss",
            "score": round(similarity, 4),
        }

    def store(
        self,
        normalized_query: str,
        intent: str,
        answer: str,
        evidence_summary: list[dict],
        source_metadata: list[dict],
        ttl: int = CACHE_TTL_HOURS,
    ) -> None:
        doc_id = f"cache_{abs(hash(normalized_query))}_{int(time.time())}"
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[self._embed(normalized_query)],
            documents=[normalized_query],
            metadatas=[{
                "normalized_query": normalized_query,
                "intent": intent,
                "answer": answer,
                "evidence_summary": json.dumps(evidence_summary, ensure_ascii=False),
                "source_metadata": json.dumps(source_metadata, ensure_ascii=False),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ttl": ttl,
            }],
        )