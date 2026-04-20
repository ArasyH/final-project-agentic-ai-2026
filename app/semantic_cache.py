import time
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from config import (
    CHROMA_DB_PATH, SIMILARITY_THRESHOLD, CACHE_TTL_HOURS
)

_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
_model  = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# Collection TERPISAH dari knowledge base
_cache_col = _client.get_or_create_collection(
    name="semantic_cache",
    metadata={"hnsw:space": "cosine"},
)

def check_cache(question: str) -> dict:
    vec     = _model.encode(question).tolist()
    results = _cache_col.query(
        query_embeddings=[vec],
        n_results=1,
        include=["metadatas", "distances", "documents"],
    )

    if not results["ids"][0]:
        return {"hit": False, "score": 0.0, "answer": None, "contexts": []}

    distance   = results["distances"][0][0]
    similarity = 1.0 - distance          # cosine: distance 0 = identik
    meta       = results["metadatas"][0][0]

    cached_time = datetime.fromisoformat(meta["cached_at"])
    is_fresh    = (datetime.now() - cached_time) < timedelta(hours=CACHE_TTL_HOURS)

    if similarity >= SIMILARITY_THRESHOLD and is_fresh:
        return {
            "hit":              True,
            "score":            round(similarity, 4),
            "answer":           meta["answer"],
            "contexts":         meta.get("contexts", "[]"),
            "matched_question": results["documents"][0][0],
        }

    return {"hit": False, "score": round(similarity, 4),
            "answer": None, "contexts": []}

def store_cache(question: str, answer: str, contexts: list[str]):
    vec    = _model.encode(question).tolist()
    doc_id = f"cache_{abs(hash(question))}_{int(time.time())}"

    _cache_col.upsert(
        ids=[doc_id],
        embeddings=[vec],
        documents=[question],
        metadatas=[{
            "answer":    answer,
            "contexts":  str(contexts),     # ChromaDB metadata = string only
            "cached_at": datetime.now().isoformat(),
        }],
    )