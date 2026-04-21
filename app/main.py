import time
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langfuse import observe
from agent import rag_chain
from semantic_cache import check_cache, store_cache
from observability import (
    create_trace, log_cache_span,
    log_rag_generation, finalize_trace, flush
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server siap. RAG chain dan ChromaDB terload.")
    yield

app = FastAPI(title="IDX30 Stock Chatbot", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question:   str
    session_id: str | None = None

class ChatResponse(BaseModel):
    answer:     str
    source:     str
    similarity: float
    latency_ms: float
    session_id: str


# ── fungsi sync dibungkus @observe ──────────────────────────
@observe()                          # tanda kurung wajib di v4
def process_chat(question: str) -> dict:
    t_start = time.perf_counter()

    # 1. Cek cache
    t0           = time.perf_counter()
    cache_result = check_cache(question)
    cache_ms     = (time.perf_counter() - t0) * 1000

    log_cache_span(
        hit=cache_result["hit"],
        score=cache_result["score"],
        latency_ms=cache_ms,
    )

    if cache_result["hit"]:
        total_ms = (time.perf_counter() - t_start) * 1000
        finalize_trace(cache_result["answer"], cache_hit=True, total_ms=total_ms)
        flush()
        return {
            "answer":     cache_result["answer"],
            "source":     "cache",
            "similarity": cache_result["score"],
            "latency_ms": round(total_ms, 2),
        }

    # 2. Cache miss → RAG
    t1     = time.perf_counter()
    result = rag_chain.invoke({"query": question})
    rag_ms = (time.perf_counter() - t1) * 1000

    answer   = result["result"]
    contexts = [doc.page_content for doc in result["source_documents"]]

    log_rag_generation(
        question=question,
        answer=answer,
        contexts=contexts,
        latency_ms=rag_ms,
    )

    # 3. Simpan ke cache
    store_cache(question, answer, contexts)

    total_ms = (time.perf_counter() - t_start) * 1000
    finalize_trace(answer, cache_hit=False, total_ms=total_ms)
    flush()

    return {
        "answer":     answer,
        "source":     "agent",
        "similarity": cache_result["score"],
        "latency_ms": round(total_ms, 2),
    }


# ── endpoint async memanggil fungsi sync ────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Pertanyaan kosong.")

    session_id = req.session_id or str(uuid.uuid4())

    try:
        result = process_chat(question=req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        **result,
        session_id=session_id,
    )

@app.get("/health")
async def health():
    return {"status": "ok"}