# import time
# import uuid
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from contextlib import asynccontextmanager

# from agent import rag_chain
# from semantic_cache import check_cache, store_cache
# from observability import (
#     create_trace, log_cache_span,
#     log_rag_generation, finalize_trace,
# )

# # ── startup / shutdown ────────────────────────────────────────
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("Server siap. RAG chain dan ChromaDB terload.")
#     yield
#     print("Server mati.")

# app = FastAPI(
#     title="IDX30 Stock Chatbot API",
#     version="1.0.0",
#     lifespan=lifespan,
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],    # ganti dengan domain spesifik di production
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ── schema ────────────────────────────────────────────────────
# class ChatRequest(BaseModel):
#     question: str
#     session_id: str | None = None   # opsional, untuk grouping di Langfuse

# class ChatResponse(BaseModel):
#     answer:     str
#     source:     str             # "cache" atau "agent"
#     similarity: float
#     latency_ms: float
#     session_id: str

# # ── endpoint utama ────────────────────────────────────────────
# @app.post("/chat", response_model=ChatResponse)
# async def chat(req: ChatRequest):
#     if not req.question.strip():
#         raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")

#     session_id = req.session_id or str(uuid.uuid4())
#     t_start    = time.perf_counter()

#     # Langfuse trace
#     trace = create_trace(session_id, req.question)

#     # ── 1. Cek semantic cache ─────────────────────────────────
#     t_cache_start  = time.perf_counter()
#     cache_result   = check_cache(req.question)
#     cache_latency  = (time.perf_counter() - t_cache_start) * 1000

#     log_cache_span(trace, cache_result["hit"],
#                    cache_result["score"], cache_latency)

#     if cache_result["hit"]:
#         total_ms = (time.perf_counter() - t_start) * 1000
#         finalize_trace(trace, cache_result["answer"], True, total_ms)

#         return ChatResponse(
#             answer=cache_result["answer"],
#             source="cache",
#             similarity=cache_result["score"],
#             latency_ms=round(total_ms, 2),
#             session_id=session_id,
#         )

#     # ── 2. Cache miss → jalankan RAG agent ───────────────────
#     t_rag_start = time.perf_counter()
#     try:
#         result = rag_chain.invoke({"query": req.question})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")

#     answer   = result["result"]
#     contexts = [doc.page_content for doc in result["source_documents"]]
#     rag_ms   = (time.perf_counter() - t_rag_start) * 1000

#     log_rag_generation(trace, req.question, answer, contexts, rag_ms)

#     # ── 3. Simpan ke cache ────────────────────────────────────
#     store_cache(req.question, answer, contexts)

#     total_ms = (time.perf_counter() - t_start) * 1000
#     finalize_trace(trace, answer, False, total_ms)

#     return ChatResponse(
#         answer=answer,
#         source="agent",
#         similarity=cache_result["score"],
#         latency_ms=round(total_ms, 2),
#         session_id=session_id,
#     )

# # ── health check ──────────────────────────────────────────────
# @app.get("/health")
# async def health():
#     return {"status": "ok"}

# main.py
import time
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langfuse.decorators import observe

from agent import rag_chain
from semantic_cache import check_cache, store_cache
from observability import (
    log_cache_span,
    log_rag_generation,
    update_trace_metadata,
    flush,
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
        update_trace_metadata(cache_hit=True, total_ms=total_ms)
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
    update_trace_metadata(cache_hit=False, total_ms=total_ms)
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