from __future__ import annotations
# Memisahkan layer HTTP dari core logic untuk menjaga modularitas dan kemudahan testing.
# app/chat_api.py
import time
import uuid
from fastapi import APIRouter, HTTPException
from app.config import DEFAULT_EXPERIMENT_MODE
from app.schemas import ChatRequest, ChatResponse
from app.services.orchestrator_service import OrchestratorService

router = APIRouter()
orchestrator = OrchestratorService()

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Pertanyaan kosong.")

    session_id = req.session_id or str(uuid.uuid4())
    mode = req.mode or DEFAULT_EXPERIMENT_MODE
    started = time.perf_counter()

    try:
        result = orchestrator.run(question=req.question, mode=mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    return ChatResponse(
        answer=result.answer,
        mode=result.mode,
        source="cache" if result.cache_status == "hit" else "agent",
        similarity=result.metadata.get("cache_similarity", 0.0),
        latency_ms=latency_ms,
        session_id=session_id,
        validator_status=result.validator_status,
        cache_status=result.cache_status,
        sources=result.sources,
        confidence=result.confidence,
    )