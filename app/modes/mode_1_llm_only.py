from __future__ import annotations
# app/modes/mode_1_llm_only.py
from datetime import datetime, timezone

from app.schemas import InternalResponse
from app.services.llm_service import build_llm
from app.services.telemetry_service import TelemetryService

SYSTEM_PROMPT = """Kamu adalah asisten informasi pasar saham Indonesia.
Jawab singkat dan jelas.
Jika tidak yakin, katakan bahwa informasi belum dapat diverifikasi."""


def run_mode_1(question: str, session_id: str, question_id: str) -> InternalResponse:
    """Mode 1: LLM-only baseline tanpa retrieval, cache, guardrails, atau critic.

    Digunakan sebagai control group eksperimen — mengukur tingkat halusinasi
    murni dari LLM tanpa augmentasi apa pun (§5 system prompt).

    Args:
        question: pertanyaan asli pengguna.
        session_id: ID sesi untuk Langfuse trace.
        question_id: ID pertanyaan untuk paired comparison lintas mode.

    Returns:
        InternalResponse dengan validator_status="skipped", cache_status="bypassed",
        hallucination_flags=[] (tidak ada checker yang berjalan di mode ini).
        latency_ms_retrieval=0, latency_ms_critic=0 (stage tidak ada di mode ini).
    """
    telemetry = TelemetryService()
    llm = build_llm(temperature=0.0)

    trace = telemetry.start_trace(
        session_id=session_id,
        question=question,
        mode="mode_1_llm_only",
        question_id=question_id,
    )

    with telemetry.measure_latency(trace, "total"):
        with telemetry.measure_latency(trace, "generation"):
            answer = llm.invoke(f"{SYSTEM_PROMPT}\n\nPertanyaan: {question}").content
        telemetry._record_latency(trace, "retrieval", 0.0)
        telemetry._record_latency(trace, "critic", 0.0)

    result = InternalResponse(
        answer=answer,
        evidence=[],
        sources=[],
        tickers=[],
        timestamp=datetime.now(timezone.utc).isoformat(),
        confidence=0.4,
        validator_status="skipped",
        cache_status="bypassed",
        mode="mode_1_llm_only",
        hallucination_flags=[],
    )

    telemetry.end_trace(trace, metadata={
        "mode": "mode_1_llm_only",
        "question_id": question_id,
        "cache_status": "bypassed",
        "validator_status": "skipped",
        "hallucination_flags": [],
        "evidence_count": 0,
        "confidence": result.confidence,
    })

    return result
