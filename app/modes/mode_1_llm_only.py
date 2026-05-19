from __future__ import annotations
# app/modes/mode_1_llm_only.py
from datetime import datetime, timezone

from app.schemas import InternalResponse
from app.services.llm_service import build_llm

SYSTEM_PROMPT = """Kamu adalah asisten informasi pasar saham Indonesia.
Jawab singkat dan jelas.
Jika tidak yakin, katakan bahwa informasi belum dapat diverifikasi."""


def run_mode_1(question: str, session_id: str) -> InternalResponse:
    """Mode 1: LLM-only baseline tanpa retrieval, cache, guardrails, atau critic.

    Digunakan sebagai control group eksperimen — mengukur tingkat halusinasi
    murni dari LLM tanpa augmentasi apa pun (§5 system prompt).

    Args:
        question: pertanyaan asli pengguna.
        session_id: ID sesi untuk Langfuse trace (dipakai oleh task #10).

    Returns:
        InternalResponse dengan validator_status="skipped", cache_status="bypassed",
        hallucination_flags=[] (tidak ada checker yang berjalan di mode ini).
    """
    llm = build_llm(temperature=0.0)
    answer = llm.invoke(f"{SYSTEM_PROMPT}\n\nPertanyaan: {question}").content

    return InternalResponse(
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
