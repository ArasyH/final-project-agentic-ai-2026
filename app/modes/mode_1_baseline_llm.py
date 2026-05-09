from __future__ import annotations
# app/modes/mode_1_baseline_llm.py
from datetime import datetime, timezone
from app.schemas import InternalResponse
from app.services.llm_service import build_llm

SYSTEM_PROMPT = """
Kamu adalah asisten informasi pasar saham Indonesia.
Jawab singkat dan jelas.
Jika tidak yakin, katakan bahwa informasi belum dapat diverifikasi.
"""

def run_mode_1(question: str) -> InternalResponse:
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
        mode="mode_1_baseline_llm",
    )