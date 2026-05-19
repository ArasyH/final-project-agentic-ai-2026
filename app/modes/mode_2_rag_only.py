from __future__ import annotations
# app/modes/mode_2_rag_only.py
from datetime import datetime, timezone

from app.schemas import EvidenceItem, InternalResponse, SourceItem
from app.services.llm_service import build_llm
from app.services.retrieval_service import RetrievalService

RAG_PROMPT = """Jawab hanya berdasarkan konteks berikut.
Jika informasi tidak tersedia, katakan data tidak tersedia.

Konteks:
{context}

Pertanyaan:
{question}"""


def run_mode_2(question: str, session_id: str) -> InternalResponse:
    """Mode 2: RAG-only tanpa cache, guardrails, atau critic.

    Digunakan untuk mengukur kontribusi retrieval (RAG) terhadap reduksi
    halusinasi dibandingkan mode_1 (LLM-only) — §5 system prompt.

    Args:
        question: pertanyaan asli pengguna.
        session_id: ID sesi untuk Langfuse trace (dipakai oleh task #10).

    Returns:
        InternalResponse dengan validator_status="skipped", cache_status="bypassed",
        hallucination_flags=[] (tidak ada checker yang berjalan di mode ini).
        confidence: 0.7 jika dokumen berhasil di-retrieve, 0.3 jika tidak ada.
            Nilai ini adalah intrinsic property mode_2, bukan hyperparameter eksperimen.
    """
    retriever = RetrievalService()
    llm = build_llm(temperature=0.0)

    docs = retriever.retrieve(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = llm.invoke(RAG_PROMPT.format(context=context, question=question)).content

    evidence = [
        EvidenceItem(content=doc.page_content, source_id=f"kb_{i}")
        for i, doc in enumerate(docs)
    ]
    sources = [
        SourceItem(
            source_id=f"kb_{i}",
            title=doc.metadata.get("title"),
            snippet=doc.page_content[:240],
            metadata=doc.metadata,
        )
        for i, doc in enumerate(docs)
    ]

    return InternalResponse(
        answer=answer,
        evidence=evidence,
        sources=sources,
        tickers=[],
        timestamp=datetime.now(timezone.utc).isoformat(),
        confidence=0.7 if docs else 0.3,
        validator_status="skipped",
        cache_status="bypassed",
        mode="mode_2_rag_only",
        hallucination_flags=[],
    )
