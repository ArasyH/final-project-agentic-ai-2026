# app/modes/mode_2_rag_only.py
from __future__ import annotations
from datetime import datetime, timezone
from app.schemas import InternalResponse, EvidenceItem, SourceItem
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import build_llm

RAG_PROMPT = """
Jawab hanya berdasarkan konteks berikut.
Jika informasi tidak tersedia, katakan data tidak tersedia.

Konteks:
{context}

Pertanyaan:
{question}
"""

def run_mode_2(question: str) -> InternalResponse:
    retriever = RetrievalService()
    llm = build_llm()

    docs = retriever.retrieve(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = llm.invoke(RAG_PROMPT.format(context=context, question=question)).content

    evidence = [EvidenceItem(content=doc.page_content, source_id=f"kb_{i}") for i, doc in enumerate(docs)]
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
    )