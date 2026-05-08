# app/modes/mode_3_full_agentic.py
from __future__ import annotations
from datetime import datetime, timezone
from app.schemas import InternalResponse, EvidenceItem, SourceItem
from app.services.query_normalizer import normalize_query
from app.services.cache_service import CacheService
from app.services.retrieval_service import RetrievalService
from app.services.validator_service import ValidatorService
from app.services.llm_service import build_llm

FULL_PROMPT = """
Kamu adalah grounded multi-agent stock information assistant untuk pasar saham Indonesia.

Aturan:
- Jawab hanya berdasarkan evidence.
- Jangan mengarang angka.
- Jika evidence tidak cukup, katakan data tidak cukup.
- Pastikan ticker, periode, dan satuan konsisten.

Evidence:
{context}

Pertanyaan:
{question}
"""

def run_mode_3(question: str) -> InternalResponse:
    normalized = normalize_query(question)
    cache = CacheService()
    cache_result = cache.lookup(normalized.normalized_query)

    if cache_result["hit"]:
        return InternalResponse(
            answer=cache_result["answer"],
            evidence=[EvidenceItem(content=str(x)) for x in cache_result["evidence_summary"]],
            sources=[
                SourceItem(source_id=f"cache_{i}", metadata=item)
                for i, item in enumerate(cache_result["source_metadata"])
            ],
            tickers=normalized.detected_tickers,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=0.9,
            validator_status="passed",
            cache_status="hit",
            mode="mode_3_full_agentic",
            metadata={
                "normalized_query": normalized.normalized_query,
                "intent": normalized.intent,
                "cache_similarity": cache_result["score"],
            },
        )

    retriever = RetrievalService()
    llm = build_llm()
    validator = ValidatorService()

    docs = retriever.retrieve(
    normalized.normalized_query,
    tickers=normalized.detected_tickers  # pass ticker yang terdeteksi
)
    if not docs:
        return InternalResponse(
            answer="Data tidak cukup untuk menjawab pertanyaan ini.",
            evidence=[],
            sources=[],
            tickers=normalized.detected_tickers,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=0.2,
            validator_status="failed",
            cache_status=cache_result["status"],
            mode="mode_3_full_agentic",
            metadata={"fallback_reason": "no_retrieval_result"},
        )

    context = "\n\n".join(doc.page_content for doc in docs)
    answer = llm.invoke(FULL_PROMPT.format(context=context, question=normalized.normalized_query)).content

    evidence = [{"content": doc.page_content, "source_id": f"kb_{i}"} for i, doc in enumerate(docs)]
    validation = validator.validate(
        answer=answer,
        evidence=evidence,
        expected_tickers=normalized.detected_tickers,
    )

    if validation["status"] == "failed":
        safe_answer = "Data tersedia tetapi belum cukup tervalidasi untuk memberikan jawaban final yang andal."
        return InternalResponse(
            answer=safe_answer,
            evidence=[EvidenceItem(**e) for e in evidence],
            sources=[
                SourceItem(
                    source_id=f"kb_{i}",
                    title=doc.metadata.get("title"),
                    snippet=doc.page_content[:240],
                    metadata=doc.metadata,
                )
                for i, doc in enumerate(docs)
            ],
            tickers=normalized.detected_tickers,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=0.35,
            validator_status="failed",
            cache_status=cache_result["status"],
            mode="mode_3_full_agentic",
            metadata={"validation_issues": validation["issues"], "normalized_query": normalized.normalized_query},
        )

    source_metadata = [doc.metadata for doc in docs]
    cache.store(
        normalized_query=normalized.normalized_query,
        intent=normalized.intent,
        answer=answer,
        evidence_summary=evidence,
        source_metadata=source_metadata,
    )

    return InternalResponse(
        answer=answer,
        evidence=[EvidenceItem(**e) for e in evidence],
        sources=[
            SourceItem(
                source_id=f"kb_{i}",
                title=doc.metadata.get("title"),
                snippet=doc.page_content[:240],
                metadata=doc.metadata,
            )
            for i, doc in enumerate(docs)
        ],
        tickers=normalized.detected_tickers,
        timestamp=datetime.now(timezone.utc).isoformat(),
        confidence=max(0.5, 0.85 - validation["confidence_penalty"]),
        validator_status="passed",
        cache_status=cache_result["status"],
        mode="mode_3_full_agentic",
        metadata={"normalized_query": normalized.normalized_query, "intent": normalized.intent},
    )