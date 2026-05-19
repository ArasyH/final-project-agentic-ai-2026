from __future__ import annotations
"""Mode 4: RAG + Judge & Critic + Semantic Cache.

Alur:
  1. Cache lookup (normalized_query, threshold 0.85, TTL 8h dari config).
  2. Cache hit  → rekonstruksi InternalResponse dari cache; semua stage latency = 0.
  3. Cache miss → jalankan _run_rag_jc_pipeline (trace yang sama, bukan trace baru).
  4. Jika validator_status == "passed" → store ke cache.
     Jika "failed" → jangan store (cegah cache pollution dengan jawaban H1–H4).

Trace unifikasi: 1 request = 1 Langfuse trace (bukan 2). Pipeline menerima
trace dari run_mode_4 via parameter; tidak membuat trace sendiri.
"""
from datetime import datetime, timezone

from app.modes.mode_3_rag_jc import _run_rag_jc_pipeline
from app.schemas import EvidenceItem, InternalResponse, SourceItem
from app.services.cache_service import CacheService
from app.services.query_normalizer import normalize_query
from app.services.telemetry_service import TelemetryService


def run_mode_4(question: str, session_id: str, question_id: str) -> InternalResponse:
    """Mode 4: RAG + Judge & Critic + Semantic Cache.

    Args:
        question: pertanyaan asli pengguna.
        session_id: ID sesi untuk Langfuse trace.
        question_id: ID pertanyaan untuk paired comparison lintas mode.

    Returns:
        InternalResponse dengan:
        - cache_status="hit": jawaban dari cache (similarity ≥ 0.85, TTL valid).
          latency_ms_retrieval/generation/critic = 0.
        - cache_status="miss": jawaban dari _run_rag_jc_pipeline.
          semua latency non-zero (diukur per stage).
        confidence: 0.85 untuk cache hit; 0.85/0.50 untuk miss sesuai validator.
    """
    telemetry = TelemetryService()
    cache = CacheService()

    trace = telemetry.start_trace(
        session_id=session_id,
        question=question,
        mode="mode_4_rag_jc_cache",
        question_id=question_id,
    )

    normalized = normalize_query(question)

    with telemetry.measure_latency(trace, "total"):
        # ── Cache lookup ─────────────────────────────────────────────────────
        try:
            cache_result = cache.lookup(normalized.normalized_query)
        except Exception as exc:
            telemetry.event(
                trace,
                name="cache_lookup_error",
                metadata={"error": type(exc).__name__, "detail": str(exc)[:300]},
            )
            cache_result = {"hit": False, "status": "miss", "score": 0.0}

        # ── Cache hit path ────────────────────────────────────────────────────
        if cache_result["hit"]:
            telemetry.event(
                trace,
                name="cache_hit",
                metadata={
                    "score": cache_result["score"],
                    "intent": cache_result.get("intent", ""),
                },
            )
            telemetry._record_latency(trace, "retrieval", 0.0)
            telemetry._record_latency(trace, "generation", 0.0)
            telemetry._record_latency(trace, "critic", 0.0)

            evidence_dicts: list[dict] = cache_result.get("evidence_summary", [])
            evidence = [
                EvidenceItem(
                    content=item.get("content", ""),
                    source_id=item.get("source_id", f"cached_{i}"),
                )
                for i, item in enumerate(evidence_dicts)
            ]
            source_meta: list[dict] = cache_result.get("source_metadata", [])
            sources = [
                SourceItem(
                    source_id=item.get("source_id", f"cached_{i}"),
                    snippet=item.get("snippet"),
                )
                for i, item in enumerate(source_meta)
            ]

            result = InternalResponse(
                answer=cache_result["answer"],
                evidence=evidence,
                sources=sources,
                tickers=[],
                timestamp=cache_result["timestamp"],
                confidence=0.85,
                validator_status="passed",
                cache_status="hit",
                mode="mode_4_rag_jc_cache",
                hallucination_flags=[],
                metadata={
                    "cache_score": cache_result["score"],
                    "cached_intent": cache_result.get("intent", ""),
                },
            )

        # ── Cache miss — run full pipeline (shared trace) ─────────────────────
        else:
            telemetry.event(
                trace,
                name="cache_miss",
                metadata={
                    "score": cache_result["score"],
                    "status": cache_result["status"],
                },
            )

            result = _run_rag_jc_pipeline(
                question=question,
                session_id=session_id,
                question_id=question_id,
                mode_str="mode_4_rag_jc_cache",
                cache_status="miss",
                trace=trace,
                telemetry=telemetry,
            )

            # ── Conditional store ─────────────────────────────────────────────
            if result.validator_status == "passed":
                evidence_summary = [
                    {"content": item.content, "source_id": item.source_id}
                    for item in result.evidence
                ]
                source_metadata = [
                    {"source_id": src.source_id, "snippet": src.snippet or ""}
                    for src in result.sources
                ]
                try:
                    cache.store(
                        normalized_query=normalized.normalized_query,
                        intent=normalized.intent or "",
                        answer=result.answer,
                        evidence_summary=evidence_summary,
                        source_metadata=source_metadata,
                    )
                    telemetry.event(
                        trace,
                        name="cache_stored",
                        metadata={"normalized_query": normalized.normalized_query[:100]},
                    )
                except Exception as exc:
                    telemetry.event(
                        trace,
                        name="cache_store_error",
                        metadata={"error": type(exc).__name__, "detail": str(exc)[:300]},
                    )
            else:
                telemetry.event(
                    trace,
                    name="cache_not_stored",
                    metadata={
                        "reason": "validator_failed",
                        "hallucination_flags": result.hallucination_flags,
                    },
                )

    telemetry.end_trace(trace, metadata={
        "mode": "mode_4_rag_jc_cache",
        "question_id": question_id,
        "cache_status": result.cache_status,
        "validator_status": result.validator_status,
        "hallucination_flags": result.hallucination_flags,
        "evidence_count": len(result.evidence),
        "confidence": result.confidence,
    })

    return result
