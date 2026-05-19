from __future__ import annotations
"""Mode 4: RAG + Judge & Critic + Semantic Cache.

Alur:
  1. Cache lookup (normalized_query, threshold 0.85, TTL 8h dari config).
  2. Cache hit  → kembalikan InternalResponse yang direkonstruksi dari cache.
  3. Cache miss → jalankan _run_rag_jc_pipeline (mode_3 pipeline).
  4. Jika validator_status == "passed" → store ke cache.
     Jika "failed" → jangan store (cegah cache pollution dengan jawaban
     yang masih punya halusinasi flag H1–H4).

Tujuan eksperimen: mengukur efisiensi cache (latency, cache-hit ratio)
sekaligus memastikan kualitas jawaban tidak turun dibanding mode_3 karena
hanya jawaban tervalidasi yang masuk cache.
"""
from datetime import datetime, timezone

from app.modes.mode_3_rag_jc import _run_rag_jc_pipeline
from app.schemas import EvidenceItem, InternalResponse, SourceItem
from app.services.cache_service import CacheService
from app.services.query_normalizer import normalize_query
from app.services.telemetry_service import TelemetryService


def run_mode_4(question: str, session_id: str) -> InternalResponse:
    """Mode 4: RAG + Judge & Critic + Semantic Cache.

    Args:
        question: pertanyaan asli pengguna.
        session_id: ID sesi untuk Langfuse trace.

    Returns:
        InternalResponse dengan:
        - cache_status="hit" jika jawaban ditemukan di cache (similarity ≥ 0.85,
          TTL belum expired).
        - cache_status="miss" jika tidak ada cache atau cache expired/stale;
          jawaban dihasilkan oleh _run_rag_jc_pipeline.
        - mode="mode_4_rag_jc_cache" selalu.
        confidence: 0.85 untuk cache hit (jawaban pernah lolos validasi);
            0.85 atau 0.50 untuk cache miss sesuai hasil validator.
    """
    telemetry = TelemetryService()
    trace = telemetry.start_trace(
        session_id=session_id,
        question=question,
        mode="mode_4_rag_jc_cache",
    )

    normalized = normalize_query(question)
    cache = CacheService()

    # ── 1. Cache lookup ──────────────────────────────────────────────────────
    try:
        cache_result = cache.lookup(normalized.normalized_query)
    except Exception as exc:  # fail-safe: treat lookup failure as miss
        telemetry.event(
            trace,
            name="cache_lookup_error",
            metadata={"error": type(exc).__name__, "detail": str(exc)[:300]},
        )
        cache_result = {"hit": False, "status": "miss", "score": 0.0}

    # ── 2. Cache hit path ────────────────────────────────────────────────────
    if cache_result["hit"]:
        telemetry.event(
            trace,
            name="cache_hit",
            metadata={
                "score": cache_result["score"],
                "intent": cache_result.get("intent", ""),
            },
        )

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

        return InternalResponse(
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

    # ── 3. Cache miss — run full pipeline ────────────────────────────────────
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
        mode_str="mode_4_rag_jc_cache",
        cache_status="miss",
    )

    # ── 4. Conditional store ─────────────────────────────────────────────────
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
        except Exception as exc:  # fail-safe: cache write bukan kritis
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

    return result
