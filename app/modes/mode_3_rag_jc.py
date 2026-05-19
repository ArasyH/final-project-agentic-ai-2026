from __future__ import annotations
"""Mode 3: RAG + Judge & Critic (tanpa semantic cache).

Alur: normalize_query → GeneratorAgent (ReAct) → GuardrailsService (H1, H3)
      → CriticAgent (H2, H4) → aggregate flags → InternalResponse.

`_run_rag_jc_pipeline` adalah shared helper yang juga dipakai mode_4.
Caller (run_mode_3 / run_mode_4) menyediakan trace + telemetry instance;
pipeline hanya mencatat latency per stage pada trace yang sudah ada.
"""
from datetime import datetime, timezone
from typing import Any

from app.agents.critic_agent import CriticAgent
from app.agents.generator_agent import GeneratorAgent
from app.schemas import InternalResponse, SourceItem
from app.services.guardrails_service import GuardrailsService
from app.services.query_normalizer import normalize_query
from app.services.retrieval_service import RetrievalService
from app.services.telemetry_service import TelemetryService

_FAILSAFE_ANSWER = (
    "Maaf, sistem tidak dapat menghasilkan jawaban saat ini. "
    "Silakan ulangi pertanyaan atau hubungi administrator."
)


def _run_rag_jc_pipeline(
    question: str,
    session_id: str,
    question_id: str,
    mode_str: str,
    cache_status: str,
    trace: Any,
    telemetry: TelemetryService,
) -> InternalResponse:
    """Shared pipeline: normalize → GeneratorAgent (ReAct) → Guardrails → Critic.

    Caller (run_mode_3 / run_mode_4) bertanggung jawab atas:
    - Membuat trace via start_trace()
    - Mengukur latency_ms_total via measure_latency(trace, "total")
    - Memanggil end_trace() setelah pipeline selesai

    Pipeline ini mencatat latency_ms_retrieval, latency_ms_generation,
    latency_ms_critic pada trace yang diterima.

    Args:
        question: pertanyaan asli pengguna.
        session_id: ID sesi untuk Langfuse span dalam generator.
        question_id: ID pertanyaan untuk paired comparison.
        mode_str: ExperimentMode string untuk InternalResponse.
        cache_status: CacheStatus string ("bypassed" / "miss").
        trace: Langfuse trace object dari caller.
        telemetry: TelemetryService instance dari caller (shared, untuk latency dict).

    Returns:
        InternalResponse. Jika generator gagal, confidence=0.2 dan
        validator_status="failed" (fail-safe path).
    """
    retriever = RetrievalService()
    guardrails = GuardrailsService()
    critic = CriticAgent()
    generator = GeneratorAgent(
        retrieval_service=retriever,
        telemetry_service=telemetry,
    )

    with telemetry.measure_latency(trace, "retrieval"):
        normalized = normalize_query(question)
        tickers = normalized.detected_tickers

    with telemetry.measure_latency(trace, "generation"):
        gen_output = generator.generate(
            question=question,
            session_id=session_id,
            tickers=tickers or None,
        )

    if not gen_output.succeeded:
        telemetry._record_latency(trace, "critic", 0.0)
        telemetry.event(
            trace,
            name="generator_failed",
            metadata={
                "generator_error": gen_output.error,
                "iterations_used": gen_output.iterations_used,
            },
        )
        return InternalResponse(
            answer=gen_output.answer,
            evidence=[],
            sources=[],
            tickers=tickers,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=0.2,
            validator_status="failed",
            cache_status=cache_status,
            mode=mode_str,
            hallucination_flags=[],
            metadata={
                "generator_error": gen_output.error,
                "iterations_used": gen_output.iterations_used,
            },
        )

    evidence_dicts = [
        {"content": item.content, "source_id": item.source_id}
        for item in gen_output.evidence
    ]

    with telemetry.measure_latency(trace, "critic"):
        guardrail_result = guardrails.check(
            answer=gen_output.answer,
            evidence=evidence_dicts,
            now=datetime.now(timezone.utc),
        )
        critic_verdict = critic.validate(
            question=question,
            answer=gen_output.answer,
            evidence=evidence_dicts,
        )

    critic_flags: list[str] = []
    if critic_verdict.H2_fabricated_metric.flag:
        critic_flags.append("H2")
    if critic_verdict.H4_incorrect_inference.flag:
        critic_flags.append("H4")

    all_flags = sorted(set(guardrail_result.hallucination_flags + critic_flags))

    validator_status = (
        "passed"
        if guardrail_result.overall_status == "passed"
        and critic_verdict.overall_verdict == "pass"
        else "failed"
    )
    confidence = 0.85 if validator_status == "passed" else 0.50

    sources = [
        SourceItem(
            source_id=item.source_id or f"kb_{i}",
            snippet=item.content[:240],
        )
        for i, item in enumerate(gen_output.evidence)
    ]

    return InternalResponse(
        answer=gen_output.answer,
        evidence=gen_output.evidence,
        sources=sources,
        tickers=tickers,
        timestamp=datetime.now(timezone.utc).isoformat(),
        confidence=confidence,
        validator_status=validator_status,
        cache_status=cache_status,
        mode=mode_str,
        hallucination_flags=all_flags,
        metadata={
            "iterations_used": gen_output.iterations_used,
            "normalized_query": normalized.normalized_query,
            "intent": normalized.intent,
            "guardrails_status": guardrail_result.overall_status,
            "critic_verdict": critic_verdict.overall_verdict,
        },
    )


def run_mode_3(question: str, session_id: str, question_id: str) -> InternalResponse:
    """Mode 3: RAG + Judge & Critic tanpa cache.

    Wrapper: membuat trace, mengukur latency_ms_total, memanggil pipeline,
    memanggil end_trace dengan 11 field metadata wajib §15.

    Args:
        question: pertanyaan asli pengguna.
        session_id: ID sesi untuk Langfuse trace.
        question_id: ID pertanyaan untuk paired comparison lintas mode.

    Returns:
        InternalResponse dari _run_rag_jc_pipeline.
    """
    telemetry = TelemetryService()
    trace = telemetry.start_trace(
        session_id=session_id,
        question=question,
        mode="mode_3_rag_jc",
        question_id=question_id,
    )

    with telemetry.measure_latency(trace, "total"):
        result = _run_rag_jc_pipeline(
            question=question,
            session_id=session_id,
            question_id=question_id,
            mode_str="mode_3_rag_jc",
            cache_status="bypassed",
            trace=trace,
            telemetry=telemetry,
        )

    telemetry.end_trace(trace, metadata={
        "mode": "mode_3_rag_jc",
        "question_id": question_id,
        "cache_status": result.cache_status,
        "validator_status": result.validator_status,
        "hallucination_flags": result.hallucination_flags,
        "evidence_count": len(result.evidence),
        "confidence": result.confidence,
    })

    return result
