from __future__ import annotations
"""Mode 3: RAG + Judge & Critic (tanpa semantic cache).

Alur: GeneratorAgent (ReAct) → GuardrailsService (H1, H3) → CriticAgent (H2, H4)
      → aggregate flags → InternalResponse.

Services di-init di `run_mode_3` SEBELUM `measure_latency("total")` untuk fairness
latency comparison antar mode (§15 SINTA 2). `_run_rag_jc_pipeline` menerima semua
services sebagai parameter injeksi — tidak ada instantiation di dalam pipeline.

`_run_rag_jc_pipeline` adalah shared helper yang juga dipakai mode_4.
"""
import time
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
    parent_trace: Any,
    *,
    retrieval_service: RetrievalService,
    guardrails: GuardrailsService,
    generator: GeneratorAgent,
    critic: CriticAgent,
    telemetry: TelemetryService,
    tickers: list[str],
) -> InternalResponse:
    """Shared pipeline: GeneratorAgent (ReAct) → Guardrails → Critic.

    Services dan tickers HARUS di-init dan di-normalize sebelum call oleh caller.
    Pipeline tidak memanggil normalize_query atau menginstansiasi service apa pun.

    Caller (run_mode_3 / run_mode_4) bertanggung jawab atas:
    - Membuat trace via start_trace()
    - Pre-init semua services di luar measure_latency("total")
    - Memanggil normalize_query() dan menyediakan tickers
    - Mengukur latency_ms_total via measure_latency(trace, "total")
    - Memanggil end_trace() setelah pipeline selesai

    Pipeline mencatat latency_ms_retrieval, latency_ms_generation, latency_ms_critic
    pada trace yang diterima.

    Args:
        question: pertanyaan asli pengguna.
        session_id: ID sesi (diteruskan ke generator.generate()).
        question_id: ID pertanyaan untuk paired comparison.
        mode_str: ExperimentMode string untuk InternalResponse.
        cache_status: CacheStatus string ("bypassed" / "miss").
        parent_trace: Langfuse trace object dari caller.
        retrieval_service: RetrievalService instance pre-init oleh caller.
        guardrails: GuardrailsService instance pre-init oleh caller.
        generator: GeneratorAgent instance pre-init oleh caller.
        critic: CriticAgent instance pre-init oleh caller.
        telemetry: TelemetryService instance dari caller (shared, untuk latency dict).
        tickers: ticker list dari normalize_query oleh caller.

    Returns:
        InternalResponse. Jika generator gagal, confidence=0.2 dan
        validator_status="failed" (fail-safe path).
    """
    _gen_start = time.perf_counter()
    gen_output = generator.generate(
        question=question,
        session_id=session_id,
        tickers=tickers or None,
        trace_handle=parent_trace,
    )
    _gen_total_ms = round((time.perf_counter() - _gen_start) * 1000, 2)

    # Retrieval: akumulasi waktu semua retrieve_from_kb calls dalam ReAct loop
    telemetry._record_latency(parent_trace, "retrieval", round(gen_output.retrieval_latency_ms, 2))
    # Generation pure: total ReAct loop minus waktu KB retrieval (pure LLM compute)
    telemetry._record_latency(
        parent_trace, "generation",
        max(0.0, round(_gen_total_ms - gen_output.retrieval_latency_ms, 2)),
    )

    if not gen_output.succeeded:
        telemetry._record_latency(parent_trace, "critic", 0.0)
        telemetry.event(
            parent_trace,
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

    with telemetry.measure_latency(parent_trace, "critic"):
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
            "guardrails_status": guardrail_result.overall_status,
            "critic_verdict": critic_verdict.overall_verdict,
            "critic_details": {
                "H1": {
                    "flag": critic_verdict.H1_unsupported_numeric.flag,
                    "rationale": critic_verdict.H1_unsupported_numeric.rationale,
                },
                "H2": {
                    "flag": critic_verdict.H2_fabricated_metric.flag,
                    "rationale": critic_verdict.H2_fabricated_metric.rationale,
                },
                "H3": {
                    "flag": critic_verdict.H3_stale_timestamp.flag,
                    "rationale": critic_verdict.H3_stale_timestamp.rationale,
                },
                "H4": {
                    "flag": critic_verdict.H4_incorrect_inference.flag,
                    "rationale": critic_verdict.H4_incorrect_inference.rationale,
                },
            },
            "guardrails_details": {
                "H1": guardrail_result.H1_unsupported_numeric,
                "H3": guardrail_result.H3_stale_timestamp,
                "no_investment_recommendation": guardrail_result.no_investment_recommendation,
            },
        },
    )


def run_mode_3(question: str, session_id: str, question_id: str) -> InternalResponse:
    """Mode 3: RAG + Judge & Critic tanpa cache.

    Semua services di-init SEBELUM measure_latency("total") agar latency_ms_total
    hanya mengukur pipeline execution — konsisten dengan mode_1 dan mode_2
    untuk fairness comparison §15 SINTA 2.

    Args:
        question: pertanyaan asli pengguna.
        session_id: ID sesi untuk Langfuse trace.
        question_id: ID pertanyaan untuk paired comparison lintas mode.

    Returns:
        InternalResponse dari _run_rag_jc_pipeline.
    """
    # Service init di luar measure_latency("total") — fairness §15
    telemetry = TelemetryService()
    retrieval = RetrievalService()
    guardrails = GuardrailsService()
    critic = CriticAgent()
    normalized = normalize_query(question)
    generator = GeneratorAgent(
        retrieval_service=retrieval,
        telemetry_service=telemetry,
    )

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
            parent_trace=trace,
            retrieval_service=retrieval,
            guardrails=guardrails,
            generator=generator,
            critic=critic,
            telemetry=telemetry,
            tickers=normalized.detected_tickers,
        )

    telemetry.end_trace(trace, metadata={
        "mode": "mode_3_rag_jc",
        "question_id": question_id,
        "cache_status": result.cache_status,
        "validator_status": result.validator_status,
        "hallucination_flags": result.hallucination_flags,
        "evidence_count": len(result.evidence),
        "confidence": result.confidence,
    }, output=result.answer)

    return result
