from __future__ import annotations
"""Mode 3: RAG + Judge & Critic (tanpa semantic cache).

Alur: normalize_query → GeneratorAgent (ReAct) → GuardrailsService (H1, H3)
      → CriticAgent (H2, H4) → aggregate flags → InternalResponse.

Tujuan eksperimen: mengukur kontribusi pola Judge & Critic terhadap reduksi
halusinasi dibandingkan mode_2 (RAG-only). Tidak ada cache di mode ini.

`_run_rag_jc_pipeline` adalah shared helper yang juga dipakai mode_4.
"""
from datetime import datetime, timezone

from app.agents.critic_agent import CriticAgent
from app.agents.generator_agent import GeneratorAgent
from app.schemas import EvidenceItem, InternalResponse, SourceItem
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
    mode_str: str,
    cache_status: str = "bypassed",
) -> InternalResponse:
    """Shared pipeline: normalize → GeneratorAgent (ReAct) → Guardrails → Critic.

    Dipakai oleh run_mode_3 (cache_status="bypassed") dan run_mode_4
    pada cache-miss path (cache_status="miss").

    Args:
        question: pertanyaan asli pengguna.
        session_id: ID sesi untuk Langfuse trace.
        mode_str: nilai ExperimentMode yang diisi ke InternalResponse.
        cache_status: nilai CacheStatus yang diisi ke InternalResponse.
            "bypassed" untuk mode_3, "miss" untuk mode_4 cache-miss path.

    Returns:
        InternalResponse dengan validator_status dan hallucination_flags
        berdasarkan hasil gabungan GuardrailsService (H1, H3) dan
        CriticAgent (H2, H4).
        confidence: 0.85 jika validator passed, 0.50 jika failed,
            0.20 jika generator gagal (fail-safe path).
    """
    telemetry = TelemetryService()
    retriever = RetrievalService()
    guardrails = GuardrailsService()
    critic = CriticAgent()
    generator = GeneratorAgent(
        retrieval_service=retriever,
        telemetry_service=telemetry,
    )

    trace = telemetry.start_trace(
        session_id=session_id,
        question=question,
        mode=mode_str,
    )

    normalized = normalize_query(question)
    tickers = normalized.detected_tickers

    gen_output = generator.generate(
        question=question,
        session_id=session_id,
        tickers=tickers or None,
    )

    if not gen_output.succeeded:
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

    telemetry.event(
        trace,
        name="pipeline_complete",
        output_data=gen_output.answer,
        metadata={
            "validator_status": validator_status,
            "hallucination_flags": all_flags,
            "guardrails_status": guardrail_result.overall_status,
            "critic_verdict": critic_verdict.overall_verdict,
            "iterations_used": gen_output.iterations_used,
            "evidence_count": len(gen_output.evidence),
            "confidence": confidence,
        },
    )

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


def run_mode_3(question: str, session_id: str) -> InternalResponse:
    """Mode 3: RAG + Judge & Critic tanpa cache.

    Wrapper tipis di atas _run_rag_jc_pipeline dengan mode_str dan
    cache_status tetap untuk mode ini.

    Args:
        question: pertanyaan asli pengguna.
        session_id: ID sesi untuk Langfuse trace.

    Returns:
        InternalResponse dari _run_rag_jc_pipeline.
    """
    return _run_rag_jc_pipeline(
        question=question,
        session_id=session_id,
        mode_str="mode_3_rag_jc",
        cache_status="bypassed",
    )
