"""app/run_experiment_gemini.py — Eksperimen komparatif model alternatif.

Menjalankan mode_3 dan mode_4 dengan model berbeda dari arsitektur frozen:
  Generator : Mistral Small 2603 (Mistral) — menggantikan Llama-3.1-8B KHUSUS eksperimen ini
  Critic    : Mistral Medium Latest        — menggantikan Llama-3.3-70B KHUSUS eksperimen ini

Model Groq (Llama) tetap frozen untuk sistem utama dan tidak diubah.
Runner ini hanya menggunakan DI (dependency injection) ke GeneratorAgent dan CriticAgent.

Prompt yang digunakan identik dengan eksperimen full V2:
  Generator : REACT_PROMPT_V5  (react_v5)
  Critic    : CRITIC_PROMPT_V4 (critic_v4)

Output: app/data/experiment_results_mistral.csv (100 baris: 50Q × 2 mode).

Fitur:
- CSV ditulis per-baris (checkpoint) — aman jika di-interrupt
- Resume otomatis: skip (mode, question_id) yang sudah ada di CSV
- stdout di-flush setiap baris — progress terlihat real-time

Catatan: Mode 4 menggunakan CacheService yang sama dengan eksperimen lain.
Jika experiment_full_v2 (mode_4) sudah menyimpan jawaban di cache, eksperimen
ini berpotensi mendapat cache hit. Ini adalah limitasi yang dapat dicatat
sebagai catatan metodologi.

Jalankan:
    source venv/bin/activate
    python3 -m app.run_experiment_gemini
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agents.critic_agent import CriticAgent
from app.agents.generator_agent import GeneratorAgent
from app.modes.mode_3_rag_jc import _run_rag_jc_pipeline
from app.schemas import EvidenceItem, InternalResponse, SourceItem
from app.services.cache_service import CacheService
from app.services.guardrails_service import GuardrailsService
from app.config import MISTRAL_GENERATOR_MODEL
from app.services.llm_service import build_critic_mistral, build_generator_mistral
from app.services.query_normalizer import normalize_query
from app.services.retrieval_service import RetrievalService
from app.services.telemetry_service import TelemetryService

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
DATASET_PATH = Path(__file__).parent / "data" / "evaluation_dataset.json"
OUTPUT_PATH  = Path(__file__).parent / "data" / "experiment_results_mistral.csv"

SLEEP_BETWEEN_QUESTIONS: float = 5.0   # Mistral API — jeda lebih panjang dari Groq
SLEEP_BETWEEN_MODES: float     = 15.0

GENERATOR_TEMPERATURE: float = 0.0
CRITIC_TEMPERATURE: float    = 0.0

CSV_COLUMNS = [
    "question_id",
    "question",
    "category",
    "mode",
    "answer",
    "hallucination_flags",
    "validator_status",
    "cache_status",
    "confidence",
    "latency_ms_total",
    "evidence_count",
    "iterations_used",
    "error",
    "prompt_version",
]

EXPERIMENT_SESSION_ID = (
    f"exp-mistral-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
)


# ---------------------------------------------------------------------------
# Mode runner wrappers (DI Mistral Generator + Mistral Critic)
# ---------------------------------------------------------------------------

def run_mode_3_mistral(
    question: str,
    session_id: str,
    question_id: str,
) -> InternalResponse:
    """Mode 3 dengan Generator Mistral Small 2603 + Critic Mistral Medium.

    Replikasi logika run_mode_3 dengan LLM berbeda via DI.
    Pipeline (_run_rag_jc_pipeline), GuardrailsService, RetrievalService,
    dan prompt template identik dengan eksperimen utama.

    Args:
        question: pertanyaan asli pengguna.
        session_id: ID sesi untuk Langfuse trace.
        question_id: ID pertanyaan untuk paired comparison.

    Returns:
        InternalResponse dari _run_rag_jc_pipeline.
    """
    telemetry  = TelemetryService()
    retrieval  = RetrievalService()
    guardrails = GuardrailsService()
    critic     = CriticAgent(llm=build_critic_mistral(temperature=CRITIC_TEMPERATURE))
    normalized = normalize_query(question)
    generator  = GeneratorAgent(
        retrieval_service=retrieval,
        telemetry_service=telemetry,
        llm=build_generator_mistral(temperature=GENERATOR_TEMPERATURE),
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
        "mode":               "mode_3_rag_jc",
        "question_id":        question_id,
        "cache_status":       result.cache_status,
        "validator_status":   result.validator_status,
        "hallucination_flags": result.hallucination_flags,
        "evidence_count":     len(result.evidence),
        "confidence":         result.confidence,
        "generator_model":    MISTRAL_GENERATOR_MODEL,
        "critic_model":       "mistral-medium-latest",
    }, output=result.answer)

    return result


def run_mode_4_mistral(
    question: str,
    session_id: str,
    question_id: str,
) -> InternalResponse:
    """Mode 4 dengan Generator Mistral Small 2603 + Critic Mistral Medium + Semantic Cache.

    Replikasi logika run_mode_4 dengan LLM berbeda via DI.
    CacheService menggunakan koleksi ChromaDB yang sama dengan eksperimen lain —
    cache hit bisa terjadi jika eksperimen lain (full_v2 mode_4) sudah menyimpan
    jawaban untuk pertanyaan yang sama.

    Args:
        question: pertanyaan asli pengguna.
        session_id: ID sesi untuk Langfuse trace.
        question_id: ID pertanyaan untuk paired comparison.

    Returns:
        InternalResponse dengan cache_status "hit" atau "miss".
    """
    telemetry  = TelemetryService()
    cache      = CacheService()
    retrieval  = RetrievalService()
    guardrails = GuardrailsService()
    critic     = CriticAgent(llm=build_critic_mistral(temperature=CRITIC_TEMPERATURE))
    normalized = normalize_query(question)
    tickers    = normalized.detected_tickers
    generator  = GeneratorAgent(
        retrieval_service=retrieval,
        telemetry_service=telemetry,
        llm=build_generator_mistral(temperature=GENERATOR_TEMPERATURE),
    )

    trace = telemetry.start_trace(
        session_id=session_id,
        question=question,
        mode="mode_4_rag_jc_cache",
        question_id=question_id,
    )

    with telemetry.measure_latency(trace, "total"):
        # ── Cache lookup ──────────────────────────────────────────────────────
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
                metadata={"score": cache_result["score"]},
            )
            telemetry._record_latency(trace, "retrieval", 0.0)
            telemetry._record_latency(trace, "generation", 0.0)
            telemetry._record_latency(trace, "critic",     0.0)

            evidence_dicts: list[dict] = cache_result.get("evidence_summary", [])
            evidence = [
                EvidenceItem(
                    content=item.get("content", ""),
                    source_id=item.get("source_id", f"cached_{i}"),
                )
                for i, item in enumerate(evidence_dicts)
            ]
            sources = [
                SourceItem(
                    source_id=item.get("source_id", f"cached_{i}"),
                    snippet=item.get("snippet"),
                )
                for i, item in enumerate(cache_result.get("source_metadata", []))
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
                metadata={"cache_score": cache_result["score"]},
            )

        # ── Cache miss — jalankan pipeline ────────────────────────────────────
        else:
            telemetry.event(
                trace,
                name="cache_miss",
                metadata={"score": cache_result["score"]},
            )
            result = _run_rag_jc_pipeline(
                question=question,
                session_id=session_id,
                question_id=question_id,
                mode_str="mode_4_rag_jc_cache",
                cache_status="miss",
                parent_trace=trace,
                retrieval_service=retrieval,
                guardrails=guardrails,
                generator=generator,
                critic=critic,
                telemetry=telemetry,
                tickers=tickers,
            )

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
                    telemetry.event(trace, name="cache_stored",
                                    metadata={"query": normalized.normalized_query[:100]})
                except Exception as exc:
                    telemetry.event(trace, name="cache_store_error",
                                    metadata={"error": str(exc)[:300]})
            else:
                telemetry.event(trace, name="cache_not_stored",
                                metadata={"reason": "validator_failed",
                                          "hallucination_flags": result.hallucination_flags})

    telemetry.end_trace(trace, metadata={
        "mode":               "mode_4_rag_jc_cache",
        "question_id":        question_id,
        "cache_status":       result.cache_status,
        "validator_status":   result.validator_status,
        "hallucination_flags": result.hallucination_flags,
        "evidence_count":     len(result.evidence),
        "confidence":         result.confidence,
        "generator_model":    MISTRAL_GENERATOR_MODEL,
        "critic_model":       "mistral-medium-latest",
    }, output=result.answer)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODES = [
    ("mode_3_rag_jc",       run_mode_3_mistral),
    ("mode_4_rag_jc_cache", run_mode_4_mistral),
]


def _load_completed(path: Path) -> set[tuple[str, str]]:
    """Baca pasangan (mode, question_id) yang sudah selesai dari CSV."""
    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("mode") and row.get("question_id"):
                done.add((row["mode"], row["question_id"]))
    return done


def _append_row(path: Path, row: dict) -> None:
    """Tulis satu baris ke CSV — buat header jika file belum ada."""
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _run_one(
    runner,
    question: str,
    session_id: str,
    question_id: str,
    mode_name: str,
    prompt_version: str,
) -> dict:
    """Jalankan satu pertanyaan di satu mode, tangkap latency dan error."""
    t0    = time.perf_counter()
    error = ""
    result: InternalResponse | None = None

    try:
        result = runner(question, session_id, question_id)
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:200]}"

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    if result is not None:
        return {
            "question_id":        question_id,
            "question":           question,
            "category":           "",
            "mode":               mode_name,
            "answer":             result.answer,
            "hallucination_flags": ",".join(result.hallucination_flags),
            "validator_status":   result.validator_status,
            "cache_status":       result.cache_status,
            "confidence":         result.confidence,
            "latency_ms_total":   latency_ms,
            "evidence_count":     len(result.evidence),
            "iterations_used":    result.metadata.get("iterations_used", ""),
            "error":              "",
            "prompt_version":     prompt_version,
        }

    return {
        "question_id":        question_id,
        "question":           question,
        "category":           "",
        "mode":               mode_name,
        "answer":             "",
        "hallucination_flags": "",
        "validator_status":   "failed",
        "cache_status":       "miss",
        "confidence":         0.0,
        "latency_ms_total":   latency_ms,
        "evidence_count":     0,
        "iterations_used":    "",
        "error":              error,
        "prompt_version":     prompt_version,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Jalankan eksperimen komparatif 50Q × 2 mode dengan Mistral Generator + Mistral Critic.

    Resume otomatis: skip (mode, question_id) yang sudah ada di OUTPUT_PATH.
    """
    from app.agents.generator_agent import PROMPT_VERSION as gen_pv
    from app.agents.critic_agent import PROMPT_VERSION as crit_pv
    from app.config import MISTRAL_GENERATOR_MODEL, MISTRAL_CRITIC_MODEL

    prompt_version_tag = (
        f"gen={gen_pv}@{MISTRAL_GENERATOR_MODEL},"
        f"crit={crit_pv}@{MISTRAL_CRITIC_MODEL}"
    )

    dataset    = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    total_q    = len(dataset)
    total_runs = total_q * len(MODES)
    completed  = _load_completed(OUTPUT_PATH)
    skipped    = len(completed)

    print("=== EKSPERIMEN MISTRAL MULAI ===", flush=True)
    print(f"Generator      : {MISTRAL_GENERATOR_MODEL}", flush=True)
    print(f"Critic         : {MISTRAL_CRITIC_MODEL}", flush=True)
    print(f"Prompt version : {prompt_version_tag}", flush=True)
    print(f"Session ID     : {EXPERIMENT_SESSION_ID}", flush=True)
    print(f"Pertanyaan     : {total_q}", flush=True)
    print(f"Mode           : {len(MODES)} (mode_3 + mode_4)", flush=True)
    print(f"Total runs     : {total_runs}", flush=True)
    print(f"Sudah selesai  : {skipped} (di-skip)", flush=True)
    print(f"Output         : {OUTPUT_PATH}", flush=True)
    print(f"Mulai          : {datetime.now(timezone.utc).isoformat()}", flush=True)
    print(flush=True)
    print(
        "CATATAN: CacheService menggunakan koleksi ChromaDB yang sama dengan eksperimen lain.\n"
        "Mode 4 mungkin mendapat cache hit dari jawaban yang disimpan eksperimen sebelumnya.",
        flush=True,
    )
    print(flush=True)

    run_counter = skipped

    for mode_name, runner in MODES:
        mode_done = sum(1 for (m, _) in completed if m == mode_name)
        if mode_done == total_q:
            print(f"--- MODE: {mode_name} [SKIP — sudah selesai] ---", flush=True)
            continue

        print(f"--- MODE: {mode_name} ---", flush=True)

        for idx, item in enumerate(dataset):
            question_id = item["question_id"]
            question    = item["question"]
            category    = item.get("category", "")

            if (mode_name, question_id) in completed:
                continue

            run_counter += 1
            print(
                f"  [{run_counter:>3}/{total_runs}] {question_id} | "
                f"{question[:55]}{'...' if len(question) > 55 else ''}",
                end="",
                flush=True,
            )

            row = _run_one(
                runner=runner,
                question=question,
                session_id=EXPERIMENT_SESSION_ID,
                question_id=question_id,
                mode_name=mode_name,
                prompt_version=prompt_version_tag,
            )
            row["category"] = category

            _append_row(OUTPUT_PATH, row)
            completed.add((mode_name, question_id))

            status_icon = "✗" if row["error"] else "✓"
            flags = row["hallucination_flags"] or "-"
            print(
                f" {status_icon} "
                f"latency={row['latency_ms_total']}ms "
                f"flags=[{flags}] "
                f"cache={row['cache_status']}",
                flush=True,
            )

            if row["error"]:
                print(f"    ERROR: {row['error']}", flush=True)

            if idx < total_q - 1:
                time.sleep(SLEEP_BETWEEN_QUESTIONS)

        print(f"  Mode {mode_name} selesai.\n", flush=True)

        if mode_name != MODES[-1][0]:
            print(
                f"  Jeda {SLEEP_BETWEEN_MODES}s sebelum mode berikutnya...",
                flush=True,
            )
            time.sleep(SLEEP_BETWEEN_MODES)

    # Ringkasan akhir
    all_rows: list[dict] = []
    with OUTPUT_PATH.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    total_errors = sum(1 for r in all_rows if r.get("error"))
    total_flags  = sum(1 for r in all_rows if r.get("hallucination_flags"))

    by_mode = {}
    for r in all_rows:
        by_mode.setdefault(r["mode"], []).append(r)

    print("=== EKSPERIMEN MISTRAL SELESAI ===", flush=True)
    print(f"Selesai    : {datetime.now(timezone.utc).isoformat()}", flush=True)
    print(f"Total rows : {len(all_rows)}", flush=True)
    print(f"Errors     : {total_errors}", flush=True)
    print(f"Rows dgn hallucination flags : {total_flags}", flush=True)
    print(f"CSV        : {OUTPUT_PATH}", flush=True)
    print(flush=True)
    print("Ringkasan per mode:", flush=True)
    for mode_name, _ in MODES:
        mrs = by_mode.get(mode_name, [])
        if not mrs:
            continue
        n       = len(mrs)
        halluc  = sum(1 for r in mrs if r.get("hallucination_flags", "").strip())
        passed  = sum(1 for r in mrs if r.get("validator_status") == "passed")
        ev0     = sum(1 for r in mrs if int(r.get("evidence_count", "0") or 0) == 0)
        ch      = sum(1 for r in mrs if r.get("cache_status") == "hit")
        print(
            f"  {mode_name}: halluc={halluc}/{n} passed={passed}/{n} "
            f"ev0={ev0}/{n} cache_hit={ch}/{n}",
            flush=True,
        )


if __name__ == "__main__":
    main()
