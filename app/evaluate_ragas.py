"""app/evaluate_ragas.py — Evaluasi RAGAS untuk 4 mode eksperimen.

Metrik:
  - faithfulness       : apakah klaim jawaban didukung oleh evidence? (mode 2/3/4)
  - answer_relevancy   : apakah jawaban relevan dengan pertanyaan? (semua mode)

Konteks (contexts) diperoleh dengan re-retrieve dari KB — deterministik karena
KB, embedding model, dan top-k identik dengan yang dipakai saat eksperimen.
Mode 1 (LLM Only) tidak memiliki konteks → hanya answer_relevancy.

Judge LLM: llama-3.3-70b-versatile (Groq) — sama dengan Critic Agent.
Embedding : paraphrase-multilingual-MiniLM-L12-v2 — sama dengan KB.

Eksekusi sequential (max_workers=1, 1 sample per evaluate() call) untuk
menghindari Groq rate limit (RPM=30, TPD=100k). Sleep SLEEP_BETWEEN_SAMPLES
detik antar sample.

Resume-capable: baris yang sudah ada di output CSV dilewati.

Output:
  - app/data/ragas_results_per_question.csv  : skor per (mode, question_id)
  - app/data/ragas_results_per_mode.csv      : mean ± SD per mode

Jalankan:
    source venv/bin/activate
    python3 -m app.evaluate_ragas
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness
from ragas.run_config import RunConfig

from app.config import (
    EMBEDDING_MODEL_NAME,
    GROQ_API_KEY,
    GROQ_CRITIC_MODEL,
    RETRIEVAL_TOP_K,
)
from app.services.retrieval_service import RetrievalService

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
EXPERIMENT_CSV  = Path(__file__).parent / "data" / "experiment_results.csv"
DATASET_JSON    = Path(__file__).parent / "data" / "evaluation_dataset.json"
OUT_PER_Q       = Path(__file__).parent / "data" / "ragas_results_per_question.csv"
OUT_PER_MODE    = Path(__file__).parent / "data" / "ragas_results_per_mode.csv"

MODES_ORDER = [
    "mode_1_llm_only",
    "mode_2_rag_only",
    "mode_3_rag_jc",
    "mode_4_rag_jc_cache",
]
MODE_LABEL = {
    "mode_1_llm_only":     "Mode 1 (LLM Only)",
    "mode_2_rag_only":     "Mode 2 (RAG Only)",
    "mode_3_rag_jc":       "Mode 3 (RAG+J&C)",
    "mode_4_rag_jc_cache": "Mode 4 (RAG+J&C+Cache)",
}

# Jeda antar sample — menjaga agar total Groq request/menit di bawah RPM=30.
# Faithfulness membuat ~3–6 internal LLM call per sample. Dengan sleep 8 detik,
# throughput ~7.5 sample/menit = ~30–45 LLM call/menit → masih dalam batas.
SLEEP_BETWEEN_SAMPLES: float = 8.0

# RunConfig: sequential, retry hingga 3x dengan backoff max 60 detik.
RAGAS_RUN_CONFIG = RunConfig(max_workers=1, max_retries=3, max_wait=60)

# Kolom output CSV per-question
OUT_COLS = ["mode", "question_id", "category", "faithfulness", "answer_relevancy"]


# ---------------------------------------------------------------------------
# Helpers statistik (ignore nan)
# ---------------------------------------------------------------------------

def _mean(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(clean) / len(clean) if clean else None

def _std(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    if len(clean) < 2:
        return 0.0 if len(clean) == 1 else None
    m = sum(clean) / len(clean)
    return math.sqrt(sum((v - m) ** 2 for v in clean) / (len(clean) - 1))


# ---------------------------------------------------------------------------
# Re-retrieve contexts dari KB
# ---------------------------------------------------------------------------

def _build_contexts(retriever: RetrievalService, question: str, tickers: list[str]) -> list[str]:
    """Re-retrieve top-k dokumen dari KB untuk satu pertanyaan.

    Deterministik: KB, embedding, dan top-k identik dengan eksperimen.
    """
    docs = retriever.retrieve(question, tickers=tickers or None)
    return [doc.page_content for doc in docs] if docs else []


# ---------------------------------------------------------------------------
# Build RAGAS judge & embeddings
# ---------------------------------------------------------------------------

def _build_judge() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Judge LLM: llama-3.3-70b-versatile (sama dengan Critic Agent)."""
    llm = ChatGroq(
        model=GROQ_CRITIC_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.0,
    )
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


# ---------------------------------------------------------------------------
# Load completed (resume support)
# ---------------------------------------------------------------------------

def _load_completed() -> set[tuple[str, str]]:
    """Return set of (mode, question_id) already written to OUT_PER_Q."""
    done: set[tuple[str, str]] = set()
    if not OUT_PER_Q.exists():
        return done
    with OUT_PER_Q.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["mode"], row["question_id"]))
    return done


def _append_row(row: dict) -> None:
    """Append satu baris ke OUT_PER_Q (checkpoint per sample)."""
    file_exists = OUT_PER_Q.exists()
    with OUT_PER_Q.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Evaluate satu sample
# ---------------------------------------------------------------------------

def _evaluate_one(
    sample: SingleTurnSample,
    metrics: list,
    ragas_llm: LangchainLLMWrapper,
    ragas_emb: LangchainEmbeddingsWrapper,
    has_context: bool,
) -> tuple[float | None, float | None]:
    """Jalankan RAGAS evaluate pada 1 sample. Return (faithfulness, answer_relevancy).

    Mengembalikan None jika metric tidak applicable atau terjadi error.
    """
    dataset = EvaluationDataset(samples=[sample])
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=RAGAS_RUN_CONFIG,
        raise_exceptions=False,
        show_progress=False,
    )
    df = result.to_pandas()

    faith: float | None = None
    relev: float | None = None

    if has_context and "faithfulness" in df.columns:
        val = df.iloc[0]["faithfulness"]
        faith = float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else None

    if "answer_relevancy" in df.columns:
        val = df.iloc[0]["answer_relevancy"]
        relev = float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else None

    return faith, relev


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load experiment results
    rows_by_mode: dict[str, list[dict]] = {m: [] for m in MODES_ORDER}
    with EXPERIMENT_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["mode"] in rows_by_mode:
                rows_by_mode[row["mode"]].append(row)

    # Load dataset untuk expected_tickers
    dataset_items = {
        item["question_id"]: item
        for item in json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    }

    print("=== RAGAS EVALUATION (sequential, 1 sample/call) ===", flush=True)
    print(f"Judge LLM  : {GROQ_CRITIC_MODEL}", flush=True)
    print(f"Embeddings : {EMBEDDING_MODEL_NAME}", flush=True)
    print(f"Top-k      : {RETRIEVAL_TOP_K}", flush=True)
    print(f"Sleep/sample: {SLEEP_BETWEEN_SAMPLES}s", flush=True)
    print(flush=True)

    # Resume check
    completed = _load_completed()
    if completed:
        print(f"Resume: {len(completed)} sample sudah selesai, dilanjutkan...", flush=True)

    # Init retriever (sekali, dipakai semua mode)
    print("Init RetrievalService...", flush=True)
    retriever = RetrievalService()

    # Init judge
    print("Init RAGAS judge...", flush=True)
    ragas_llm, ragas_emb = _build_judge()

    # Akumulasi hasil per mode untuk summary
    per_mode_results: dict[str, dict] = {}

    for mode in MODES_ORDER:
        mode_rows = rows_by_mode[mode]
        has_context = mode != "mode_1_llm_only"
        metrics = [faithfulness, answer_relevancy] if has_context else [answer_relevancy]
        metric_names = ["faithfulness", "answer_relevancy"] if has_context else ["answer_relevancy"]

        print(f"\n--- {MODE_LABEL[mode]} ---", flush=True)
        print(f"Metrik     : {', '.join(metric_names)}", flush=True)
        print(f"Pertanyaan : {len(mode_rows)}", flush=True)

        faith_scores: list[float | None] = []
        relev_scores: list[float | None] = []

        for idx, row in enumerate(mode_rows):
            qid      = row["question_id"]
            question = row["question"]
            answer   = row["answer"]
            category = row.get("category", "")

            # Resume: skip jika sudah ada
            if (mode, qid) in completed:
                print(f"  [{idx+1:>2}/{len(mode_rows)}] {qid} SKIP (sudah ada)", flush=True)
                continue

            # Re-retrieve contexts
            if has_context:
                ds_item  = dataset_items.get(qid, {})
                tickers  = ds_item.get("expected_tickers", [])
                contexts = _build_contexts(retriever, question, tickers)
            else:
                contexts = []

            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts if has_context else None,
            )

            print(f"  [{idx+1:>2}/{len(mode_rows)}] {qid} ({len(contexts)} ctx) ...", end=" ", flush=True)

            # Evaluate 1 sample
            faith, relev = _evaluate_one(sample, metrics, ragas_llm, ragas_emb, has_context)

            faith_str = f"{faith:.3f}" if faith is not None else "nan"
            relev_str = f"{relev:.3f}" if relev is not None else "nan"
            print(f"faith={faith_str} relev={relev_str}", flush=True)

            faith_scores.append(faith)
            relev_scores.append(relev)

            _append_row({
                "mode":             mode,
                "question_id":      qid,
                "category":         category,
                "faithfulness":     round(faith, 4) if faith is not None else "",
                "answer_relevancy": round(relev, 4) if relev is not None else "",
            })
            completed.add((mode, qid))

            # Sleep antar sample (kecuali sample terakhir)
            if idx < len(mode_rows) - 1:
                time.sleep(SLEEP_BETWEEN_SAMPLES)

        # Kumpulkan semua skor mode ini (termasuk yang dari resume)
        per_mode_results[mode] = {
            "faith_scores": faith_scores,
            "relev_scores": relev_scores,
            "has_context": has_context,
            "n": len(mode_rows),
        }

    # ---------------------------------------------------------------------------
    # Tulis per-mode summary dari per-question CSV (termasuk resume rows)
    # ---------------------------------------------------------------------------
    print("\n=== MEREKAP HASIL PER MODE ===", flush=True)

    per_q_data: dict[str, list[dict]] = {m: [] for m in MODES_ORDER}
    with OUT_PER_Q.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["mode"] in per_q_data:
                per_q_data[row["mode"]].append(row)

    per_mode_summary: list[dict] = []
    for mode in MODES_ORDER:
        rows = per_q_data[mode]
        has_ctx = mode != "mode_1_llm_only"

        faith_vals: list[float | None] = []
        relev_vals: list[float | None] = []
        for r in rows:
            if has_ctx:
                faith_vals.append(float(r["faithfulness"]) if r["faithfulness"] else None)
            relev_vals.append(float(r["answer_relevancy"]) if r["answer_relevancy"] else None)

        f_mean = _mean(faith_vals) if has_ctx else None
        f_std  = _std(faith_vals)  if has_ctx else None
        r_mean = _mean(relev_vals)
        r_std  = _std(relev_vals)

        faith_str = f"{f_mean:.3f} ± {f_std:.3f}" if f_mean is not None else "N/A (no context)"
        print(f"  {MODE_LABEL[mode]}", flush=True)
        print(f"    faithfulness     : {faith_str}", flush=True)
        print(f"    answer_relevancy : {r_mean:.3f} ± {r_std:.3f}" if r_mean is not None else "    answer_relevancy : nan", flush=True)

        per_mode_summary.append({
            "mode":                    MODE_LABEL[mode],
            "faithfulness_mean":       round(f_mean, 4) if f_mean is not None else "",
            "faithfulness_sd":         round(f_std, 4)  if f_std  is not None else "",
            "answer_relevancy_mean":   round(r_mean, 4) if r_mean is not None else "",
            "answer_relevancy_sd":     round(r_std, 4)  if r_std  is not None else "",
            "n_questions":             len(rows),
        })

    with OUT_PER_MODE.open("w", newline="", encoding="utf-8") as f:
        cols = ["mode", "faithfulness_mean", "faithfulness_sd",
                "answer_relevancy_mean", "answer_relevancy_sd", "n_questions"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(per_mode_summary)

    print(f"\nPer-question : {OUT_PER_Q}", flush=True)
    print(f"Per-mode     : {OUT_PER_MODE}", flush=True)

    # Ringkasan akhir
    print("\n=== RINGKASAN AKHIR ===", flush=True)
    print(f"{'Mode':<28} {'Faithfulness':>15} {'Ans Relevancy':>15}", flush=True)
    print("-" * 60, flush=True)
    for s in per_mode_summary:
        f_str = (
            f"{s['faithfulness_mean']:.3f} ± {s['faithfulness_sd']:.3f}"
            if s["faithfulness_mean"] != "" else "N/A"
        )
        r_str = (
            f"{s['answer_relevancy_mean']:.3f} ± {s['answer_relevancy_sd']:.3f}"
            if s["answer_relevancy_mean"] != "" else "nan"
        )
        print(f"{s['mode']:<28} {f_str:>15} {r_str:>15}", flush=True)


if __name__ == "__main__":
    main()
