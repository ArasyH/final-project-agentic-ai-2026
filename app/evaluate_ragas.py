"""app/evaluate_ragas.py — Evaluasi RAGAS untuk 4 mode eksperimen.

Metrik:
  - faithfulness       : apakah klaim jawaban didukung oleh evidence? (mode 2/3/4)
  - answer_relevancy   : apakah jawaban relevan dengan pertanyaan? (semua mode)

Konteks (contexts) diperoleh dengan re-retrieve dari KB — deterministik karena
KB, embedding model, dan top-k identik dengan yang dipakai saat eksperimen.
Mode 1 (LLM Only) tidak memiliki konteks → hanya answer_relevancy.

Judge LLM — fallback chain (prioritas menurun):
  1. llama-3.3-70b-versatile (Groq, TPD=100k)  — sama dengan Critic Agent
  2. llama-3.1-8b-instant    (Groq, TPD≈500k)  — Generator model
  3. gemini-1.5-flash        (Google, TPM besar) — butuh GOOGLE_API_KEY di .env
  4. mistral-small-latest    (Mistral)           — butuh MISTRAL_API_KEY di .env

Jika satu model mengembalikan nan (rate limit / timeout), evaluasi sample
diulangi dengan model berikutnya dalam chain. Setiap baris mencatat
`model_used` untuk audit trail publikasi SINTA 2.

Embedding: paraphrase-multilingual-MiniLM-L12-v2 — frozen, tidak berubah.

Resume-capable: baris dengan `answer_relevancy` non-empty dianggap valid dan
dilewati. Baris nan/empty dihapus dari file dan dievaluasi ulang.

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
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    GROQ_CRITIC_MODEL,
    GROQ_GENERATOR_MODEL,
    MISTRAL_API_KEY,
    RAGAS_JUDGE_GOOGLE_MODEL,
    RAGAS_JUDGE_MISTRAL_MODEL,
    RETRIEVAL_TOP_K,
)
from app.services.retrieval_service import RetrievalService

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
EXPERIMENT_CSV = Path(__file__).parent / "data" / "experiment_results.csv"
DATASET_JSON   = Path(__file__).parent / "data" / "evaluation_dataset.json"
OUT_PER_Q      = Path(__file__).parent / "data" / "ragas_results_per_question.csv"
OUT_PER_MODE   = Path(__file__).parent / "data" / "ragas_results_per_mode.csv"

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

# Jeda antar sample setelah evaluasi berhasil — menjaga throughput < RPM limit.
SLEEP_BETWEEN_SAMPLES: float = 8.0
# Jeda singkat setelah fallback sebelum mencoba model berikutnya.
SLEEP_AFTER_FALLBACK: float = 2.0

# RunConfig: sequential, max_retries=2, max_wait=60s.
RAGAS_RUN_CONFIG = RunConfig(max_workers=1, max_retries=2, max_wait=60)

# Kolom output CSV per-question
OUT_COLS = [
    "mode", "question_id", "category",
    "faithfulness", "answer_relevancy", "model_used",
]


# ---------------------------------------------------------------------------
# Helpers statistik (ignore None/nan)
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

def _build_contexts(
    retriever: RetrievalService,
    question: str,
    tickers: list[str],
) -> list[str]:
    """Re-retrieve top-k dokumen dari KB untuk satu pertanyaan.

    Deterministik: KB, embedding, dan top-k identik dengan eksperimen.
    """
    docs = retriever.retrieve(question, tickers=tickers or None)
    return [doc.page_content for doc in docs] if docs else []


# ---------------------------------------------------------------------------
# Build embeddings (frozen, satu instance dipakai seluruh run)
# ---------------------------------------------------------------------------

def _build_embeddings() -> LangchainEmbeddingsWrapper:
    """Embedding model frozen: paraphrase-multilingual-MiniLM-L12-v2."""
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return LangchainEmbeddingsWrapper(emb)


# ---------------------------------------------------------------------------
# Build judge fallback chain
# ---------------------------------------------------------------------------

def _build_judge_chain() -> list[tuple[str, LangchainLLMWrapper]]:
    """Bangun chain judge LLM dengan prioritas:

    1. llama-3.3-70b-versatile (Groq)
    2. llama-3.1-8b-instant    (Groq fallback)
    3. gemini-1.5-flash        (Google, jika GOOGLE_API_KEY ada)
    4. mistral-small-latest    (Mistral, jika MISTRAL_API_KEY ada)

    Model yang tidak punya API key dilewati otomatis.
    Return list of (model_name, LangchainLLMWrapper).
    """
    chain: list[tuple[str, LangchainLLMWrapper]] = []

    # 1. Groq primary: llama-3.3-70b-versatile
    if GROQ_API_KEY:
        llm = ChatGroq(model=GROQ_CRITIC_MODEL, api_key=GROQ_API_KEY, temperature=0.0)
        chain.append((GROQ_CRITIC_MODEL, LangchainLLMWrapper(llm)))

    # 2. Groq fallback: llama-3.1-8b-instant
    if GROQ_API_KEY and GROQ_GENERATOR_MODEL != GROQ_CRITIC_MODEL:
        llm = ChatGroq(model=GROQ_GENERATOR_MODEL, api_key=GROQ_API_KEY, temperature=0.0)
        chain.append((GROQ_GENERATOR_MODEL, LangchainLLMWrapper(llm)))

    # 3. Google Gemini
    if GOOGLE_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=RAGAS_JUDGE_GOOGLE_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.0,
            )
            chain.append((RAGAS_JUDGE_GOOGLE_MODEL, LangchainLLMWrapper(llm)))
        except ImportError:
            print("  [WARN] langchain-google-genai tidak terinstall, Google dilewati.", flush=True)

    # 4. Mistral
    if MISTRAL_API_KEY:
        try:
            from langchain_mistralai import ChatMistralAI
            llm = ChatMistralAI(
                model=RAGAS_JUDGE_MISTRAL_MODEL,
                api_key=MISTRAL_API_KEY,
                temperature=0.0,
            )
            chain.append((RAGAS_JUDGE_MISTRAL_MODEL, LangchainLLMWrapper(llm)))
        except ImportError:
            print("  [WARN] langchain-mistralai tidak terinstall, Mistral dilewati.", flush=True)

    return chain


# ---------------------------------------------------------------------------
# Evaluate satu sample (satu model)
# ---------------------------------------------------------------------------

def _evaluate_one(
    sample: SingleTurnSample,
    metrics: list,
    ragas_llm: LangchainLLMWrapper,
    ragas_emb: LangchainEmbeddingsWrapper,
    has_context: bool,
) -> tuple[float | None, float | None]:
    """Jalankan RAGAS evaluate pada 1 sample dengan 1 model.

    Return (faithfulness, answer_relevancy). None berarti metric gagal (nan).
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

    def _safe(col: str) -> float | None:
        if col not in df.columns:
            return None
        val = df.iloc[0][col]
        if val is None:
            return None
        try:
            f = float(val)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    faith = _safe("faithfulness") if has_context else None
    relev = _safe("answer_relevancy")
    return faith, relev


# ---------------------------------------------------------------------------
# Evaluate satu sample dengan fallback chain
# ---------------------------------------------------------------------------

def _evaluate_with_fallback(
    sample: SingleTurnSample,
    metrics: list,
    judge_chain: list[tuple[str, LangchainLLMWrapper]],
    ragas_emb: LangchainEmbeddingsWrapper,
    has_context: bool,
) -> tuple[float | None, float | None, str]:
    """Coba setiap model dalam chain hingga berhasil.

    Return (faithfulness, answer_relevancy, model_used).
    Jika semua gagal, return (None, None, 'all_failed').
    Indikator kegagalan: answer_relevancy is None (nan dari RAGAS).
    """
    for model_name, ragas_llm in judge_chain:
        faith, relev = _evaluate_one(sample, metrics, ragas_llm, ragas_emb, has_context)
        if relev is not None:
            return faith, relev, model_name
        print(f"→fallback({model_name} gagal) ", end="", flush=True)
        time.sleep(SLEEP_AFTER_FALLBACK)

    return None, None, "all_failed"


# ---------------------------------------------------------------------------
# Resume: cleanup baris nan, kembalikan set (mode, qid) yang valid
# ---------------------------------------------------------------------------

def _cleanup_and_load_completed() -> set[tuple[str, str]]:
    """Baca OUT_PER_Q, simpan ulang hanya baris dengan answer_relevancy valid.

    Return set (mode, question_id) yang sudah selesai dengan hasil valid.
    Baris nan (answer_relevancy kosong) dihapus → akan dievaluasi ulang.
    """
    if not OUT_PER_Q.exists():
        return set()

    valid_rows: list[dict] = []
    done: set[tuple[str, str]] = set()

    with OUT_PER_Q.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_cols = reader.fieldnames or []
        for row in reader:
            if row.get("answer_relevancy"):  # non-empty → valid
                # Migrasi: tambah model_used jika belum ada
                if "model_used" not in existing_cols or not row.get("model_used"):
                    row["model_used"] = GROQ_CRITIC_MODEL
                # Pastikan semua kolom OUT_COLS ada
                cleaned = {col: row.get(col, "") for col in OUT_COLS}
                valid_rows.append(cleaned)
                done.add((row["mode"], row["question_id"]))

    # Tulis ulang file hanya dengan baris valid
    with OUT_PER_Q.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS)
        writer.writeheader()
        writer.writerows(valid_rows)

    return done


def _append_row(row: dict) -> None:
    """Append satu baris ke OUT_PER_Q (checkpoint per sample)."""
    file_exists = OUT_PER_Q.exists() and OUT_PER_Q.stat().st_size > 0
    with OUT_PER_Q.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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

    print("=== RAGAS EVALUATION (sequential + multi-model fallback) ===", flush=True)
    print(f"Embeddings  : {EMBEDDING_MODEL_NAME}", flush=True)
    print(f"Top-k       : {RETRIEVAL_TOP_K}", flush=True)
    print(f"Sleep/sample: {SLEEP_BETWEEN_SAMPLES}s", flush=True)
    print(flush=True)

    # Resume: cleanup baris nan, kembalikan yang sudah valid
    completed = _cleanup_and_load_completed()
    nan_removed = 0  # hitung dari selisih sebelum/setelah cleanup
    print(f"Resume      : {len(completed)} sample valid ditemukan", flush=True)

    # Init retriever
    print("Init RetrievalService...", flush=True)
    retriever = RetrievalService()

    # Init embeddings (frozen)
    print("Init embeddings (frozen)...", flush=True)
    ragas_emb = _build_embeddings()

    # Build judge chain
    print("Build judge chain...", flush=True)
    judge_chain = _build_judge_chain()
    if not judge_chain:
        print("ERROR: Tidak ada model judge yang tersedia. Set API key di .env.", flush=True)
        return
    chain_names = " → ".join(name for name, _ in judge_chain)
    print(f"Chain       : {chain_names}", flush=True)
    print(flush=True)

    for mode in MODES_ORDER:
        mode_rows = rows_by_mode[mode]
        has_context = mode != "mode_1_llm_only"
        metrics = [faithfulness, answer_relevancy] if has_context else [answer_relevancy]
        metric_names = ["faithfulness", "answer_relevancy"] if has_context else ["answer_relevancy"]

        print(f"\n--- {MODE_LABEL[mode]} ---", flush=True)
        print(f"Metrik     : {', '.join(metric_names)}", flush=True)
        print(f"Pertanyaan : {len(mode_rows)}", flush=True)

        for idx, row in enumerate(mode_rows):
            qid      = row["question_id"]
            question = row["question"]
            answer   = row["answer"]
            category = row.get("category", "")

            # Skip jika sudah ada hasil valid
            if (mode, qid) in completed:
                print(f"  [{idx+1:>2}/{len(mode_rows)}] {qid} SKIP", flush=True)
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

            print(f"  [{idx+1:>2}/{len(mode_rows)}] {qid} ({len(contexts)} ctx) ... ", end="", flush=True)

            faith, relev, model_used = _evaluate_with_fallback(
                sample, metrics, judge_chain, ragas_emb, has_context
            )

            faith_str = f"{faith:.3f}" if faith is not None else "nan"
            relev_str = f"{relev:.3f}" if relev is not None else "nan"
            print(f"faith={faith_str} relev={relev_str} [{model_used}]", flush=True)

            _append_row({
                "mode":             mode,
                "question_id":      qid,
                "category":         category,
                "faithfulness":     round(faith, 4) if faith is not None else "",
                "answer_relevancy": round(relev, 4) if relev is not None else "",
                "model_used":       model_used,
            })
            completed.add((mode, qid))

            if idx < len(mode_rows) - 1:
                time.sleep(SLEEP_BETWEEN_SAMPLES)

    # ---------------------------------------------------------------------------
    # Rekap per-mode dari per-question CSV
    # ---------------------------------------------------------------------------
    print("\n=== REKAP PER MODE ===", flush=True)

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
        models_used: set[str] = set()

        for r in rows:
            if has_ctx:
                faith_vals.append(float(r["faithfulness"]) if r["faithfulness"] else None)
            relev_vals.append(float(r["answer_relevancy"]) if r["answer_relevancy"] else None)
            if r.get("model_used"):
                models_used.add(r["model_used"])

        f_mean = _mean(faith_vals) if has_ctx else None
        f_std  = _std(faith_vals)  if has_ctx else None
        r_mean = _mean(relev_vals)
        r_std  = _std(relev_vals)

        faith_str = f"{f_mean:.3f} ± {f_std:.3f}" if f_mean is not None else "N/A"
        relev_str = f"{r_mean:.3f} ± {r_std:.3f}" if r_mean is not None else "nan"

        print(f"  {MODE_LABEL[mode]}", flush=True)
        print(f"    faithfulness     : {faith_str}", flush=True)
        print(f"    answer_relevancy : {relev_str}", flush=True)
        print(f"    models used      : {', '.join(sorted(models_used)) or '-'}", flush=True)

        nan_count = sum(1 for v in relev_vals if v is None)
        if nan_count:
            print(f"    [WARN] {nan_count}/{len(rows)} sample masih nan — coba jalankan ulang", flush=True)

        per_mode_summary.append({
            "mode":                    MODE_LABEL[mode],
            "faithfulness_mean":       round(f_mean, 4) if f_mean is not None else "",
            "faithfulness_sd":         round(f_std, 4)  if f_std  is not None else "",
            "answer_relevancy_mean":   round(r_mean, 4) if r_mean is not None else "",
            "answer_relevancy_sd":     round(r_std, 4)  if r_std  is not None else "",
            "n_questions":             len(rows),
            "n_nan":                   nan_count,
            "models_used":             ", ".join(sorted(models_used)),
        })

    with OUT_PER_MODE.open("w", newline="", encoding="utf-8") as f:
        cols = [
            "mode", "faithfulness_mean", "faithfulness_sd",
            "answer_relevancy_mean", "answer_relevancy_sd",
            "n_questions", "n_nan", "models_used",
        ]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(per_mode_summary)

    print(f"\nPer-question : {OUT_PER_Q}", flush=True)
    print(f"Per-mode     : {OUT_PER_MODE}", flush=True)

    print("\n=== RINGKASAN AKHIR ===", flush=True)
    print(f"{'Mode':<28} {'Faithfulness':>15} {'Ans Relevancy':>15} {'Models':>30}", flush=True)
    print("-" * 92, flush=True)
    for s in per_mode_summary:
        f_str = (
            f"{s['faithfulness_mean']:.3f} ± {s['faithfulness_sd']:.3f}"
            if s["faithfulness_mean"] != "" else "N/A"
        )
        r_str = (
            f"{s['answer_relevancy_mean']:.3f} ± {s['answer_relevancy_sd']:.3f}"
            if s["answer_relevancy_mean"] != "" else "nan"
        )
        print(f"{s['mode']:<28} {f_str:>15} {r_str:>15} {s['models_used']:>30}", flush=True)


if __name__ == "__main__":
    main()
