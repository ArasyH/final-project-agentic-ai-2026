"""app/evaluate_answer_correctness.py — Evaluasi RAGAS metrik answer_correctness.

Melengkapi app/evaluate_ragas.py dengan metrik ketiga yang membutuhkan ground truth:
  - answer_correctness : seberapa benar jawaban vs referensi ideal?

Ground truth diambil dari field `ground_truth` di evaluation_dataset.json
(diisi via app/create_ground_truth.py + app/apply_ground_truth.py).

Berbeda dengan faithfulness + answer_relevancy:
  - Semua 4 mode dievaluasi (termasuk Mode 1 yang tidak butuh contexts)
  - Ground truth "DATA_TIDAK_TERSEDIA" juga valid — mengukur apakah sistem
    tahu kapan harus mengatakan data tidak ada

Judge LLM: fallback chain identik dengan evaluate_ragas.py
  1. llama-3.3-70b-versatile (Groq)
  2. llama-3.1-8b-instant    (Groq fallback)
  3. gemini-2.5-flash        (Google, jika GOOGLE_API_KEY ada)
  4. mistral-small-latest    (Mistral, jika MISTRAL_API_KEY ada)

Resume-capable: baris dengan answer_correctness non-empty dilewati.

Output:
  - app/data/ragas_answer_correctness.csv  : skor per (mode, question_id)
  - app/data/ragas_results_per_mode.csv    : diperbarui dengan kolom answer_correctness

Jalankan:
    source venv/bin/activate
    python3 -m app.evaluate_answer_correctness
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
from ragas.metrics import answer_correctness
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
)

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
EXPERIMENT_CSV  = Path(__file__).parent / "data" / "experiment_results.csv"
DATASET_JSON    = Path(__file__).parent / "data" / "evaluation_dataset.json"
OUT_AC          = Path(__file__).parent / "data" / "ragas_answer_correctness.csv"
OUT_PER_MODE    = Path(__file__).parent / "data" / "ragas_results_per_mode.csv"
OUT_PER_Q       = Path(__file__).parent / "data" / "ragas_results_per_question.csv"

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

SLEEP_BETWEEN_SAMPLES: float = 8.0
SLEEP_AFTER_FALLBACK: float  = 2.0
RAGAS_RUN_CONFIG = RunConfig(max_workers=1, max_retries=2, max_wait=60)

OUT_COLS = ["mode", "question_id", "category", "answer_correctness", "model_used"]


# ---------------------------------------------------------------------------
# Statistik helpers
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
# Judge chain (identik dengan evaluate_ragas.py)
# ---------------------------------------------------------------------------

def _build_judge_chain() -> list[tuple[str, LangchainLLMWrapper]]:
    """Bangun fallback chain: Groq 70b → 8b → Gemini → Mistral."""
    chain: list[tuple[str, LangchainLLMWrapper]] = []

    if GROQ_API_KEY:
        llm = ChatGroq(model=GROQ_CRITIC_MODEL, api_key=GROQ_API_KEY, temperature=0.0)
        chain.append((GROQ_CRITIC_MODEL, LangchainLLMWrapper(llm)))

    if GROQ_API_KEY and GROQ_GENERATOR_MODEL != GROQ_CRITIC_MODEL:
        llm = ChatGroq(model=GROQ_GENERATOR_MODEL, api_key=GROQ_API_KEY, temperature=0.0)
        chain.append((GROQ_GENERATOR_MODEL, LangchainLLMWrapper(llm)))

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
            print("  [WARN] langchain-google-genai tidak terinstall, Google dilewati.")

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
            print("  [WARN] langchain-mistralai tidak terinstall, Mistral dilewati.")

    return chain


# ---------------------------------------------------------------------------
# Evaluate satu sample
# ---------------------------------------------------------------------------

def _evaluate_one(
    sample: SingleTurnSample,
    ragas_llm: LangchainLLMWrapper,
    ragas_emb: LangchainEmbeddingsWrapper,
) -> float | None:
    """Jalankan answer_correctness pada 1 sample. Return float atau None jika gagal."""
    dataset = EvaluationDataset(samples=[sample])
    result = evaluate(
        dataset=dataset,
        metrics=[answer_correctness],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=RAGAS_RUN_CONFIG,
        raise_exceptions=False,
        show_progress=False,
    )
    df = result.to_pandas()
    if "answer_correctness" not in df.columns:
        return None
    val = df.iloc[0]["answer_correctness"]
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _evaluate_with_fallback(
    sample: SingleTurnSample,
    judge_chain: list[tuple[str, LangchainLLMWrapper]],
    ragas_emb: LangchainEmbeddingsWrapper,
) -> tuple[float | None, str]:
    """Coba setiap model dalam chain. Return (answer_correctness, model_used)."""
    for model_name, ragas_llm in judge_chain:
        score = _evaluate_one(sample, ragas_llm, ragas_emb)
        if score is not None:
            return score, model_name
        print(f"→fallback({model_name} gagal) ", end="", flush=True)
        time.sleep(SLEEP_AFTER_FALLBACK)
    return None, "all_failed"


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def _load_completed() -> set[tuple[str, str]]:
    """Kembalikan set (mode, question_id) yang sudah punya answer_correctness valid."""
    if not OUT_AC.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with OUT_AC.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("answer_correctness"):
                done.add((row["mode"], row["question_id"]))
    return done


def _append_row(row: dict) -> None:
    """Append satu baris ke OUT_AC (checkpoint per sample)."""
    file_exists = OUT_AC.exists() and OUT_AC.stat().st_size > 0
    with OUT_AC.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Update ragas_results_per_mode.csv dengan kolom answer_correctness
# ---------------------------------------------------------------------------

def _update_per_mode_summary() -> None:
    """Tambahkan kolom answer_correctness ke ragas_results_per_mode.csv."""
    if not OUT_AC.exists():
        return

    # Baca answer_correctness per mode dari OUT_AC
    ac_by_mode: dict[str, list[float | None]] = {m: [] for m in MODES_ORDER}
    with OUT_AC.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mode = row.get("mode", "")
            if mode not in ac_by_mode:
                continue
            val = row.get("answer_correctness", "")
            ac_by_mode[mode].append(float(val) if val else None)

    # Baca per_mode yang sudah ada (faithfulness + answer_relevancy)
    existing: list[dict] = []
    if OUT_PER_MODE.exists():
        with OUT_PER_MODE.open(encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    # Buat lookup berdasarkan label mode
    label_to_row = {r["mode"]: r for r in existing}

    updated_rows: list[dict] = []
    for mode in MODES_ORDER:
        label = MODE_LABEL[mode]
        base  = label_to_row.get(label, {"mode": label})
        vals  = ac_by_mode[mode]
        m     = _mean(vals)
        s     = _std(vals)
        base["answer_correctness_mean"] = round(m, 4) if m is not None else ""
        base["answer_correctness_sd"]   = round(s, 4) if s is not None else ""
        base["n_nan_ac"] = sum(1 for v in vals if v is None)
        updated_rows.append(base)

    # Tulis ulang per_mode dengan kolom baru
    all_cols = list(updated_rows[0].keys()) if updated_rows else []
    with OUT_PER_MODE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"\nDiperbarui: {OUT_PER_MODE}", flush=True)


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

    # Load ground truth dari evaluation_dataset.json
    ground_truth_map: dict[str, str] = {}
    dataset_raw = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    for item in dataset_raw:
        gt = item.get("ground_truth") or ""
        ground_truth_map[item["question_id"]] = gt

    n_with_gt = sum(1 for v in ground_truth_map.values() if v)
    print("=== RAGAS ANSWER CORRECTNESS EVALUATION ===", flush=True)
    print(f"Ground truth tersedia: {n_with_gt}/50", flush=True)
    print(f"Sleep/sample         : {SLEEP_BETWEEN_SAMPLES}s", flush=True)

    if n_with_gt == 0:
        print("ERROR: Tidak ada ground truth. Jalankan dulu app/apply_ground_truth.py")
        return

    # Resume
    completed = _load_completed()
    print(f"Resume               : {len(completed)} sample valid ditemukan", flush=True)

    # Init embeddings
    print("Init embeddings...", flush=True)
    ragas_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    )

    # Build judge chain
    print("Build judge chain...", flush=True)
    judge_chain = _build_judge_chain()
    if not judge_chain:
        print("ERROR: Tidak ada model judge. Set API key di .env.")
        return
    print(f"Chain: {' → '.join(n for n, _ in judge_chain)}", flush=True)
    print(flush=True)

    for mode in MODES_ORDER:
        mode_rows = rows_by_mode[mode]
        print(f"\n--- {MODE_LABEL[mode]} ---", flush=True)
        print(f"Pertanyaan: {len(mode_rows)}", flush=True)

        for idx, row in enumerate(mode_rows):
            qid      = row["question_id"]
            question = row["question"]
            answer   = row["answer"]
            category = row.get("category", "")
            ref      = ground_truth_map.get(qid, "")

            if (mode, qid) in completed:
                print(f"  [{idx+1:>2}/{len(mode_rows)}] {qid} SKIP", flush=True)
                continue

            if not ref:
                print(f"  [{idx+1:>2}/{len(mode_rows)}] {qid} SKIP (no ground_truth)", flush=True)
                continue

            # answer_correctness tidak butuh retrieved_contexts
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                reference=ref,
            )

            print(f"  [{idx+1:>2}/{len(mode_rows)}] {qid} ... ", end="", flush=True)

            score, model_used = _evaluate_with_fallback(sample, judge_chain, ragas_emb)

            score_str = f"{score:.3f}" if score is not None else "nan"
            print(f"answer_correctness={score_str} [{model_used}]", flush=True)

            _append_row({
                "mode":               mode,
                "question_id":        qid,
                "category":           category,
                "answer_correctness": round(score, 4) if score is not None else "",
                "model_used":         model_used,
            })
            completed.add((mode, qid))

            if idx < len(mode_rows) - 1:
                time.sleep(SLEEP_BETWEEN_SAMPLES)

    # Ringkasan
    print("\n=== RINGKASAN ANSWER CORRECTNESS ===", flush=True)
    ac_rows: list[dict] = []
    with OUT_AC.open(encoding="utf-8") as f:
        ac_rows = list(csv.DictReader(f))

    for mode in MODES_ORDER:
        vals = [
            float(r["answer_correctness"])
            for r in ac_rows
            if r["mode"] == mode and r.get("answer_correctness")
        ]
        m = _mean(vals)
        s = _std(vals)
        score_str = f"{m:.3f} ± {s:.3f}" if m is not None else "nan"
        n_nan = sum(
            1 for r in ac_rows
            if r["mode"] == mode and not r.get("answer_correctness")
        )
        warn = f"  [WARN] {n_nan} nan" if n_nan else ""
        print(f"  {MODE_LABEL[mode]:<28}: {score_str}{warn}", flush=True)

    _update_per_mode_summary()

    print(f"\nPer-question: {OUT_AC}", flush=True)
    print(f"Per-mode    : {OUT_PER_MODE} (diperbarui)", flush=True)


if __name__ == "__main__":
    main()
