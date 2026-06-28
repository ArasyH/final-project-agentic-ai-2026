"""app/evaluate_ragas_full.py — Evaluasi RAGAS 4 metrik pada full_v2 (200 baris).

Metrik:
  - faithfulness        (mode 2/3/4) — klaim didukung context?
  - answer_relevancy    (semua mode) — jawaban relevan dengan pertanyaan?
  - context_precision   (mode 2/3/4) — chunk relevan terhadap pertanyaan + ground truth?
  - context_recall      (mode 2/3/4) — ground truth ter-cover oleh chunk?

Sumber data:
  - app/data/full_v2_with_contexts.csv  (200 baris × 4 mode dengan kolom `contexts`)
  - app/data/evaluation_dataset.json    (ground_truth per question_id)

Mode 1 (LLM only) tidak punya retrieval → hanya answer_relevancy.

Judge LLM — fallback chain:
  llama-3.3-70b-versatile → llama-3.1-8b-instant → gemini-2.5-flash → mistral-small-latest

Resume-safe per (mode, question_id).

Output:
  - app/data/ragas_full_v2_per_question.csv  : skor per baris
  - app/data/ragas_full_v2_per_mode.csv      : mean ± SD per mode

Jalankan:
    source venv/bin/activate
    python3 -m app.evaluate_ragas_full
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
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
INPUT_CSV    = Path(__file__).parent / "data" / "full_v2_with_contexts.csv"
DATASET_JSON = Path(__file__).parent / "data" / "evaluation_dataset.json"
# OUT_PATH env override → ganti basename CSV output (per-question + per-mode)
_OUT_BASENAME = os.getenv("OUT_BASENAME", "ragas_full_v2")
OUT_PER_Q    = Path(__file__).parent / "data" / f"{_OUT_BASENAME}_per_question.csv"
OUT_PER_MODE = Path(__file__).parent / "data" / f"{_OUT_BASENAME}_per_mode.csv"

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
SLEEP_AFTER_FALLBACK:  float = 2.0
RAGAS_RUN_CONFIG = RunConfig(max_workers=1, max_retries=2, max_wait=60)

OUT_COLS = [
    "mode", "question_id", "category",
    "faithfulness", "answer_relevancy",
    "context_precision", "context_recall",
    "model_used",
]


def _mean(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(clean) / len(clean) if clean else None


def _std(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    if len(clean) < 2:
        return 0.0 if len(clean) == 1 else None
    m = sum(clean) / len(clean)
    return math.sqrt(sum((v - m) ** 2 for v in clean) / (len(clean) - 1))


def _build_judge_chain() -> list[tuple[str, LangchainLLMWrapper]]:
    """Bangun chain judge LLM.

    Urutan eksekusi diatur via env CHAIN_PRIMARY (default: groq).
    Set CHAIN_PRIMARY=mistral untuk menaruh Mistral di depan
    (dipakai saat Groq TPD limit terpakai habis pada hari yg sama).
    """
    primary = os.getenv("CHAIN_PRIMARY", "groq").lower()
    chain: list[tuple[str, LangchainLLMWrapper]] = []

    def _add_groq():
        if GROQ_API_KEY:
            chain.append((GROQ_CRITIC_MODEL, LangchainLLMWrapper(
                ChatGroq(model=GROQ_CRITIC_MODEL, api_key=GROQ_API_KEY, temperature=0.0))))
            if GROQ_GENERATOR_MODEL != GROQ_CRITIC_MODEL:
                chain.append((GROQ_GENERATOR_MODEL, LangchainLLMWrapper(
                    ChatGroq(model=GROQ_GENERATOR_MODEL, api_key=GROQ_API_KEY, temperature=0.0))))

    def _add_mistral():
        if MISTRAL_API_KEY:
            try:
                from langchain_mistralai import ChatMistralAI
                chain.append((RAGAS_JUDGE_MISTRAL_MODEL, LangchainLLMWrapper(
                    ChatMistralAI(model=RAGAS_JUDGE_MISTRAL_MODEL,
                                  api_key=MISTRAL_API_KEY, temperature=0.0))))
            except ImportError:
                print("  [WARN] langchain-mistralai tidak terinstall.", flush=True)

    def _add_gemini():
        if GOOGLE_API_KEY:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                chain.append((RAGAS_JUDGE_GOOGLE_MODEL, LangchainLLMWrapper(
                    ChatGoogleGenerativeAI(model=RAGAS_JUDGE_GOOGLE_MODEL,
                                           google_api_key=GOOGLE_API_KEY, temperature=0.0))))
            except ImportError:
                print("  [WARN] langchain-google-genai tidak terinstall.", flush=True)

    if primary == "mistral":
        _add_mistral()
        _add_gemini()
        _add_groq()
    else:
        _add_groq()
        _add_gemini()
        _add_mistral()

    return chain


def _evaluate_one(
    sample: SingleTurnSample,
    metrics: list,
    ragas_llm: LangchainLLMWrapper,
    ragas_emb: LangchainEmbeddingsWrapper,
) -> dict[str, float | None]:
    """Evaluate satu sample dengan satu model. Return dict metric_name → value."""
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

    return {
        "faithfulness":      _safe("faithfulness"),
        "answer_relevancy":  _safe("answer_relevancy"),
        "context_precision": _safe("context_precision"),
        "context_recall":    _safe("context_recall"),
    }


def _evaluate_with_fallback(
    sample: SingleTurnSample,
    metrics: list,
    judge_chain: list[tuple[str, LangchainLLMWrapper]],
    ragas_emb: LangchainEmbeddingsWrapper,
) -> tuple[dict[str, float | None], str]:
    """Coba setiap model hingga answer_relevancy non-None.

    Return (scores_dict, model_used).
    """
    for model_name, ragas_llm in judge_chain:
        scores = _evaluate_one(sample, metrics, ragas_llm, ragas_emb)
        if scores["answer_relevancy"] is not None:
            return scores, model_name
        print(f"→fallback({model_name} gagal) ", end="", flush=True)
        time.sleep(SLEEP_AFTER_FALLBACK)
    return {k: None for k in ["faithfulness", "answer_relevancy",
                              "context_precision", "context_recall"]}, "all_failed"


def _cleanup_and_load_completed() -> set[tuple[str, str]]:
    if not OUT_PER_Q.exists():
        return set()
    valid_rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    with OUT_PER_Q.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("answer_relevancy"):
                cleaned = {col: row.get(col, "") for col in OUT_COLS}
                valid_rows.append(cleaned)
                done.add((row["mode"], row["question_id"]))
    with OUT_PER_Q.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS)
        writer.writeheader()
        writer.writerows(valid_rows)
    return done


def _append_row(row: dict) -> None:
    file_exists = OUT_PER_Q.exists() and OUT_PER_Q.stat().st_size > 0
    with OUT_PER_Q.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} belum ada. Jalankan extract_contexts_full_v2 dulu.",
              file=sys.stderr)
        return 1

    # Load full_v2 dengan contexts
    with INPUT_CSV.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # Load ground truth
    dataset_items = {
        item["question_id"]: item
        for item in json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    }

    print("=== RAGAS 4-METRIC EVALUATION (full_v2, 200 rows) ===", flush=True)
    print(f"Input  : {INPUT_CSV}", flush=True)
    print(f"GT     : {DATASET_JSON}", flush=True)
    print(f"Output : {OUT_PER_Q}", flush=True)
    print(f"Embeddings : {EMBEDDING_MODEL_NAME}", flush=True)
    print(f"Sleep/sample : {SLEEP_BETWEEN_SAMPLES}s", flush=True)

    completed = _cleanup_and_load_completed()
    print(f"Resume : {len(completed)} sample valid", flush=True)

    print("Init embeddings...", flush=True)
    ragas_emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))

    print("Build judge chain...", flush=True)
    judge_chain = _build_judge_chain()
    if not judge_chain:
        print("ERROR: Tidak ada judge LLM. Set API key di .env.", flush=True)
        return 1
    print("Chain : " + " → ".join(n for n, _ in judge_chain), flush=True)

    # Group rows by mode untuk reporting
    by_mode: dict[str, list[dict]] = {m: [] for m in MODES_ORDER}
    for row in all_rows:
        if row["mode"] in by_mode:
            by_mode[row["mode"]].append(row)

    for mode in MODES_ORDER:
        rows = by_mode[mode]
        has_context = mode != "mode_1_llm_only"
        if has_context:
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        else:
            metrics = [answer_relevancy]

        print(f"\n--- {MODE_LABEL[mode]} ({len(rows)} sample) ---", flush=True)

        for idx, row in enumerate(rows):
            qid = row["question_id"]
            question = row["question"]
            answer = row["answer"]
            category = row.get("category", "")

            if (mode, qid) in completed:
                print(f"  [{idx+1:>2}/{len(rows)}] {qid} SKIP", flush=True)
                continue

            try:
                contexts = json.loads(row.get("contexts") or "[]")
            except json.JSONDecodeError:
                contexts = []
            ground_truth = dataset_items.get(qid, {}).get("ground_truth", "")

            sample_kwargs = {
                "user_input": question,
                "response":   answer,
            }
            if has_context:
                sample_kwargs["retrieved_contexts"] = contexts
                sample_kwargs["reference"] = ground_truth
            sample = SingleTurnSample(**sample_kwargs)

            print(f"  [{idx+1:>2}/{len(rows)}] {qid} ({len(contexts)} ctx) ... ",
                  end="", flush=True)

            scores, model_used = _evaluate_with_fallback(
                sample, metrics, judge_chain, ragas_emb
            )

            def fmt(v):
                return f"{v:.3f}" if v is not None else "nan"

            print(
                f"f={fmt(scores['faithfulness'])} "
                f"ar={fmt(scores['answer_relevancy'])} "
                f"cp={fmt(scores['context_precision'])} "
                f"cr={fmt(scores['context_recall'])} "
                f"[{model_used}]",
                flush=True,
            )

            _append_row({
                "mode":              mode,
                "question_id":       qid,
                "category":          category,
                "faithfulness":      round(scores["faithfulness"], 4) if scores["faithfulness"] is not None else "",
                "answer_relevancy":  round(scores["answer_relevancy"], 4) if scores["answer_relevancy"] is not None else "",
                "context_precision": round(scores["context_precision"], 4) if scores["context_precision"] is not None else "",
                "context_recall":    round(scores["context_recall"], 4) if scores["context_recall"] is not None else "",
                "model_used":        model_used,
            })
            completed.add((mode, qid))

            if idx < len(rows) - 1:
                time.sleep(SLEEP_BETWEEN_SAMPLES)

    # ----- Rekap per-mode -----
    print("\n=== REKAP PER MODE ===", flush=True)
    per_q_data: dict[str, list[dict]] = {m: [] for m in MODES_ORDER}
    with OUT_PER_Q.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["mode"] in per_q_data:
                per_q_data[row["mode"]].append(row)

    summary: list[dict] = []
    for mode in MODES_ORDER:
        rows = per_q_data[mode]
        has_ctx = mode != "mode_1_llm_only"

        def col_vals(name):
            return [float(r[name]) if r.get(name) else None for r in rows]

        f_vals = col_vals("faithfulness") if has_ctx else [None]*len(rows)
        a_vals = col_vals("answer_relevancy")
        cp_vals = col_vals("context_precision") if has_ctx else [None]*len(rows)
        cr_vals = col_vals("context_recall") if has_ctx else [None]*len(rows)

        models = {r["model_used"] for r in rows if r.get("model_used")}

        def stat(vals, label):
            m = _mean(vals); s = _std(vals)
            if m is None:
                return "N/A"
            return f"{m:.3f} ± {s:.3f}"

        print(f"  {MODE_LABEL[mode]}", flush=True)
        print(f"    faithfulness      : {stat(f_vals, 'f')}", flush=True)
        print(f"    answer_relevancy  : {stat(a_vals, 'a')}", flush=True)
        print(f"    context_precision : {stat(cp_vals, 'cp')}", flush=True)
        print(f"    context_recall    : {stat(cr_vals, 'cr')}", flush=True)
        print(f"    models used       : {', '.join(sorted(models)) or '-'}", flush=True)

        summary.append({
            "mode": MODE_LABEL[mode],
            "n_questions": len(rows),
            "faithfulness_mean": round(_mean(f_vals), 4) if _mean(f_vals) is not None else "",
            "faithfulness_sd":   round(_std(f_vals), 4)  if _std(f_vals)  is not None else "",
            "answer_relevancy_mean": round(_mean(a_vals), 4) if _mean(a_vals) is not None else "",
            "answer_relevancy_sd":   round(_std(a_vals), 4)  if _std(a_vals)  is not None else "",
            "context_precision_mean": round(_mean(cp_vals), 4) if _mean(cp_vals) is not None else "",
            "context_precision_sd":   round(_std(cp_vals), 4)  if _std(cp_vals)  is not None else "",
            "context_recall_mean":   round(_mean(cr_vals), 4) if _mean(cr_vals) is not None else "",
            "context_recall_sd":     round(_std(cr_vals), 4)  if _std(cr_vals)  is not None else "",
            "models_used": ", ".join(sorted(models)),
        })

    cols = ["mode", "n_questions",
            "faithfulness_mean", "faithfulness_sd",
            "answer_relevancy_mean", "answer_relevancy_sd",
            "context_precision_mean", "context_precision_sd",
            "context_recall_mean", "context_recall_sd",
            "models_used"]
    with OUT_PER_MODE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nPer-question : {OUT_PER_Q}", flush=True)
    print(f"Per-mode     : {OUT_PER_MODE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
