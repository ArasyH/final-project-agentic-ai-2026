"""app/run_experiment_v2.py — Runner eksperimen Mode 3 & 4 dengan prompt V2.

Menjalankan hanya mode_3_rag_jc dan mode_4_rag_jc_cache menggunakan
REACT_PROMPT_V2 + CRITIC_PROMPT_V2 (prompt yang diperbarui).

Tujuan: perbandingan V1 vs V2 tanpa menimpa data eksperimen asli (V1).
Output: app/data/experiment_results_v2.csv (100 baris: 50Q × 2 mode).

Struktur kolom identik dengan experiment_results.csv (V1) untuk kemudahan
analisis komparatif di pandas/Excel.

Fitur:
- CSV ditulis per-baris (checkpoint) — aman jika proses di-interrupt
- Resume otomatis: skip (mode, question_id) yang sudah ada di CSV V2
- stdout di-flush setiap baris — progress terlihat real-time

Jalankan:
    source venv/bin/activate
    python3 -m app.run_experiment_v2
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app.modes.mode_3_rag_jc import run_mode_3
from app.modes.mode_4_rag_jc_cache import run_mode_4
from app.schemas import InternalResponse

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
DATASET_PATH = Path(__file__).parent / "data" / "evaluation_dataset.json"
# V6: file baru — experiment_results_v5.csv berisi data react_v5+critic_v3 (FP tinggi).
OUTPUT_PATH = Path(__file__).parent / "data" / "experiment_results_v6.csv"

# Jeda antar-pertanyaan (detik) — cegah Groq rate-limit 429
SLEEP_BETWEEN_QUESTIONS: float = 3.0
# Jeda antar-mode (detik)
SLEEP_BETWEEN_MODES: float = 10.0

# Hanya mode yang terpengaruh perubahan prompt V2
MODES = [
    ("mode_3_rag_jc", run_mode_3),
    ("mode_4_rag_jc_cache", run_mode_4),
]

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
    f"exp-v6-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
)


# ---------------------------------------------------------------------------
# Helpers (identik dengan run_experiment.py untuk konsistensi)
# ---------------------------------------------------------------------------

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
    t0 = time.perf_counter()
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
    """Jalankan eksperimen 50 pertanyaan × 2 mode (mode_3, mode_4) dengan prompt V2.

    Resume otomatis: skip (mode, question_id) yang sudah ada di OUTPUT_PATH
    sehingga proses bisa dilanjutkan setelah interrupt.
    """
    # Import di sini untuk membaca PROMPT_VERSION aktif saat runtime
    from app.agents.generator_agent import PROMPT_VERSION as gen_pv
    from app.agents.critic_agent import PROMPT_VERSION as crit_pv
    prompt_version_tag = f"gen={gen_pv},crit={crit_pv}"

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    total_q = len(dataset)
    total_runs = total_q * len(MODES)
    completed = _load_completed(OUTPUT_PATH)
    skipped = len(completed)

    print("=== EKSPERIMEN V6 MULAI ===", flush=True)
    print(f"Prompt version : {prompt_version_tag}", flush=True)
    print(f"Session ID     : {EXPERIMENT_SESSION_ID}", flush=True)
    print(f"Pertanyaan     : {total_q}", flush=True)
    print(f"Mode           : {len(MODES)} (mode_3 + mode_4 saja)", flush=True)
    print(f"Total runs     : {total_runs}", flush=True)
    print(f"Sudah selesai  : {skipped} (di-skip)", flush=True)
    print(f"Output         : {OUTPUT_PATH}", flush=True)
    print(f"Mulai          : {datetime.now(timezone.utc).isoformat()}", flush=True)
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
            question = item["question"]
            category = item.get("category", "")

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
    rows_with_flags = sum(1 for r in all_rows if r.get("hallucination_flags"))

    m3_flags = sum(
        1 for r in all_rows
        if r.get("mode") == "mode_3_rag_jc" and r.get("hallucination_flags")
    )
    m4_flags = sum(
        1 for r in all_rows
        if r.get("mode") == "mode_4_rag_jc_cache" and r.get("hallucination_flags")
    )

    print("=== EKSPERIMEN V6 SELESAI ===", flush=True)
    print(f"Selesai    : {datetime.now(timezone.utc).isoformat()}", flush=True)
    print(f"Total rows : {len(all_rows)}", flush=True)
    print(f"Errors     : {total_errors}", flush=True)
    print(f"Rows dgn hallucination flags  : {rows_with_flags}", flush=True)
    print(f"  Mode 3 dengan flags : {m3_flags}/{total_q}", flush=True)
    print(f"  Mode 4 dengan flags : {m4_flags}/{total_q}", flush=True)
    print(f"CSV        : {OUTPUT_PATH}", flush=True)
    print(flush=True)
    print("Perbandingan dengan V1 (dari experiment_results.csv):", flush=True)
    print("  Mode 3 V1 = 9/50 (18%)  →  Mode 3 V6 = ?", flush=True)
    print("  Mode 4 V1 = 7/50 (14%)  →  Mode 4 V6 = ?", flush=True)


if __name__ == "__main__":
    main()
