"""app/run_experiment_full_v2.py — Runner eksperimen ulang 50Q × 4 mode (F2).

Menjalankan seluruh 4 mode menggunakan prompt aktif saat ini:
  Generator : REACT_PROMPT_V5  (react_v5) — basis V1 + ticker hint + timestamp
  Critic    : CRITIC_PROMPT_V4 (critic_v4) — basis V1 + klarifikasi H2 + H4=false jika ev kosong

Output: app/data/experiment_results_full_v2.csv (200 baris: 50Q × 4 mode).

Tujuan (F2): mendapatkan baseline baru yang konsisten dengan KB Juni 2026,
sehingga perbandingan antar-mode valid (KB identik untuk semua 4 mode).

Kolom identik dengan experiment_results.csv + satu kolom tambahan:
    prompt_version: "gen=react_v5,crit=critic_v4" (hanya relevan untuk mode_3/4)

Fitur:
- CSV ditulis per-baris (checkpoint) — aman jika proses di-interrupt
- Resume otomatis: skip (mode, question_id) yang sudah ada di CSV
- stdout di-flush setiap baris — progress terlihat real-time

Jalankan:
    source venv/bin/activate
    python3 -m app.run_experiment_full_v2
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.modes.mode_1_llm_only import run_mode_1
from app.modes.mode_2_rag_only import run_mode_2
from app.modes.mode_3_rag_jc import run_mode_3
from app.modes.mode_4_rag_jc_cache import run_mode_4
from app.schemas import InternalResponse

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
DATASET_PATH = Path(__file__).parent / "data" / "evaluation_dataset.json"
OUTPUT_PATH  = Path(__file__).parent / "data" / "experiment_results_full_v2.csv"

SLEEP_BETWEEN_QUESTIONS: float = 3.0
SLEEP_BETWEEN_MODES: float = 10.0

MODES = [
    ("mode_1_llm_only",      run_mode_1),
    ("mode_2_rag_only",      run_mode_2),
    ("mode_3_rag_jc",        run_mode_3),
    ("mode_4_rag_jc_cache",  run_mode_4),
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
    f"exp-full-v2-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
)


# ---------------------------------------------------------------------------
# Helpers
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
    """Jalankan eksperimen ulang 50Q × 4 mode dengan KB dan prompt aktif saat ini.

    Resume otomatis: skip (mode, question_id) yang sudah ada di OUTPUT_PATH.
    Kolom prompt_version mencatat versi prompt Generator dan Critic aktif
    (hanya relevan untuk mode_3 dan mode_4; mode_1/2 diisi 'n/a').
    """
    from app.agents.generator_agent import PROMPT_VERSION as gen_pv
    from app.agents.critic_agent import PROMPT_VERSION as crit_pv

    prompt_version_m3m4 = f"gen={gen_pv},crit={crit_pv}"
    prompt_version_m1m2 = "n/a"

    mode_prompt_map = {
        "mode_1_llm_only":     prompt_version_m1m2,
        "mode_2_rag_only":     prompt_version_m1m2,
        "mode_3_rag_jc":       prompt_version_m3m4,
        "mode_4_rag_jc_cache": prompt_version_m3m4,
    }

    dataset    = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    total_q    = len(dataset)
    total_runs = total_q * len(MODES)
    completed  = _load_completed(OUTPUT_PATH)
    skipped    = len(completed)

    print("=== EKSPERIMEN FULL V2 MULAI ===", flush=True)
    print(f"Prompt aktif   : {prompt_version_m3m4} (mode_3 & mode_4)", flush=True)
    print(f"Session ID     : {EXPERIMENT_SESSION_ID}", flush=True)
    print(f"Pertanyaan     : {total_q}", flush=True)
    print(f"Mode           : {len(MODES)} (semua 4 mode)", flush=True)
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
        pv = mode_prompt_map[mode_name]

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
                prompt_version=pv,
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

    print("=== EKSPERIMEN FULL V2 SELESAI ===", flush=True)
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
        print(
            f"  {mode_name}: halluc={halluc}/{n} "
            f"passed={passed}/{n} "
            f"ev0={ev0}/{n}",
            flush=True,
        )


if __name__ == "__main__":
    main()
