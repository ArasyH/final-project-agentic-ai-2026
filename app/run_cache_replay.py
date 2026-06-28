"""app/run_cache_replay.py — Mini-eksperimen replay untuk cache hit ratio realistis.

Tujuan: membuktikan mekanisme semantic cache di luar dataset paired-unique (V6).
30 pertanyaan = 10 original + 20 parafrase (2 per original) dari
`app/data/cache_replay_dataset.json`.

Skema eksekusi (Mode 4 saja):
  Phase 1 — 10 ORIGINAL berurutan → cache populate (jika Critic passed)
  Phase 2 — 20 PARAFRASE berurutan → ukur hit rate (similarity ≥ 0,85)

Output: app/data/cache_replay_results.csv
  Kolom tambahan vs CSV eksperimen utama: origin_qid, variant, phase.

Resume-safe: skip (question_id) yang sudah ada di CSV output.

Jalankan:
    source venv/bin/activate
    python3 -m app.run_cache_replay
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app.modes.mode_4_rag_jc_cache import run_mode_4
from app.schemas import InternalResponse

DATASET_PATH = Path(__file__).parent / "data" / "cache_replay_dataset.json"
OUTPUT_PATH = Path(__file__).parent / "data" / "cache_replay_results.csv"

SLEEP_BETWEEN_QUESTIONS: float = 3.0  # Hindari Groq rate-limit
SLEEP_BETWEEN_PHASES: float = 5.0

CSV_COLUMNS = [
    "question_id",
    "origin_qid",
    "variant",
    "phase",
    "question",
    "category",
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
    f"cache-replay-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
)


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("question_id"):
                done.add(row["question_id"])
    return done


def _append_row(path: Path, row: dict) -> None:
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _run_one(item: dict, phase: str, prompt_version: str) -> dict:
    """Jalankan satu pertanyaan di Mode 4, tangkap latency + error."""
    qid = item["question_id"]
    t0 = time.perf_counter()
    error = ""
    result: InternalResponse | None = None

    try:
        result = run_mode_4(item["question"], EXPERIMENT_SESSION_ID, qid)
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:200]}"

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    base = {
        "question_id":  qid,
        "origin_qid":   item["origin_qid"],
        "variant":      item["variant"],
        "phase":        phase,
        "question":     item["question"],
        "category":     item["category"],
        "prompt_version": prompt_version,
    }

    if result is not None:
        base.update({
            "answer":              result.answer,
            "hallucination_flags": ",".join(result.hallucination_flags),
            "validator_status":    result.validator_status,
            "cache_status":        result.cache_status,
            "confidence":          result.confidence,
            "latency_ms_total":    latency_ms,
            "evidence_count":      len(result.evidence),
            "iterations_used":     result.metadata.get("iterations_used", ""),
            "error":               "",
        })
    else:
        base.update({
            "answer":              "",
            "hallucination_flags": "",
            "validator_status":    "failed",
            "cache_status":        "miss",
            "confidence":          0.0,
            "latency_ms_total":    latency_ms,
            "evidence_count":      0,
            "iterations_used":     "",
            "error":               error,
        })
    return base


def main() -> int:
    from app.agents.generator_agent import PROMPT_VERSION as gen_pv
    from app.agents.critic_agent import PROMPT_VERSION as crit_pv
    prompt_version_tag = f"gen={gen_pv},crit={crit_pv}"

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    originals = [e for e in dataset if e["variant"] == "original"]
    paraphrases = [e for e in dataset if e["variant"] != "original"]

    completed = _load_completed(OUTPUT_PATH)
    skipped = len(completed)

    print("=== CACHE REPLAY EXPERIMENT MULAI ===", flush=True)
    print(f"Prompt version : {prompt_version_tag}", flush=True)
    print(f"Session ID     : {EXPERIMENT_SESSION_ID}", flush=True)
    print(f"Originals      : {len(originals)}", flush=True)
    print(f"Parafrase      : {len(paraphrases)}", flush=True)
    print(f"Sudah selesai  : {skipped} (di-skip)", flush=True)
    print(f"Output         : {OUTPUT_PATH}", flush=True)
    print(f"Mulai          : {datetime.now(timezone.utc).isoformat()}", flush=True)
    print(flush=True)

    # PHASE 1 — originals
    print("--- PHASE 1: ORIGINALS (cache populate) ---", flush=True)
    for idx, item in enumerate(originals):
        if item["question_id"] in completed:
            continue
        print(f"  [{idx+1}/{len(originals)}] {item['question_id']:8} | "
              f"{item['question'][:55]}{'...' if len(item['question']) > 55 else ''}",
              end="", flush=True)
        row = _run_one(item, "phase1_original", prompt_version_tag)
        _append_row(OUTPUT_PATH, row)
        completed.add(item["question_id"])
        icon = "✗" if row["error"] else "✓"
        print(f" {icon} cache={row['cache_status']:8} verdict={row['validator_status']:6} "
              f"lat={row['latency_ms_total']}ms", flush=True)
        if row["error"]:
            print(f"    ERROR: {row['error']}", flush=True)
        if idx < len(originals) - 1:
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

    print(f"\n  Jeda {SLEEP_BETWEEN_PHASES}s sebelum phase 2...\n", flush=True)
    time.sleep(SLEEP_BETWEEN_PHASES)

    # PHASE 2 — paraphrases
    print("--- PHASE 2: PARAPHRASES (cache hit test) ---", flush=True)
    for idx, item in enumerate(paraphrases):
        if item["question_id"] in completed:
            continue
        print(f"  [{idx+1}/{len(paraphrases)}] {item['question_id']:10} | "
              f"{item['question'][:55]}{'...' if len(item['question']) > 55 else ''}",
              end="", flush=True)
        row = _run_one(item, "phase2_paraphrase", prompt_version_tag)
        _append_row(OUTPUT_PATH, row)
        completed.add(item["question_id"])
        icon = "✗" if row["error"] else "✓"
        print(f" {icon} cache={row['cache_status']:8} verdict={row['validator_status']:6} "
              f"lat={row['latency_ms_total']}ms", flush=True)
        if row["error"]:
            print(f"    ERROR: {row['error']}", flush=True)
        if idx < len(paraphrases) - 1:
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

    # Ringkasan
    with OUTPUT_PATH.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    p1 = [r for r in all_rows if r["phase"] == "phase1_original"]
    p2 = [r for r in all_rows if r["phase"] == "phase2_paraphrase"]
    p1_pass = sum(1 for r in p1 if r["validator_status"] == "passed")
    p1_hit = sum(1 for r in p1 if r["cache_status"] == "hit")
    p2_hit = sum(1 for r in p2 if r["cache_status"] == "hit")

    print("\n=== CACHE REPLAY SELESAI ===", flush=True)
    print(f"Selesai    : {datetime.now(timezone.utc).isoformat()}", flush=True)
    print(f"Phase 1 originals  : {len(p1)} | passed={p1_pass} | hit={p1_hit}", flush=True)
    print(f"Phase 2 parafrase  : {len(p2)} | hit={p2_hit}/{len(p2)} "
          f"= {p2_hit/max(len(p2),1)*100:.1f}%", flush=True)
    print(f"CSV        : {OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())