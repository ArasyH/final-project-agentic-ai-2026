"""app/analyze_results.py — Analisis deskriptif hasil eksperimen 50Q × 4 mode.

Menghasilkan semua tabel Bab IV sesuai §15 SINTA 2:
  - Hallucination rate per mode
  - Latency: mean ± SD, median (IQR) per mode
  - Distribusi H1–H4 per mode
  - Cache hit rate mode_4
  - Stratifikasi per kategori pertanyaan
  - Selisih absolut + relatif (%) antar mode vs mode_1 (baseline)

Output:
  - Tabel di terminal
  - app/data/analysis_*.csv untuk import Excel/Word

Jalankan:
    source venv/bin/activate
    python3 -m app.analyze_results
"""
from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

INPUT_PATH  = Path(__file__).parent / "data" / "experiment_results.csv"
OUTPUT_DIR  = Path(__file__).parent / "data"

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
HALLUCINATION_CODES = ["H1", "H2", "H3", "H4"]
HALLUCINATION_LABEL = {
    "H1": "H1 Unsupported Numeric",
    "H2": "H2 Fabricated Metric",
    "H3": "H3 Stale Timestamp",
    "H4": "H4 Incorrect Inference",
}

SEP  = "=" * 72
SEP2 = "-" * 72


# ---------------------------------------------------------------------------
# Statistik helpers
# ---------------------------------------------------------------------------

def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

def median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]

def iqr(vals: list[float]) -> tuple[float, float]:
    """Kembalikan (Q1, Q3)."""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0, 0.0
    q1 = median(s[: n // 2])
    q3 = median(s[(n + 1) // 2 :])
    return q1, q3

def pct_diff(base: float, compare: float) -> str:
    """Selisih relatif compare vs base dalam persen."""
    if base == 0:
        return "N/A"
    diff = compare - base
    pct  = diff / base * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{pct:.1f}%"

def abs_diff(base: float, compare: float) -> str:
    diff = compare - base
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f}"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load() -> dict[str, list[dict]]:
    """Baca CSV, return dict mode → list of rows."""
    data: dict[str, list[dict]] = defaultdict(list)
    with INPUT_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["mode"]].append(row)
    return data


# ---------------------------------------------------------------------------
# Tabel 1 — Hallucination rate per mode
# ---------------------------------------------------------------------------

def table_hallucination_rate(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 1 — Hallucination Rate per Mode")
    print(SEP)
    header = f"{'Mode':<28} {'Q dgn Flags':>12} {'Total Q':>8} {'Rate':>8} {'vs Mode1':>10}"
    print(header)
    print(SEP2)

    rows_out = []
    baseline_rate = None

    for mode in MODES_ORDER:
        rows = data.get(mode, [])
        total = len(rows)
        flagged = sum(1 for r in rows if r["hallucination_flags"].strip())
        rate = flagged / total * 100 if total else 0.0

        vs = "-" if baseline_rate is None else pct_diff(baseline_rate, rate)
        if baseline_rate is None:
            baseline_rate = rate

        print(f"{MODE_LABEL[mode]:<28} {flagged:>12} {total:>8} {rate:>7.1f}% {vs:>10}")
        rows_out.append({
            "mode": MODE_LABEL[mode],
            "q_dengan_flags": flagged,
            "total_q": total,
            "hallucination_rate_pct": round(rate, 2),
            "vs_mode1": vs,
        })

    return rows_out


# ---------------------------------------------------------------------------
# Tabel 2 — Latency descriptive per mode
# ---------------------------------------------------------------------------

def table_latency(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 2 — Latency (ms): Mean ± SD, Median (IQR)")
    print(SEP)
    header = (
        f"{'Mode':<28} {'Mean':>9} {'SD':>8} {'Median':>9} "
        f"{'Q1':>8} {'Q3':>8} {'vs Mode1 Mean':>14}"
    )
    print(header)
    print(SEP2)

    rows_out = []
    baseline_mean = None

    for mode in MODES_ORDER:
        rows = data.get(mode, [])
        lats = [float(r["latency_ms_total"]) for r in rows if r["latency_ms_total"]]
        m   = mean(lats)
        s   = std(lats)
        med = median(lats)
        q1, q3 = iqr(lats)

        vs = "-" if baseline_mean is None else pct_diff(baseline_mean, m)
        if baseline_mean is None:
            baseline_mean = m

        print(
            f"{MODE_LABEL[mode]:<28} {m:>8.0f}ms {s:>7.0f}ms "
            f"{med:>8.0f}ms {q1:>7.0f}ms {q3:>7.0f}ms {vs:>14}"
        )
        rows_out.append({
            "mode": MODE_LABEL[mode],
            "mean_ms": round(m, 1),
            "sd_ms": round(s, 1),
            "median_ms": round(med, 1),
            "q1_ms": round(q1, 1),
            "q3_ms": round(q3, 1),
            "vs_mode1_mean": vs,
        })

    return rows_out


# ---------------------------------------------------------------------------
# Tabel 3 — Distribusi H1–H4 per mode
# ---------------------------------------------------------------------------

def table_flag_distribution(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 3 — Distribusi Kategori Halusinasi (H1–H4) per Mode")
    print(SEP)
    header = f"{'Mode':<28} {'H1':>6} {'H2':>6} {'H3':>6} {'H4':>6} {'Total Flags':>12}"
    print(header)
    print(SEP2)

    rows_out = []
    for mode in MODES_ORDER:
        rows = data.get(mode, [])
        counts = {h: 0 for h in HALLUCINATION_CODES}
        for r in rows:
            for flag in r["hallucination_flags"].split(","):
                flag = flag.strip()
                if flag in counts:
                    counts[flag] += 1
        total_flags = sum(counts.values())
        print(
            f"{MODE_LABEL[mode]:<28} "
            f"{counts['H1']:>6} {counts['H2']:>6} {counts['H3']:>6} {counts['H4']:>6} "
            f"{total_flags:>12}"
        )
        rows_out.append({
            "mode": MODE_LABEL[mode],
            "H1": counts["H1"],
            "H2": counts["H2"],
            "H3": counts["H3"],
            "H4": counts["H4"],
            "total_flags": total_flags,
        })

    return rows_out


# ---------------------------------------------------------------------------
# Tabel 4 — Cache hit rate mode_4
# ---------------------------------------------------------------------------

def table_cache(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 4 — Cache Status Mode 4 (RAG+J&C+Cache)")
    print(SEP)

    rows = data.get("mode_4_rag_jc_cache", [])
    total = len(rows)
    hits   = sum(1 for r in rows if r["cache_status"] == "hit")
    misses = sum(1 for r in rows if r["cache_status"] == "miss")
    stales = sum(1 for r in rows if r["cache_status"] == "stale")

    hit_rate = hits / total * 100 if total else 0.0
    print(f"  Total Q     : {total}")
    print(f"  Cache hit   : {hits}  ({hit_rate:.1f}%)")
    print(f"  Cache miss  : {misses}  ({100 - hit_rate:.1f}%)")
    print(f"  Cache stale : {stales}")

    rows_out = [{"status": "hit", "count": hits, "pct": round(hit_rate, 2)},
                {"status": "miss", "count": misses, "pct": round(100 - hit_rate, 2)},
                {"status": "stale", "count": stales, "pct": 0.0}]
    return rows_out


# ---------------------------------------------------------------------------
# Tabel 5 — Hallucination rate per kategori × mode
# ---------------------------------------------------------------------------

def table_by_category(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 5 — Hallucination Rate per Kategori Pertanyaan × Mode (%)")
    print(SEP)

    # Kumpulkan semua kategori
    all_cats: set[str] = set()
    for rows in data.values():
        for r in rows:
            if r.get("category"):
                all_cats.add(r["category"])
    cats = sorted(all_cats)

    # Header
    mode_cols = [MODE_LABEL[m][:18] for m in MODES_ORDER]
    header = f"{'Kategori':<22} " + " ".join(f"{c:>20}" for c in mode_cols)
    print(header)
    print(SEP2)

    rows_out = []
    for cat in cats:
        rates = []
        for mode in MODES_ORDER:
            subset = [r for r in data.get(mode, []) if r.get("category") == cat]
            total  = len(subset)
            flagged = sum(1 for r in subset if r["hallucination_flags"].strip())
            rate = flagged / total * 100 if total else 0.0
            rates.append(rate)
        rate_strs = " ".join(f"{r:>19.1f}%" for r in rates)
        print(f"{cat:<22} {rate_strs}")
        rows_out.append({"category": cat, **{MODES_ORDER[i]: round(rates[i], 2) for i in range(4)}})

    return rows_out


# ---------------------------------------------------------------------------
# Tabel 6 — Validator status per mode
# ---------------------------------------------------------------------------

def table_validator(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 6 — Validator Status per Mode")
    print(SEP)
    header = f"{'Mode':<28} {'passed':>8} {'failed':>8} {'skipped':>8}"
    print(header)
    print(SEP2)

    rows_out = []
    for mode in MODES_ORDER:
        rows = data.get(mode, [])
        passed  = sum(1 for r in rows if r["validator_status"] == "passed")
        failed  = sum(1 for r in rows if r["validator_status"] == "failed")
        skipped = sum(1 for r in rows if r["validator_status"] == "skipped")
        print(f"{MODE_LABEL[mode]:<28} {passed:>8} {failed:>8} {skipped:>8}")
        rows_out.append({"mode": MODE_LABEL[mode], "passed": passed, "failed": failed, "skipped": skipped})

    return rows_out


# ---------------------------------------------------------------------------
# Ringkasan naratif
# ---------------------------------------------------------------------------

def narasi(data: dict) -> None:
    print(f"\n{SEP}")
    print("RINGKASAN NARATIF (untuk Bab IV)")
    print(SEP)

    for mode in MODES_ORDER:
        rows = data.get(mode, [])
        total   = len(rows)
        flagged = sum(1 for r in rows if r["hallucination_flags"].strip())
        rate    = flagged / total * 100 if total else 0.0
        lats    = [float(r["latency_ms_total"]) for r in rows if r["latency_ms_total"]]
        m       = mean(lats)
        s       = std(lats)
        med     = median(lats)
        print(
            f"  {MODE_LABEL[mode]}: hallucination rate {rate:.1f}% ({flagged}/{total} Q), "
            f"latency mean {m:.0f}ms ± {s:.0f}ms, median {med:.0f}ms"
        )

    # Mode 4 cache
    m4 = data.get("mode_4_rag_jc_cache", [])
    hits = sum(1 for r in m4 if r["cache_status"] == "hit")
    print(f"\n  Cache hit rate Mode 4: {hits}/{len(m4)} = {hits/len(m4)*100:.1f}%")

    # Selisih mode_3 vs mode_1
    def rate(mode):
        rows = data.get(mode, [])
        return sum(1 for r in rows if r["hallucination_flags"].strip()) / len(rows) * 100

    r1, r3, r4 = rate("mode_1_llm_only"), rate("mode_3_rag_jc"), rate("mode_4_rag_jc_cache")
    print(f"\n  Reduksi halusinasi mode_3 vs mode_1: "
          f"{abs(r3 - r1):.1f} poin absolut ({pct_diff(r1, r3)} relatif)")
    print(f"  Reduksi halusinasi mode_4 vs mode_1: "
          f"{abs(r4 - r1):.1f} poin absolut ({pct_diff(r1, r4)} relatif)")


# ---------------------------------------------------------------------------
# Simpan CSV output
# ---------------------------------------------------------------------------

def save_csv(filename: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = OUTPUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Disimpan: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} tidak ditemukan. Jalankan eksperimen dulu.")
        return

    data = load()
    total = sum(len(v) for v in data.values())
    print(f"\n{SEP}")
    print(f"ANALISIS HASIL EKSPERIMEN — {total} baris dari {INPUT_PATH.name}")
    print(SEP)

    t1 = table_hallucination_rate(data)
    t2 = table_latency(data)
    t3 = table_flag_distribution(data)
    t4 = table_cache(data)
    t5 = table_by_category(data)
    t6 = table_validator(data)
    narasi(data)

    print(f"\n{SEP}")
    print("MENYIMPAN CSV OUTPUT ...")
    save_csv("analysis_hallucination_rate.csv", t1)
    save_csv("analysis_latency.csv", t2)
    save_csv("analysis_flag_distribution.csv", t3)
    save_csv("analysis_cache.csv", t4)
    save_csv("analysis_by_category.csv", t5)
    save_csv("analysis_validator.csv", t6)
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
