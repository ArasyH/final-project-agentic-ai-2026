"""app/analyze_results.py — Analisis deskriptif hasil eksperimen 50Q × 4 mode.

Menghasilkan semua tabel Bab IV sesuai §15 SINTA 2:
  - Hallucination rate per mode (any-flag, per H1–H4)
  - Latency: mean ± SD, median (IQR) per mode
  - Evidence count per mode
  - Confidence per mode
  - Iterations used (mode_3/4 ReAct)
  - Cache hit rate mode_4
  - Stratifikasi per kategori pertanyaan
  - Selisih absolut + relatif (%) antar mode vs mode_1 (baseline)
  - Perbandingan Groq vs Gemini (jika experiment_results_gemini.csv tersedia)
  - Paired analysis per question_id

Output:
  - Tabel di terminal
  - app/data/analysis_*.csv untuk import Excel/Word

Jalankan:
    source venv/bin/activate

    # Default: experiment_results_full_v2.csv
    python3 -m app.analyze_results

    # File lain:
    python3 -m app.analyze_results --file experiment_results_v6.csv

    # Dengan perbandingan Gemini:
    python3 -m app.analyze_results --gemini
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

DATA_DIR    = Path(__file__).parent / "data"
OUTPUT_DIR  = DATA_DIR

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
H_CODES  = ["H1", "H2", "H3", "H4"]
H_LABELS = {
    "H1": "H1 Unsupported Numeric",
    "H2": "H2 Fabricated Metric",
    "H3": "H3 Stale Timestamp",
    "H4": "H4 Incorrect Inference",
}

SEP  = "=" * 76
SEP2 = "-" * 76


# ---------------------------------------------------------------------------
# Statistik helpers (stdlib only — tidak perlu numpy untuk ini)
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2.0 if n % 2 == 0 else s[mid]

def _iqr(vals: list[float]) -> tuple[float, float]:
    """Kembalikan (Q1, Q3)."""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0, 0.0
    q1 = _median(s[: n // 2])
    q3 = _median(s[(n + 1) // 2 :])
    return q1, q3

def _pct_diff(base: float, compare: float) -> str:
    """Selisih relatif compare vs base dalam persen (dengan tanda)."""
    if base == 0:
        return "N/A"
    diff = compare - base
    pct  = diff / base * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{pct:.1f}%"

def _abs_diff_str(base: float, compare: float, unit: str = "") -> str:
    diff = compare - base
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f}{unit}"

def _desc(vals: list[float]) -> str:
    """Ringkas: 'mean ± SD (median [Q1–Q3])'."""
    if not vals:
        return "N/A"
    m   = _mean(vals)
    s   = _std(vals)
    med = _median(vals)
    q1, q3 = _iqr(vals)
    return f"{m:.1f} ± {s:.1f} (median {med:.1f} [{q1:.1f}–{q3:.1f}])"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict[str, list[dict]]:
    """Baca CSV, return dict mode → list of rows."""
    data: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["mode"]].append(row)
    return data

def _floats(rows: list[dict], col: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(col, "")
        try:
            if v not in (None, "", "N/A"):
                out.append(float(v))
        except (ValueError, TypeError):
            pass
    return out

def _flags_in(row: dict) -> list[str]:
    raw = row.get("hallucination_flags", "")
    return [f.strip() for f in raw.split(",") if f.strip()]


# ---------------------------------------------------------------------------
# Tabel 1 — Hallucination rate per mode
# ---------------------------------------------------------------------------

def table_hallucination_rate(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 1 — Hallucination Rate per Mode")
    print(SEP)
    print(f"{'Mode':<26} {'Q Flagged':>10} {'Total':>7} {'Rate%':>7} {'vs Mode1 (pp)':>15} {'vs Mode1 (%)':>13}")
    print(SEP2)

    rows_out: list[dict] = []
    baseline_rate: float | None = None

    for mode in MODES_ORDER:
        rows   = data.get(mode, [])
        total  = len(rows)
        flagged = sum(1 for r in rows if _flags_in(r))
        rate   = flagged / total * 100 if total else 0.0

        abs_d = "-"
        rel_d = "-"
        if baseline_rate is not None:
            abs_d = _abs_diff_str(baseline_rate, rate, " pp")
            rel_d = _pct_diff(baseline_rate, rate)
        else:
            baseline_rate = rate

        print(f"{MODE_LABEL[mode]:<26} {flagged:>10} {total:>7} {rate:>6.1f}% {abs_d:>15} {rel_d:>13}")
        rows_out.append({
            "mode":                  MODE_LABEL[mode],
            "q_flagged":             flagged,
            "total_q":               total,
            "hallucination_rate_pct": round(rate, 2),
            "selisih_absolut_pp":    abs_d,
            "selisih_relatif_pct":   rel_d,
        })

    return rows_out


# ---------------------------------------------------------------------------
# Tabel 2 — Distribusi H1–H4 per mode
# ---------------------------------------------------------------------------

def table_flag_distribution(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 2 — Distribusi Kategori Halusinasi H1–H4 per Mode")
    print(SEP)
    print(f"{'Mode':<26} {'H1':>6} {'H2':>6} {'H3':>6} {'H4':>6} {'Total':>7}")
    print(SEP2)

    rows_out: list[dict] = []
    for mode in MODES_ORDER:
        rows   = data.get(mode, [])
        counts = {h: 0 for h in H_CODES}
        for r in rows:
            for flag in _flags_in(r):
                if flag in counts:
                    counts[flag] += 1
        total_flags = sum(counts.values())
        print(
            f"{MODE_LABEL[mode]:<26} "
            f"{counts['H1']:>6} {counts['H2']:>6} {counts['H3']:>6} {counts['H4']:>6} {total_flags:>7}"
        )
        rows_out.append({
            "mode": MODE_LABEL[mode],
            **counts,
            "total_flags": total_flags,
        })

    return rows_out


# ---------------------------------------------------------------------------
# Tabel 3 — Latency per mode
# ---------------------------------------------------------------------------

def table_latency(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 3 — Latency (ms): Mean ± SD, Median [Q1–Q3]")
    print(SEP)
    print(f"{'Mode':<26} {'Mean':>8} {'SD':>8} {'Median':>9} {'Q1':>8} {'Q3':>8} {'vs Mode1':>10}")
    print(SEP2)

    rows_out: list[dict] = []
    baseline_mean: float | None = None

    for mode in MODES_ORDER:
        rows = data.get(mode, [])
        lats = _floats(rows, "latency_ms_total")
        m    = _mean(lats)
        s    = _std(lats)
        med  = _median(lats)
        q1, q3 = _iqr(lats)

        vs = "-"
        if baseline_mean is not None:
            vs = _pct_diff(baseline_mean, m)
        else:
            baseline_mean = m

        print(f"{MODE_LABEL[mode]:<26} {m:>7.0f}ms {s:>7.0f}ms {med:>8.0f}ms {q1:>7.0f}ms {q3:>7.0f}ms {vs:>10}")
        rows_out.append({
            "mode":         MODE_LABEL[mode],
            "mean_ms":      round(m, 1),
            "sd_ms":        round(s, 1),
            "median_ms":    round(med, 1),
            "q1_ms":        round(q1, 1),
            "q3_ms":        round(q3, 1),
            "vs_mode1_pct": vs,
        })

    return rows_out


# ---------------------------------------------------------------------------
# Tabel 4 — Evidence count per mode
# ---------------------------------------------------------------------------

def table_evidence(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 4 — Evidence Count per Mode (jumlah chunk KB diambil)")
    print(SEP)
    print(f"{'Mode':<26} {'Mean':>7} {'SD':>7} {'Median':>8} {'Q1':>6} {'Q3':>6} {'ev=0':>7} {'ev=0%':>7}")
    print(SEP2)

    rows_out: list[dict] = []
    for mode in MODES_ORDER:
        rows = data.get(mode, [])
        evs  = _floats(rows, "evidence_count")
        m    = _mean(evs)
        s    = _std(evs)
        med  = _median(evs)
        q1, q3 = _iqr(evs)
        ev0  = sum(1 for v in evs if v == 0)
        ev0p = ev0 / len(evs) * 100 if evs else 0.0
        print(
            f"{MODE_LABEL[mode]:<26} {m:>7.2f} {s:>7.2f} {med:>8.1f} {q1:>6.1f} {q3:>6.1f} "
            f"{ev0:>7} {ev0p:>6.1f}%"
        )
        rows_out.append({
            "mode":    MODE_LABEL[mode],
            "mean":    round(m, 2),
            "sd":      round(s, 2),
            "median":  round(med, 1),
            "q1":      round(q1, 1),
            "q3":      round(q3, 1),
            "ev0_count": ev0,
            "ev0_pct":   round(ev0p, 1),
        })

    return rows_out


# ---------------------------------------------------------------------------
# Tabel 5 — Confidence per mode
# ---------------------------------------------------------------------------

def table_confidence(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 5 — Confidence Score per Mode (0–1)")
    print(SEP)
    print(f"{'Mode':<26} {'Mean':>7} {'SD':>7} {'Median':>8} {'Q1':>6} {'Q3':>6}")
    print(SEP2)

    rows_out: list[dict] = []
    for mode in MODES_ORDER:
        rows = data.get(mode, [])
        conf = _floats(rows, "confidence")
        m    = _mean(conf)
        s    = _std(conf)
        med  = _median(conf)
        q1, q3 = _iqr(conf)
        print(f"{MODE_LABEL[mode]:<26} {m:>7.3f} {s:>7.3f} {med:>8.3f} {q1:>6.3f} {q3:>6.3f}")
        rows_out.append({
            "mode":   MODE_LABEL[mode],
            "mean":   round(m, 3),
            "sd":     round(s, 3),
            "median": round(med, 3),
            "q1":     round(q1, 3),
            "q3":     round(q3, 3),
        })

    return rows_out


# ---------------------------------------------------------------------------
# Tabel 6 — Iterations used (mode_3, mode_4 ReAct)
# ---------------------------------------------------------------------------

def table_iterations(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 6 — ReAct Iterations per Mode (mode_3 & mode_4 saja)")
    print(SEP)
    print(f"{'Mode':<26} {'Mean':>7} {'SD':>7} {'Median':>8} {'Q1':>6} {'Q3':>6} {'Max':>6}")
    print(SEP2)

    rows_out: list[dict] = []
    react_modes = ["mode_3_rag_jc", "mode_4_rag_jc_cache"]
    for mode in react_modes:
        rows = data.get(mode, [])
        iters = _floats(rows, "iterations_used")
        if not iters:
            print(f"{MODE_LABEL[mode]:<26} {'N/A':>7}")
            rows_out.append({"mode": MODE_LABEL[mode], "mean": "N/A"})
            continue
        m    = _mean(iters)
        s    = _std(iters)
        med  = _median(iters)
        q1, q3 = _iqr(iters)
        mx   = max(iters)
        print(f"{MODE_LABEL[mode]:<26} {m:>7.2f} {s:>7.2f} {med:>8.1f} {q1:>6.1f} {q3:>6.1f} {mx:>6.0f}")
        rows_out.append({
            "mode":   MODE_LABEL[mode],
            "mean":   round(m, 2),
            "sd":     round(s, 2),
            "median": round(med, 1),
            "q1":     round(q1, 1),
            "q3":     round(q3, 1),
            "max":    int(mx),
        })

    return rows_out


# ---------------------------------------------------------------------------
# Tabel 7 — Cache status mode_4
# ---------------------------------------------------------------------------

def table_cache(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 7 — Cache Status Mode 4 (RAG+J&C+Cache)")
    print(SEP)

    rows  = data.get("mode_4_rag_jc_cache", [])
    total = len(rows)
    if total == 0:
        print("  (Tidak ada data mode_4)")
        return []

    from collections import Counter
    counter = Counter(r.get("cache_status", "unknown") for r in rows)
    rows_out: list[dict] = []
    for status in ["hit", "miss", "stale", "bypassed", "unknown"]:
        cnt = counter.get(status, 0)
        pct = cnt / total * 100 if total else 0.0
        if cnt > 0:
            print(f"  {status:<12}: {cnt:>4}  ({pct:.1f}%)")
            rows_out.append({"status": status, "count": cnt, "pct": round(pct, 2)})

    # Latency: hit vs miss
    lats_hit  = _floats([r for r in rows if r.get("cache_status") == "hit"],  "latency_ms_total")
    lats_miss = _floats([r for r in rows if r.get("cache_status") == "miss"], "latency_ms_total")
    if lats_hit and lats_miss:
        print(f"\n  Latency cache HIT  : {_desc(lats_hit)}")
        print(f"  Latency cache MISS : {_desc(lats_miss)}")
        print(f"  Selisih mean        : {_abs_diff_str(_mean(lats_miss), _mean(lats_hit), 'ms')} "
              f"({_pct_diff(_mean(lats_miss), _mean(lats_hit))} relatif vs miss)")

    return rows_out


# ---------------------------------------------------------------------------
# Tabel 8 — Hallucination rate per kategori × mode
# ---------------------------------------------------------------------------

def table_by_category(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 8 — Hallucination Rate per Kategori Pertanyaan × Mode (%)")
    print(SEP)

    all_cats: set[str] = set()
    for rows in data.values():
        for r in rows:
            if r.get("category"):
                all_cats.add(r["category"])
    cats = sorted(all_cats)

    if not cats:
        print("  (Tidak ada kolom 'category' di data)")
        return []

    mode_shorts = ["M1 LLM", "M2 RAG", "M3 RAG+JC", "M4+Cache"]
    header = f"{'Kategori':<22} " + " ".join(f"{c:>12}" for c in mode_shorts)
    print(header)
    print(SEP2)

    rows_out: list[dict] = []
    for cat in cats:
        rates: list[float] = []
        for mode in MODES_ORDER:
            subset  = [r for r in data.get(mode, []) if r.get("category") == cat]
            total   = len(subset)
            flagged = sum(1 for r in subset if _flags_in(r))
            rates.append(flagged / total * 100 if total else 0.0)

        rate_strs = " ".join(f"{r:>11.1f}%" for r in rates)
        print(f"{cat:<22} {rate_strs}")
        rows_out.append({
            "category": cat,
            **{MODES_ORDER[i]: round(rates[i], 2) for i in range(len(MODES_ORDER))},
        })

    return rows_out


# ---------------------------------------------------------------------------
# Tabel 9 — Validator status per mode
# ---------------------------------------------------------------------------

def table_validator(data: dict) -> list[dict]:
    print(f"\n{SEP}")
    print("TABEL 9 — Validator Status per Mode")
    print(SEP)
    print(f"{'Mode':<26} {'passed':>8} {'failed':>8} {'skipped':>9} {'pass%':>7}")
    print(SEP2)

    rows_out: list[dict] = []
    for mode in MODES_ORDER:
        rows    = data.get(mode, [])
        total   = len(rows)
        passed  = sum(1 for r in rows if r.get("validator_status") == "passed")
        failed  = sum(1 for r in rows if r.get("validator_status") == "failed")
        skipped = sum(1 for r in rows if r.get("validator_status") == "skipped")
        passp   = passed / total * 100 if total else 0.0
        print(f"{MODE_LABEL[mode]:<26} {passed:>8} {failed:>8} {skipped:>9} {passp:>6.1f}%")
        rows_out.append({
            "mode":    MODE_LABEL[mode],
            "passed":  passed,
            "failed":  failed,
            "skipped": skipped,
            "pass_pct": round(passp, 1),
        })

    return rows_out


# ---------------------------------------------------------------------------
# Tabel 10 — Paired analysis: per question_id, jumlah mode yang flag
# ---------------------------------------------------------------------------

def table_paired(data: dict) -> list[dict]:
    """Per question_id: berapa mode yang flag halusinasi."""
    print(f"\n{SEP}")
    print("TABEL 10 — Paired Analysis: Distribusi Q berdasarkan jumlah mode yang flag")
    print(SEP)

    # Kumpulkan per question_id
    q_mode_flag: dict[str, set[str]] = defaultdict(set)
    for mode, rows in data.items():
        if mode not in MODES_ORDER:
            continue
        for r in rows:
            qid = r.get("question_id", "")
            if _flags_in(r):
                q_mode_flag[qid].add(mode)

    # Hitung semua question_id
    all_qids: set[str] = set()
    for rows in data.values():
        for r in rows:
            if r.get("question_id"):
                all_qids.add(r["question_id"])

    dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for qid in all_qids:
        n = len(q_mode_flag.get(qid, set()))
        dist[n] = dist.get(n, 0) + 1

    total_q = len(all_qids)
    print(f"{'Mode yang flag':>20} {'Jumlah Q':>10} {'Pct':>8}")
    print(SEP2)
    rows_out: list[dict] = []
    for k in sorted(dist.keys()):
        cnt = dist[k]
        pct = cnt / total_q * 100 if total_q else 0.0
        label = "Tidak ada mode" if k == 0 else f"{k} mode"
        print(f"{label:>20} {cnt:>10} {pct:>7.1f}%")
        rows_out.append({"jumlah_mode_flag": k, "jumlah_q": cnt, "pct": round(pct, 1)})

    # Q yang flagged di semua 4 mode (persistent hallucination)
    persistent = [qid for qid in all_qids if len(q_mode_flag.get(qid, set())) == 4]
    if persistent:
        print(f"\n  Q flagged semua 4 mode ({len(persistent)}): {', '.join(sorted(persistent))}")

    # Q tidak flagged di mode mana pun (clean)
    clean = [qid for qid in all_qids if len(q_mode_flag.get(qid, set())) == 0]
    print(f"  Q bersih tanpa flag ({len(clean)}): {', '.join(sorted(clean)[:10])}{'...' if len(clean) > 10 else ''}")

    return rows_out


# ---------------------------------------------------------------------------
# Perbandingan Gemini vs Groq (opsional)
# ---------------------------------------------------------------------------

def table_gemini_comparison(groq_data: dict, gemini_path: Path) -> None:
    """Bandingkan mode_3 & mode_4 Groq Llama vs Gemini/Mistral."""
    if not gemini_path.exists():
        print(f"\n  (Skip perbandingan Gemini — {gemini_path.name} belum tersedia)")
        return

    gemini_data = _load(gemini_path)

    print(f"\n{SEP}")
    print(f"TABEL BONUS — Perbandingan Groq Llama vs Gemini/Mistral (mode_3 & mode_4)")
    print(f"  Sumber Gemini: {gemini_path.name}")
    print(SEP)

    compare_modes = ["mode_3_rag_jc", "mode_4_rag_jc_cache"]
    metric_groups = [
        ("Hallucination Rate (%)", lambda rows: (
            sum(1 for r in rows if _flags_in(r)) / len(rows) * 100 if rows else 0.0
        )),
        ("Latency Mean (ms)", lambda rows: _mean(_floats(rows, "latency_ms_total"))),
        ("Evidence Count Mean", lambda rows: _mean(_floats(rows, "evidence_count"))),
        ("Confidence Mean", lambda rows: _mean(_floats(rows, "confidence"))),
    ]

    for metric_name, metric_fn in metric_groups:
        print(f"\n  {metric_name}")
        print(f"  {'Mode':<26} {'Groq (Llama)':>14} {'Gemini/Mistral':>16} {'Selisih':>10}")
        print(f"  {'-'*68}")
        for mode in compare_modes:
            groq_rows   = groq_data.get(mode, [])
            gemini_rows = gemini_data.get(mode, [])
            v_groq      = metric_fn(groq_rows)
            v_gemini    = metric_fn(gemini_rows)
            diff        = _abs_diff_str(v_groq, v_gemini)
            print(
                f"  {MODE_LABEL[mode]:<26} {v_groq:>13.2f}  {v_gemini:>14.2f}  {diff:>10}"
            )

    # Distribusi H1–H4 Gemini vs Groq
    print(f"\n  Distribusi H1–H4 Gemini")
    print(f"  {'Mode':<26} {'H1':>5} {'H2':>5} {'H3':>5} {'H4':>5}")
    print(f"  {'-'*50}")
    for mode in compare_modes:
        rows   = gemini_data.get(mode, [])
        counts = {h: 0 for h in H_CODES}
        for r in rows:
            for flag in _flags_in(r):
                if flag in counts:
                    counts[flag] += 1
        print(
            f"  {MODE_LABEL[mode]:<26} {counts['H1']:>5} {counts['H2']:>5} "
            f"{counts['H3']:>5} {counts['H4']:>5}"
        )


# ---------------------------------------------------------------------------
# Ringkasan naratif
# ---------------------------------------------------------------------------

def print_narasi(data: dict) -> None:
    print(f"\n{SEP}")
    print("RINGKASAN NARATIF (untuk Bab IV — pelaporan deskriptif)")
    print(SEP)

    rates: dict[str, float] = {}
    for mode in MODES_ORDER:
        rows    = data.get(mode, [])
        total   = len(rows)
        if total == 0:
            continue
        flagged = sum(1 for r in rows if _flags_in(r))
        rate    = flagged / total * 100
        rates[mode] = rate
        lats = _floats(rows, "latency_ms_total")
        print(
            f"  {MODE_LABEL[mode]}: hallucination rate {rate:.1f}% "
            f"({flagged}/{total}), latency {_mean(lats):.0f} ± {_std(lats):.0f} ms"
        )

    # Selisih antar mode
    r1 = rates.get("mode_1_llm_only", 0.0)
    for mode in ["mode_2_rag_only", "mode_3_rag_jc", "mode_4_rag_jc_cache"]:
        rx = rates.get(mode, 0.0)
        if r1 > 0:
            label = MODE_LABEL[mode]
            print(
                f"\n  {label} vs Mode 1: "
                f"{abs(rx - r1):.1f} poin absolut lebih {'rendah' if rx < r1 else 'tinggi'} "
                f"({_pct_diff(r1, rx)} relatif)"
            )

    # Cache
    m4_rows = data.get("mode_4_rag_jc_cache", [])
    if m4_rows:
        hits = sum(1 for r in m4_rows if r.get("cache_status") == "hit")
        pct  = hits / len(m4_rows) * 100
        print(f"\n  Cache hit rate Mode 4: {hits}/{len(m4_rows)} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Simpan CSV
# ---------------------------------------------------------------------------

def _save_csv(filename: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = OUTPUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analisis deskriptif hasil eksperimen.")
    parser.add_argument(
        "--file",
        default="experiment_results_full_v2.csv",
        help="Nama file CSV di app/data/ (default: experiment_results_full_v2.csv)",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Sertakan perbandingan dengan experiment_results_gemini.csv",
    )
    args = parser.parse_args()

    input_path  = DATA_DIR / args.file
    gemini_path = DATA_DIR / "experiment_results_gemini.csv"

    if not input_path.exists():
        print(f"ERROR: {input_path} tidak ditemukan. Jalankan eksperimen dulu.")
        return

    data  = _load(input_path)
    total = sum(len(v) for v in data.values())
    modes_available = [m for m in MODES_ORDER if m in data]

    print(f"\n{SEP}")
    print(f"ANALISIS DESKRIPTIF — {input_path.name}")
    print(f"Total baris: {total}  |  Mode tersedia: {len(modes_available)}/4  |  "
          f"Q per mode: {max((len(v) for v in data.values()), default=0)}")
    print(SEP)

    t1  = table_hallucination_rate(data)
    t2  = table_flag_distribution(data)
    t3  = table_latency(data)
    t4  = table_evidence(data)
    t5  = table_confidence(data)
    t6  = table_iterations(data)
    t7  = table_cache(data)
    t8  = table_by_category(data)
    t9  = table_validator(data)
    t10 = table_paired(data)
    print_narasi(data)

    if args.gemini:
        table_gemini_comparison(data, gemini_path)

    print(f"\n{SEP}")
    print("MENYIMPAN CSV OUTPUT ...")
    prefix = input_path.stem  # e.g. "experiment_results_full_v2"
    _save_csv(f"analysis_{prefix}_1_halluc_rate.csv",    t1)
    _save_csv(f"analysis_{prefix}_2_flag_dist.csv",      t2)
    _save_csv(f"analysis_{prefix}_3_latency.csv",        t3)
    _save_csv(f"analysis_{prefix}_4_evidence.csv",       t4)
    _save_csv(f"analysis_{prefix}_5_confidence.csv",     t5)
    _save_csv(f"analysis_{prefix}_6_iterations.csv",     t6)
    _save_csv(f"analysis_{prefix}_7_cache.csv",          t7)
    _save_csv(f"analysis_{prefix}_8_by_category.csv",    t8)
    _save_csv(f"analysis_{prefix}_9_validator.csv",      t9)
    _save_csv(f"analysis_{prefix}_10_paired.csv",        t10)
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
