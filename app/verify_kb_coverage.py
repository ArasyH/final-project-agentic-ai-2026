"""
Utility: verifikasi coverage KB untuk 50 pertanyaan evaluasi.

Membaca ground_truth_drafts.json (hasil create_ground_truth.py) dan menghasilkan
laporan CSV yang mudah dibaca di Excel/Numbers untuk review ground truth.

Usage:
    source venv/bin/activate
    python -m app.verify_kb_coverage

Output:
    app/data/kb_coverage_report.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

DRAFTS_PATH: Path = Path("app/data/ground_truth_drafts.json")
REPORT_PATH: Path = Path("app/data/kb_coverage_report.csv")


def _truncate(text: str, max_len: int = 1000) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."


def main() -> None:
    """Generate laporan CSV verifikasi KB coverage untuk semua 50 pertanyaan."""
    if not DRAFTS_PATH.exists():
        print(f"File tidak ditemukan: {DRAFTS_PATH}")
        print("Jalankan dulu: python -m app.create_ground_truth")
        return

    drafts: list[dict] = json.loads(DRAFTS_PATH.read_text(encoding="utf-8"))

    fieldnames = [
        "question_id",
        "category",
        "expected_kb_coverage",
        "trigger_hallucination",
        "question",
        "n_docs_retrieved",
        "doc1_symbol",
        "doc1_snapshot_date",
        "doc1_category",
        "doc1_content",
        "doc2_symbol",
        "doc2_snapshot_date",
        "doc2_content",
        "doc3_symbol",
        "doc3_snapshot_date",
        "doc3_content",
        "draft_ground_truth",
        "approved",
    ]

    rows: list[dict] = []
    for d in drafts:
        docs = d.get("retrieved_docs", [])

        def _doc_field(idx: int, field: str) -> str:
            if idx >= len(docs):
                return ""
            doc = docs[idx]
            if field == "content":
                return _truncate(doc.get("content", ""))
            return str(doc.get("metadata", {}).get(field, ""))

        rows.append({
            "question_id":           d.get("question_id", ""),
            "category":              d.get("category", ""),
            "expected_kb_coverage":  d.get("expected_kb_coverage", ""),
            "trigger_hallucination": "|".join(d.get("trigger_hallucination", [])),
            "question":              d.get("question", ""),
            "n_docs_retrieved":      len(docs),
            "doc1_symbol":           _doc_field(0, "symbol"),
            "doc1_snapshot_date":    _doc_field(0, "snapshot_date"),
            "doc1_category":         _doc_field(0, "category"),
            "doc1_content":          _doc_field(0, "content"),
            "doc2_symbol":           _doc_field(1, "symbol"),
            "doc2_snapshot_date":    _doc_field(1, "snapshot_date"),
            "doc2_content":          _doc_field(1, "content"),
            "doc3_symbol":           _doc_field(2, "symbol"),
            "doc3_snapshot_date":    _doc_field(2, "snapshot_date"),
            "doc3_content":          _doc_field(2, "content"),
            "draft_ground_truth":    d.get("draft_ground_truth", ""),
            "approved":              d.get("approved", False),
        })

    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Ringkasan ke terminal
    n_full  = sum(1 for r in rows if r["expected_kb_coverage"] == "full")
    n_none  = sum(1 for r in rows if r["expected_kb_coverage"] == "none")
    n_part  = sum(1 for r in rows if r["expected_kb_coverage"] == "partial")
    n_zero  = sum(1 for r in rows if r["n_docs_retrieved"] == 0)
    n_data_na = sum(1 for r in rows if "DATA_TIDAK_TERSEDIA" in r["draft_ground_truth"])
    n_approved = sum(1 for r in rows if r["approved"] is True)

    print(f"Laporan disimpan: {REPORT_PATH}")
    print()
    print("=== Ringkasan Coverage ===")
    print(f"  expected_kb_coverage=full    : {n_full:2d} pertanyaan")
    print(f"  expected_kb_coverage=partial : {n_part:2d} pertanyaan")
    print(f"  expected_kb_coverage=none    : {n_none:2d} pertanyaan")
    print()
    print("=== Status Retrieval ===")
    print(f"  0 docs retrieved (KB miss)   : {n_zero:2d} pertanyaan")
    print(f"  Draft = DATA_TIDAK_TERSEDIA  : {n_data_na:2d} pertanyaan")
    print(f"  Sudah approved               : {n_approved:2d} / 50")
    print()

    # Tampilkan per-baris ringkas ke terminal
    print(f"{'QID':<7} {'CAT':<22} {'COV':<8} {'DOCS':<5} {'SNAPSHOT_DATE':<14} {'DRAFT (60 char)'}")
    print("-" * 120)
    for r in rows:
        draft_short = r["draft_ground_truth"][:60].replace("\n", " ")
        print(
            f"{r['question_id']:<7} "
            f"{r['category']:<22} "
            f"{r['expected_kb_coverage']:<8} "
            f"{r['n_docs_retrieved']:<5} "
            f"{r['doc1_snapshot_date']:<14} "
            f"{draft_short}"
        )


if __name__ == "__main__":
    main()
