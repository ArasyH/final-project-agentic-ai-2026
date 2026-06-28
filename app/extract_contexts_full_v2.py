"""app/extract_contexts_full_v2.py — Re-extract retrieval contexts untuk full_v2.csv.

RAGAS context_precision dan context_recall membutuhkan kolom `contexts`
(list of chunk text) per baris. CSV eksperimen utama hanya menyimpan
`evidence_count`, tidak chunk-nya.

Skrip ini menjalankan retrieval-only (RetrievalService.retrieve) pada
setiap baris full_v2.csv:
  - Mode 1 (LLM only): contexts = [] (mode ini tidak retrieve)
  - Mode 2/3/4: contexts = chunk text top-k=3 dari Chroma kb_collection

Karena ChromaDB cosine similarity deterministik untuk embedding identik,
chunk yang dikembalikan harus sama dengan run asli (kecuali KB telah
di-update; pada eksperimen ini KB statis sejak 2026-05-28 freeze).

Output: app/data/full_v2_with_contexts.csv (200 baris + kolom `contexts`
   sebagai JSON string list).

Jalankan:
    source venv/bin/activate
    python3 -m app.extract_contexts_full_v2
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from app.services.retrieval_service import RetrievalService

INPUT_PATH = Path(__file__).parent / "data" / "experiment_results_full_v2.csv"
OUTPUT_PATH = Path(__file__).parent / "data" / "full_v2_with_contexts.csv"

MODES_WITH_RETRIEVAL = {
    "mode_2_rag_only",
    "mode_3_rag_jc",
    "mode_4_rag_jc_cache",
}


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: input tidak ditemukan: {INPUT_PATH}", file=sys.stderr)
        return 1

    retriever = RetrievalService()

    with INPUT_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys()) + ["contexts"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== EXTRACT CONTEXTS MULAI ===", flush=True)
    print(f"Input  : {INPUT_PATH}", flush=True)
    print(f"Output : {OUTPUT_PATH}", flush=True)
    print(f"Rows   : {len(rows)}", flush=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows, start=1):
            mode = row.get("mode", "")
            qid = row.get("question_id", "")
            question = row.get("question", "")

            if mode in MODES_WITH_RETRIEVAL and question:
                docs = retriever.retrieve(question)
                contexts = [doc.page_content for doc in docs]
            else:
                contexts = []

            row["contexts"] = json.dumps(contexts, ensure_ascii=False)
            writer.writerow(row)

            if i % 25 == 0 or i == len(rows):
                print(f"  [{i:>3}/{len(rows)}] {qid} {mode} | contexts={len(contexts)}",
                      flush=True)

    print(f"=== SELESAI ===", flush=True)
    print(f"CSV : {OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
