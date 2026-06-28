"""
Utility: generate draft reference answers untuk 50 pertanyaan evaluasi.

Menggunakan RetrievalService yang sama dengan eksperimen (paraphrase-multilingual-MiniLM-L12-v2)
sehingga konteks yang diambil identik dengan yang diterima Mode 2/3/4 saat eksperimen.

Usage:
    source venv/bin/activate
    python -m app.create_ground_truth

Output:
    app/data/ground_truth_drafts.json

Langkah setelah script selesai:
1. Buka ground_truth_drafts.json
2. Review setiap "draft_ground_truth" — koreksi jika ada yang salah
3. Set "approved": true untuk setiap entry yang sudah benar
4. Jalankan: python -m app.apply_ground_truth
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_GENERATOR_MODEL
from app.services.retrieval_service import RetrievalService
from app.services.query_normalizer import normalize_query

DATASET_PATH: Path  = Path("app/data/evaluation_dataset.json")
DRAFTS_PATH: Path   = Path("app/data/ground_truth_drafts.json")
SLEEP_BETWEEN: float = 3.0  # detik antara Groq call — hindari rate limit

_SYSTEM_PROMPT = (
    "Anda adalah asisten yang menyusun reference answer untuk evaluasi sistem QA pasar saham IDX30. "
    "Tulis jawaban referensi yang akurat dan singkat (1–3 kalimat) berdasarkan HANYA dokumen yang diberikan. "
    "Wajib sertakan angka spesifik dan snapshot_date dari dokumen sebagai sumber. "
    "Jika dokumen tidak cukup untuk menjawab pertanyaan, tulis persis: "
    "DATA_TIDAK_TERSEDIA: [alasan singkat mengapa data tidak ada di KB]. "
    "Jawab dalam Bahasa Indonesia."
)


def _format_contexts(docs: list) -> str:
    """Format dokumen retrieval menjadi teks terstruktur untuk LLM prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts.append(
            f"[Dokumen {i}]\n"
            f"Symbol    : {meta.get('symbol', '-')}\n"
            f"Tanggal   : {meta.get('snapshot_date', '-')}\n"
            f"Kategori  : {meta.get('category', '-')}\n"
            f"Konten    : {doc.page_content}"
        )
    return "\n\n".join(parts)


def _generate_draft(question: str, contexts_text: str, client: Groq) -> str:
    """Generate draft reference answer via Groq LLM dari konteks retrieval."""
    user_msg = f"Dokumen dari basis pengetahuan:\n{contexts_text}\n\nPertanyaan: {question}"
    response = client.chat.completions.create(
        model=GROQ_GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    """Jalankan proses generate draft ground truth untuk semua pertanyaan."""
    dataset: list[dict] = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    # Resume: skip question_id yang sudah punya draft
    if DRAFTS_PATH.exists():
        drafts: list[dict] = json.loads(DRAFTS_PATH.read_text(encoding="utf-8"))
        done_ids: set[str] = {
            d["question_id"] for d in drafts if d.get("draft_ground_truth")
        }
    else:
        drafts = []
        done_ids = set()

    retrieval = RetrievalService()
    client    = Groq(api_key=GROQ_API_KEY)
    total     = len(dataset)

    for idx, item in enumerate(dataset):
        qid      = item["question_id"]
        question = item["question"]
        tickers  = item.get("expected_tickers", [])
        coverage = item.get("expected_kb_coverage", "full")

        if qid in done_ids:
            print(f"[skip] {qid} sudah ada draft")
            continue

        print(f"[{idx+1:02d}/{total}] {qid} — {question[:60]}...")

        # Retrieval dengan embedding model yang sama seperti eksperimen
        try:
            nq   = normalize_query(question)
            docs = retrieval.retrieve(
                query=nq.normalized_query,
                tickers=nq.detected_tickers if nq.detected_tickers else tickers,
            )
        except Exception as exc:
            print(f"  [WARN] retrieval gagal: {exc}")
            docs = []

        # Generate draft
        if not docs:
            draft      = "DATA_TIDAK_TERSEDIA: tidak ada dokumen yang ditemukan di KB untuk pertanyaan ini."
            model_used = "none"
        else:
            contexts_text = _format_contexts(docs)
            try:
                draft      = _generate_draft(question, contexts_text, client)
                model_used = GROQ_GENERATOR_MODEL
            except Exception as exc:
                print(f"  [ERROR] LLM gagal: {exc}")
                draft      = f"ERROR: {exc}"
                model_used = "error"

        entry: dict = {
            "question_id":           qid,
            "question":              question,
            "category":              item.get("category"),
            "expected_tickers":      tickers,
            "expected_kb_coverage":  coverage,
            "trigger_hallucination": item.get("trigger_hallucination", []),
            "retrieved_docs": [
                {
                    "content":  d.page_content,
                    "metadata": d.metadata,
                }
                for d in docs
            ],
            "draft_ground_truth": draft,
            "model_used":         model_used,
            "approved":           False,   # Set True setelah peneliti review
        }
        drafts.append(entry)
        DRAFTS_PATH.write_text(
            json.dumps(drafts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  → draft disimpan | docs: {len(docs)} | coverage: {coverage}")

        time.sleep(SLEEP_BETWEEN)

    n_approved = sum(1 for d in drafts if d.get("approved"))
    print(f"\nSelesai. {len(drafts)}/{total} draft tersimpan di {DRAFTS_PATH}")
    print(f"Sudah approved: {n_approved} | Perlu review: {len(drafts) - n_approved}")
    print("Setelah review, jalankan: python -m app.apply_ground_truth")


if __name__ == "__main__":
    main()
