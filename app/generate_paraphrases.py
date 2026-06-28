"""app/generate_paraphrases.py — Generate parafrase Q seed untuk replay cache.

Untuk eksperimen mini-replay (Tujuan 4: cache hit ratio realistis), 10 pertanyaan
seed (terpilih dari V6 yang Critic-passed atau berekspektasi pass tinggi)
diparafrase 2× per pertanyaan via Llama-3.3-70B-Versatile (T=0.0).

Output: app/data/cache_replay_dataset.json
  Struktur: list of {question_id, question, category, origin_qid, variant}
  Total 30 entry = 10 original + 20 parafrase.

Jalankan:
    source venv/bin/activate
    python3 -m app.generate_paraphrases
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from app.services.llm_service import build_critic_llm

OUTPUT_PATH = Path(__file__).parent / "data" / "cache_replay_dataset.json"

# Seed 10 Q dari V6 (lihat memory + konfirmasi peneliti 2026-06-23)
SEED_QUESTIONS: list[dict[str, str]] = [
    {"question_id": "Q001", "category": "price_snapshot",
     "question": "Berapa harga penutupan terakhir saham BBCA?"},
    {"question_id": "Q002", "category": "price_snapshot",
     "question": "Berapa harga penutupan saham BBRI pada 30 April 2026?"},
    {"question_id": "Q006", "category": "price_snapshot",
     "question": "Berapa persen perubahan harga saham BMRI dalam 30 hari terakhir?"},
    {"question_id": "Q008", "category": "price_snapshot",
     "question": "Berapa kapitalisasi pasar terkini saham UNVR?"},
    {"question_id": "Q012", "category": "fundamental_metric",
     "question": "Berapa rasio P/B (Price-to-Book) saham UNTR berdasarkan kuartal terakhir?"},
    {"question_id": "Q020", "category": "fundamental_metric",
     "question": "Berapa Forward P/E (Forward Price-to-Earnings) AMRT?"},
    {"question_id": "Q022", "category": "fundamental_metric",
     "question": "Berapa total dividen yang dibayarkan BBCA selama tahun 2024?"},
    {"question_id": "Q025", "category": "ranking",
     "question": "Apa 3 saham IDX30 dengan kapitalisasi pasar terbesar saat ini?"},
    {"question_id": "Q044", "category": "sector_query",
     "question": ("Saham Food & Beverage mana di IDX30 yang memiliki "
                  "dividend yield TTM (Trailing Twelve Months) tertinggi?")},
    {"question_id": "Q046", "category": "sector_query",
     "question": ("Berapa harga penutupan terakhir PGEO sebagai satu-satunya "
                  "saham Utilities di IDX30?")},
]

PARAPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Anda adalah asisten parafrase pertanyaan finansial dalam bahasa Indonesia.\n"
     "Tugas: parafrase satu pertanyaan menjadi DUA variasi yang berbeda secara leksikal "
     "tetapi IDENTIK secara semantik. Aturan:\n"
     "1. Pertahankan SEMUA entitas spesifik (kode saham, tanggal, metrik, periode).\n"
     "2. Ubah struktur kalimat, sinonim, urutan klausa.\n"
     "3. Jangan menambah/mengurangi makna; jangan menyederhanakan istilah teknis.\n"
     "4. Output WAJIB JSON valid: {{\"variant_1\": \"...\", \"variant_2\": \"...\"}}\n"
     "5. Jangan tambahkan teks lain di luar JSON."),
    ("human", "Pertanyaan original: {question}"),
])


def _parse_json(raw: str) -> dict[str, str]:
    """Parse JSON dari output LLM, toleran terhadap whitespace/code-fence."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def main() -> int:
    llm = build_critic_llm(temperature=0.0)
    chain = PARAPHRASE_PROMPT | llm

    dataset: list[dict[str, str]] = []

    for seed in SEED_QUESTIONS:
        qid = seed["question_id"]
        print(f"[paraphrase] {qid} ... ", end="", flush=True)

        dataset.append({
            "question_id": qid,
            "question": seed["question"],
            "category": seed["category"],
            "origin_qid": qid,
            "variant": "original",
        })

        try:
            resp = chain.invoke({"question": seed["question"]})
            parsed = _parse_json(resp.content)
            v1 = parsed["variant_1"].strip()
            v2 = parsed["variant_2"].strip()
        except Exception as exc:
            print(f"FAIL ({exc})")
            return 1

        dataset.append({
            "question_id": f"{qid}_p1",
            "question": v1,
            "category": seed["category"],
            "origin_qid": qid,
            "variant": "paraphrase_1",
        })
        dataset.append({
            "question_id": f"{qid}_p2",
            "question": v2,
            "category": seed["category"],
            "origin_qid": qid,
            "variant": "paraphrase_2",
        })
        print("OK")
        time.sleep(2.0)  # Hindari Groq rate-limit

    OUTPUT_PATH.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[paraphrase] {len(dataset)} entry → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
