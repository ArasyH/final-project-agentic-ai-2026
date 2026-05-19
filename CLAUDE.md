# Project: final-project-agentic-ai-2026

Agentic AI untuk pasar saham IDX30 dengan pola **Judge & Critic** + **RAG** + **Semantic Cache**.
Target publikasi: **jurnal nasional SINTA 2** (bukan Q3 internasional).
Tugas Akhir S1 Informatika ITENAS Bandung. Sidang Final: Juni 2026.

> **Catatan untuk asisten coding**: file ini adalah ringkasan. Untuk aturan lengkap, sesi WAJIB dijalankan dengan `--append-system-prompt` yang me-load `SYSTEM_PROMPT_agent_specialist_app_folder.md`. Jika sesi dimulai tanpa append tersebut, **STOP** dan beri tahu peneliti.

---

## Scope Kerja (HARD)

**Hanya** sentuh `app/`. Dilarang menyentuh: `etl/`, `chroma_db/`, `tests/` di luar `app/`, `requirements.txt`, `Dockerfile`, `docker_compose.yml`, `.env`, `README.md`, notebook `*.ipynb`, atau file lain di luar `app/`.

Jika perubahan di luar `app/` dibutuhkan → **STOP dan tanya peneliti**.

## Arsitektur Frozen (Tidak Diubah)

| Komponen | Nilai Frozen |
|---|---|
| Generator | Llama-3.1-8B-Instant (Groq) + ReAct loop |
| Critic | Llama-3.3-70B-Versatile (Groq), output JSON-structured |
| Embedder | `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector store | ChromaDB (1 KB collection + 1 cache collection) |
| Similarity threshold cache | 0.85 (cosine) |
| Cache TTL | 8 jam |
| Observability | **Langfuse `3.4.16`** (pin; v2 stabil, hindari v4.x) |
| KB source | Sectors.app API (IDX30) |
| ETL schedule | Senin–Jumat 16:00 WIB |
| Temperature | 0.0 (Generator & Critic, semua call) |

## 4 Mode Eksperimen

| Mode | Komponen |
|---|---|
| `mode_1_llm_only` | LLM saja, tanpa retrieval / cache / critic / guardrails |
| `mode_2_rag_only` | Retrieval (k=3) + LLM |
| `mode_3_rag_jc` | Retrieval + Generator ReAct + Domain Guardrails + Critic |
| `mode_4_rag_jc_cache` | Mode 3 + Semantic Cache (lookup → store-after-critic-passes) |

Signature wajib identik: `def run_mode_x(question: str, session_id: str) -> InternalResponse`.
Dataset evaluasi: **50 pertanyaan paired**, urutan deterministik, dijalankan di keempat mode.

## 4 Kategori Halusinasi (telemetry tag `hallucination_flags`)

- `H1` Unsupported Numeric Claim → Domain Guardrails (deterministik)
- `H2` Fabricated Financial Metric → Critic Agent (LLM)
- `H3` Stale Timestamp Misrepresentation → Domain Guardrails
- `H4` Incorrect Inference → Critic Agent

## Behavioral Rules

1. **Always Ask Before Deciding** — keputusan statistik, metodologi, threshold tuning, dependency baru, perubahan rubric Critic → STOP, sajikan 2–3 opsi + trade-off, tunggu jawaban. **JANGAN default**.
2. **YAGNI** — tidak ada middleware, retry logic, caching tambahan, refactor di luar scope tanpa diminta.
3. **Hallucination-first** — setiap baris kode harus bisa dipetakan ke H1/H2/H3/H4 atau ke klaim publikasi SINTA 2 (reproducibility / domain Indonesia / defensible methodology).
4. **Reproducibility** — temperature=0.0, semua hyperparameter dari `app/config.py`, tidak ada `random` tanpa seed, prompt versioned (`_v2`, `_v3` — jangan replace in-place).
5. **Pelaporan deskriptif saja** — DILARANG memakai istilah "significant", "p-value", "reject null hypothesis" di code/log/telemetry. Pakai "lebih rendah X%", "reduksi N poin".

## Output Format Critic Agent (FROZEN)

JSON dengan struktur: `H1_unsupported_numeric`, `H2_fabricated_metric`, `H3_stale_timestamp`, `H4_incorrect_inference` (masing-masing `{flag: bool, rationale: str}`) + `overall_verdict: "pass"|"fail"` + `model` + `temperature`.

Parsing: Pydantic `BaseModel` atau `JsonOutputParser`. **JANGAN regex**.

## Definition of Done (per perubahan kode)

- [ ] Type hints lengkap di public functions
- [ ] Docstring (Google atau NumPy style, konsisten)
- [ ] `temperature=0.0` di semua LLM call
- [ ] Hyperparameter dari `app/config.py`
- [ ] Langfuse trace dengan metadata: `mode`, `question_id`, `cache_status`, `validator_status`, `latency_ms_*`, `hallucination_flags`, `evidence_count`, `confidence`
- [ ] Tidak ada perubahan di luar `app/`
- [ ] Pemetaan eksplisit ke H1–H4 atau ke aspek publikasi SINTA 2

## Bahasa & Komunikasi

Bahasa Indonesia profesional. Istilah teknis boleh Inggris. **Tidak boleh** emoji berlebihan, motivational phrasing, "great question". Tidak ada `print()` debug yang tertinggal. Tidak ada TODO/FIXME tanpa referensi issue tracker.
