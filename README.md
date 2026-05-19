FINAL PROJECTS 2026
IMPLEMENTATION OF AGENTIC AI USING THE SEMANTIC
CACHING METHOD TO ADDRESS THE HALLUCINATION PROBLEM
IN AN INDONESIAN STOCK MARKET DATA CHATBOT

# IDX30 Multi-Agent Stock Chatbot Backend

Backend chatbot pasar saham Indonesia untuk eksperimen skripsi dengan 3 mode:

1. `mode_1_baseline_llm`
   - LLM only
   - tanpa retrieval
   - tanpa validator
   - tanpa semantic cache

2. `mode_2_rag_only`
   - retrieval dari knowledge base
   - tanpa validator
   - tanpa semantic cache

3. `mode_3_full_agentic`
   - query normalization
   - semantic cache
   - retrieval/tool use
   - answer composer
   - validator
   - observability

## Arsitektur Sistem

Komponen utama:
- `query_normalizer`: normalisasi sinonim emiten, typo ringan, dan variasi pertanyaan
- `cache_service`: semantic cache berbasis ChromaDB
- `retrieval_service`: retrieval dari vector knowledge base
- `orchestrator_service`: pengendali utama mode eksperimen
- `validator_service`: validasi evidence, angka, ticker, periode, dan unsupported claim
- `telemetry_service`: logging Langfuse
- `chat_api`: endpoint FastAPI

## Alur Tiap Mode

### Mode 1
`question -> LLM -> answer`

### Mode 2
`question -> retrieval -> LLM with context -> answer`

### Mode 3
`question -> normalization -> semantic cache lookup -> retrieval/tool use -> answer composer -> validator -> fallback/cache store -> answer`

## Struktur Folder

```text
app/
  chat_api.py
  schemas.py
  services/
  modes/
tests/
main.py
config.py