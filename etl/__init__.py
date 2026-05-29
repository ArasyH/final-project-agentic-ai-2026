"""
Paket ETL untuk knowledge base IDX30.

Modul ini membangun knowledge base ChromaDB yang digunakan oleh
seluruh mode eksperimen (mode_2 sampai mode_4) dalam penelitian
mitigasi halusinasi pada agentic AI analisis fundamental saham IDX30.

Urutan pipeline:
    1. extract   — tarik data dari Sectors.app API (v2)
    2. transform — ubah raw JSON → dokumen .txt per-topik (<128 token)
    3. aggregate — bangun dokumen lintas-saham (ranking, sektor)
    4. load      — upsert ke ChromaDB dengan metadata lengkap
    5. run       — entry point CLI yang mengorkestrasi 1–4
"""
