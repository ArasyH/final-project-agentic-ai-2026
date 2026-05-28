"""
Konfigurasi terpusat untuk pipeline ETL IDX30.

File ini adalah single source of truth untuk semua konstanta yang
dipakai modul downstream: extract, transform, aggregate, load, dan run.
Semua hyperparameter eksperimen yang terkait ETL WAJIB didefinisikan
di sini — tidak boleh hardcode di modul lain.

Referensi locked decisions (system prompt §16):
    #2  Price history time series DROP — hanya snapshot 30 hari ringkas
    #3  Kategori dataset: price_snapshot, fundamental_metric, ranking, sector_query
    #4  ETL mode: Idempotent Full Refresh
    #6  MAX_DOC_TOKENS = 128 (HARD — embedder sequence length)
    #10 ANNUAL_HISTORY_YEARS = 5
    #11 QUARTERLY_HISTORY_QUARTERS = 6
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Direktori ─────────────────────────────────────────────────────────────────

ETL_DIR: Path = Path(__file__).parent
"""Root folder paket etl/."""

RAW_DATA_DIR: Path = ETL_DIR / "raw_data"
"""Folder output extract — file JSON mentah per saham per tanggal."""

PROCESSED_DATA_DIR: Path = ETL_DIR / "processed_data"
"""Folder output transform — file .txt per-topik per saham."""

LOGS_DIR: Path = ETL_DIR / "logs"
"""Folder log ETL — api_calls.csv dan etl_run_{snapshot_date}.json."""

# CHROMA_DB_PATH dibaca dari env agar konsisten dengan app/config.py
# yang memakai variabel lingkungan yang sama.
# Fallback: etl/processed_data/chroma_db (default lokal yang masuk akal).
_chroma_db_env: str | None = os.getenv("CHROMA_DB_PATH")
CHROMA_DB_PATH: Path = (
    Path(_chroma_db_env)
    if _chroma_db_env
    else PROCESSED_DATA_DIR / "chroma_db"
)
"""Path ke PersistentClient ChromaDB — dibaca dari env CHROMA_DB_PATH."""

# ── ChromaDB ──────────────────────────────────────────────────────────────────

CHROMA_COLLECTION_NAME: str = os.getenv(
    "KNOWLEDGE_BASE_COLLECTION", "stock_knowledge_base"
)
"""Nama koleksi knowledge base di ChromaDB.

Harus identik dengan KNOWLEDGE_BASE_COLLECTION di app/config.py.
"""

EMBEDDER_MODEL: str = os.getenv(
    "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"
)
"""Model sentence-transformers untuk embedding dokumen.

Frozen per §3 arsitektur: paraphrase-multilingual-MiniLM-L12-v2.
Max sequence length model ini = 128 token — sehingga MAX_DOC_TOKENS = 128.
"""

# ── Token & Document Budget ───────────────────────────────────────────────────

MAX_DOC_TOKENS: int = 128
"""Batas token KERAS per dokumen output.

Sesuai sequence length maksimum embedder. Dokumen yang melebihi batas ini
akan ter-truncate secara diam-diam oleh model, sehingga bagian akhir
dokumen tidak ter-embed dan tidak dapat di-retrieve.
Referensi: §16 keputusan #6.
"""

ANNUAL_HISTORY_YEARS: int = 5
"""Jumlah tahun data keuangan tahunan yang disimpan ke KB.

Diambil dari historical_financials pada Company Report.
Referensi: §16 keputusan #10.
"""

QUARTERLY_HISTORY_QUARTERS: int = 6
"""Jumlah kuartal terakhir data keuangan kuartalan yang disimpan ke KB.

Diambil dari historical_financials_quarterly pada Company Report.
Referensi: §16 keputusan #11.
"""

# ── Sectors.app API ───────────────────────────────────────────────────────────

SECTORS_BASE_URL: str = "https://api.sectors.app/v2"
"""Base URL Sectors.app API versi 2."""

SECTORS_API_KEY: str | None = os.getenv("SECTORS_API_KEY")
"""API key Sectors.app — wajib ada di .env, tidak boleh hardcode."""

API_TIMEOUT_SEC: int = 30
"""Timeout per request HTTP ke Sectors.app (detik)."""

API_RETRY_MAX: int = 3
"""Maksimum percobaan ulang setelah rate limit (HTTP 429) atau koneksi gagal."""

API_BACKOFF_INITIAL_SEC: float = 2.0
"""Delay awal exponential backoff (detik).

Pola: 2 → 4 → 8 detik. Referensi: §5 rate handling.
"""

API_SLEEP_BETWEEN_TICKERS_SEC: float = 5.0
"""Jeda antar-ticker dalam loop extract (detik) untuk menghormati rate limit."""

# ── Universe Saham IDX30 ──────────────────────────────────────────────────────

IDX30_TICKERS: list[str] = [
    "AADI", "ADRO", "AMRT", "ANTM", "ASII",
    "BBCA", "BBNI", "BBRI", "BMRI", "BRPT",
    "BUMI", "CPIN", "EMTK", "GOTO", "ICBP",
    "INCO", "INDF", "INKP", "ISAT", "JPFA",
    "KLBF", "MBMA", "MDKA", "MEDC", "PGAS",
    "PGEO", "PTBA", "TLKM", "UNTR", "UNVR",
]
"""Daftar 30 ticker saham IDX30 per April 2026.

Diurutkan alfabet untuk output yang deterministik.
Jika komposisi IDX30 berubah (delisting / rebalancing),
daftar ini WAJIB diperbarui secara eksplisit oleh peneliti.
"""

assert len(IDX30_TICKERS) == 30, (
    f"IDX30_TICKERS harus tepat 30 ticker, ditemukan {len(IDX30_TICKERS)}"
)

# ── Helper ────────────────────────────────────────────────────────────────────

def get_snapshot_date(override: str | None = None) -> str:
    """Kembalikan tanggal snapshot sebagai string ISO 8601 (YYYY-MM-DD).

    Digunakan sebagai `snapshot_date` di metadata setiap dokumen ChromaDB
    dan sebagai suffix nama file output, sehingga seluruh dokumen dalam
    satu ETL run terikat ke satu tanggal yang sama (reproducibility).

    Args:
        override: Jika diberikan, nilai ini yang dipakai (untuk CLI
            ``--snapshot-date``). Harus dalam format YYYY-MM-DD.
            Jika None, gunakan tanggal hari ini.

    Returns:
        String tanggal format ISO 8601, contoh: ``"2026-05-26"``.
    """
    if override is not None:
        return override
    return date.today().isoformat()
