from __future__ import annotations
"""Bootstrap ChromaDB knowledge base untuk deployment Streamlit.

Modul ini dipanggil satu kali di startup Streamlit untuk memastikan koleksi KB
sudah terisi. Kalau kosong (deployment fresh), memicu ETL dari Sectors.app API.

Design principle:
- Tidak memodifikasi apapun di paket `etl/` — hanya import & panggil.
- Idempotent: run berkali-kali → no-op kalau KB sudah terisi.
- Cocok untuk container ephemeral (Streamlit Cloud) yang re-deploy sesekali.
"""

from typing import Callable, Optional

from app.services.retrieval_service import RetrievalService


def kb_document_count() -> int:
    """Return jumlah dokumen di collection KB.

    Returns:
        int: jumlah dokumen. 0 kalau collection kosong atau belum ada.
    """
    try:
        retrieval = RetrievalService()
        return int(retrieval.kb._collection.count())
    except Exception:
        return 0


def ensure_kb_ready(
    snapshot_date: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    """Pastikan KB terisi; jalankan ETL kalau kosong.

    Args:
        snapshot_date: Tanggal snapshot ISO 8601. Default: today.
        on_progress: Callback untuk pesan progres (dipakai Streamlit spinner).

    Returns:
        (ok, message). ok=True kalau KB siap dipakai; False kalau ETL gagal.
    """
    def _log(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    count = kb_document_count()
    if count > 0:
        _log(f"KB siap — {count} dokumen terdeteksi.")
        return True, f"kb_ready:{count}"

    _log("KB kosong — memulai ETL dari Sectors.app API...")

    try:
        from etl import config as etl_config
        from etl.extract import extract_all
        from etl.load import load_all
    except ImportError as exc:
        return False, f"import_error: {exc}"

    resolved_date = etl_config.get_snapshot_date(snapshot_date)

    try:
        _log(f"[1/2] Extract dari Sectors API (snapshot={resolved_date})...")
        extract_summary = extract_all(resolved_date)
        _log(
            f"Extract selesai — {extract_summary['ok']} ok, "
            f"{extract_summary['skipped']} skipped."
        )
    except Exception as exc:
        return False, f"extract_failed: {type(exc).__name__}: {exc}"

    try:
        _log("[2/2] Load ke ChromaDB (transform + aggregate + upsert)...")
        load_summary = load_all(resolved_date, reset=True)
        _log(
            f"Load selesai — {load_summary['tickers']} tickers, "
            f"{load_summary['documents']} dokumen."
        )
    except Exception as exc:
        return False, f"load_failed: {type(exc).__name__}: {exc}"

    final_count = kb_document_count()
    if final_count == 0:
        return False, "kb_empty_after_etl"

    return True, f"etl_ok:{final_count}"
