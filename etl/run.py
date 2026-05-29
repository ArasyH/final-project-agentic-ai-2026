"""
CLI entry point untuk pipeline ETL IDX30.

Usage (dari root project):

    python -m etl.run
    python -m etl.run --snapshot-date 2026-05-28
    python -m etl.run --snapshot-date 2026-05-28 --skip-extract
    python -m etl.run --snapshot-date 2026-05-28 --no-reset

Phases:
    1. extract_all(snapshot_date)       — pull data dari Sectors.app API → raw_data/
    2. load_all(snapshot_date, reset)   — transform + aggregate + upsert ke ChromaDB

Flags:
    --snapshot-date YYYY-MM-DD
        Tanggal snapshot ISO 8601. Default: today.
        Menentukan suffix filename raw_data JSON dan nilai snapshot_date di metadata KB.
        Pin ke tanggal eksperimen untuk reproducibility (§7.4).

    --skip-extract
        Lewati fase extract (phase 1). Gunakan raw_data JSON yang sudah ada
        di etl/raw_data/{ticker}_{snapshot_date}.json.
        Berguna untuk re-run transform/aggregate/load tanpa memanggil API lagi,
        misalnya setelah perbaikan transform.py atau aggregate.py pada hari yang sama.
        CATATAN: file raw_data dengan snapshot_date yang ditentukan harus sudah ada.

    --no-reset
        Jangan hapus dan recreate ChromaDB collection sebelum upsert.
        Gunakan mode upsert-over ke collection existing.
        Default (tanpa flag ini): reset=True — collection lama dihapus terlebih dahulu.
        PERINGATAN: default reset=True akan menghapus SEMUA dokumen existing di
        collection config.CHROMA_COLLECTION_NAME. Konfirmasi sebelum jalankan
        ke database production.

Exit codes:
    0 — Sukses penuh: semua ticker di-extract, semua dokumen di-load.
    1 — Parsial: sebagian ticker skipped saat extract, load tetap berjalan
        dengan data yang berhasil di-extract.
    2 — Gagal fatal: extract crash total atau load gagal.

Output artifacts:
    etl/raw_data/{ticker}_{snapshot_date}.json      (phase 1, per ticker)
    etl/logs/api_calls.csv                          (audit log API calls dari extract)
    etl/logs/etl_run_{snapshot_date}.log            (log run ini — stdout + file)
    etl/logs/etl_run_{snapshot_date}.json           (run summary JSON untuk Bab III)
    <CHROMA_DB_PATH>/                               (phase 2 — ChromaDB collection)

Referensi:
    §4  — Tujuan ETL & deliverable (4 kategori dokumen)
    §7.4 — Reproducibility: idempotent, snapshot_date di-pin, summary JSON
    §8  — File Ownership Map: run.py sebagai entry point CLI
    §16 #4 — ETL mode: Idempotent Full Refresh
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

from etl import config
from etl import etl_logger
from etl.extract import extract_all
from etl.load import load_all

logger = logging.getLogger(__name__)


# ── Argparse ──────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """Parse argumen CLI.

    Returns:
        Namespace dengan field:
            - ``snapshot_date`` (str | None): nilai ``--snapshot-date``.
            - ``skip_extract`` (bool): flag ``--skip-extract``.
            - ``no_reset`` (bool): flag ``--no-reset``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m etl.run",
        description=(
            "Pipeline ETL IDX30: extract dari Sectors.app API → transform "
            "→ aggregate → upsert ke ChromaDB knowledge base."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--snapshot-date",
        metavar="YYYY-MM-DD",
        default=None,
        help=(
            "Tanggal snapshot ISO 8601 (contoh: 2026-05-28). "
            "Default: today. Menentukan suffix filename raw_data "
            "dan nilai metadata snapshot_date di KB."
        ),
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        default=False,
        help=(
            "Lewati fase extract; pakai raw_data JSON yang sudah ada. "
            "Berguna untuk re-run transform/aggregate/load tanpa memanggil API."
        ),
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        default=False,
        help=(
            "Jangan wipe ChromaDB collection sebelum upsert (upsert-over). "
            "Default tanpa flag: reset=True — wipe + recreate collection."
        ),
    )
    return parser.parse_args()


# ── Phase runners ─────────────────────────────────────────────────────────────


def _run_extract_phase(snapshot_date: str) -> dict[str, Any]:
    """Jalankan fase extract untuk semua 30 ticker IDX30.

    Memanggil ``extract_all(snapshot_date)`` yang meng-iterasi
    ``config.IDX30_TICKERS``, menyimpan JSON ke ``etl/raw_data/``, dan
    menulis audit log ke ``etl/logs/api_calls.csv``.

    Args:
        snapshot_date: Tanggal snapshot ISO 8601, mis. ``"2026-05-28"``.

    Returns:
        Summary dict dari extract_all:
        ``{"snapshot_date", "ok", "skipped", "skipped_tickers"}``.

    Raises:
        Exception: Propagate exception fatal dari extract_all ke caller
            (main() akan log dan exit dengan code 2).
    """
    logger.info("=== PHASE 1: EXTRACT ===  snapshot=%s", snapshot_date)
    summary = extract_all(snapshot_date)
    logger.info(
        "extract selesai: %d ok, %d skipped%s",
        summary["ok"],
        summary["skipped"],
        f" ({summary['skipped_tickers']})" if summary["skipped_tickers"] else "",
    )
    return summary


def _run_load_phase(snapshot_date: str, reset: bool) -> dict[str, Any]:
    """Jalankan fase load: transform → aggregate → upsert ke ChromaDB.

    Memanggil ``load_all(snapshot_date, reset)`` yang secara internal
    memanggil ``transform_all`` + ``aggregate_all`` lalu upsert ke collection
    ``config.CHROMA_COLLECTION_NAME``.

    Args:
        snapshot_date: Tanggal snapshot ISO 8601.
        reset: Jika ``True``, wipe + recreate ChromaDB collection sebelum upsert.
            Jika ``False``, upsert-over ke collection existing.

    Returns:
        Summary dict dari load_all:
        ``{"tickers", "documents", "aggregate_docs"}``.

    Raises:
        Exception: Propagate exception fatal dari load_all ke caller.
    """
    logger.info(
        "=== PHASE 2: LOAD ===  snapshot=%s  reset=%s",
        snapshot_date,
        reset,
    )
    summary = load_all(snapshot_date, reset=reset)
    logger.info(
        "load selesai: %d tickers, %d docs total (%d aggregate)",
        summary["tickers"],
        summary["documents"],
        summary["aggregate_docs"],
    )
    return summary


# ── Summary writer ────────────────────────────────────────────────────────────


def _write_final_summary(
    *,
    snapshot_date: str,
    started_at: datetime,
    t_start: float,
    extract_summary: dict[str, Any],
    load_summary: dict[str, Any],
    errors: list[str],
    exit_code: int,
    reset: bool,
) -> None:
    """Kumpulkan data run dan delegasikan penulisan ke etl_logger.

    Memanggil ``etl_logger.write_run_summary`` yang menyimpan JSON ke
    ``etl/logs/etl_run_{snapshot_date}.json``.

    Args:
        snapshot_date: Tanggal snapshot ISO 8601.
        started_at: Waktu mulai run (timezone-aware UTC).
        t_start: ``time.monotonic()`` pada mulai run, untuk hitung durasi.
        extract_summary: Dict hasil fase extract (atau ``{}`` jika skipped/gagal).
        load_summary: Dict hasil fase load (atau ``{}`` jika gagal).
        errors: List string pesan error yang terkumpul selama run.
        exit_code: Exit code final (0 / 1 / 2).
        reset: Apakah collection di-reset saat load.
    """
    finished_at = datetime.now(timezone.utc)
    duration_seconds = round(time.monotonic() - t_start, 1)

    data: dict[str, Any] = {
        "snapshot_date": snapshot_date,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration_seconds,
        "exit_code": exit_code,
        "phases": {
            "extract": {
                "skipped_phase": extract_summary.get("skipped_phase", False),
                "ok": extract_summary.get("ok", 0),
                "skipped_count": extract_summary.get("skipped", 0),
                "skipped_tickers": extract_summary.get("skipped_tickers", []),
            },
            "load": {
                "reset": reset,
                "tickers": load_summary.get("tickers", 0),
                "documents": load_summary.get("documents", 0),
                "aggregate_docs": load_summary.get("aggregate_docs", 0),
            },
        },
        "errors": errors,
    }

    etl_logger.write_run_summary(snapshot_date=snapshot_date, data=data)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    """Entry point utama pipeline ETL IDX30.

    Mengorkestrasi dua fase (extract → load) dengan logging terstruktur,
    error handling per-fase, dan penulisan run summary JSON. Idempotent:
    jalankan berkali-kali dengan snapshot_date yang sama menghasilkan KB
    yang identik (§7.4, §16 #4).

    Returns:
        Exit code integer:
            - 0 = sukses penuh
            - 1 = parsial (beberapa ticker skipped, load tetap jalan)
            - 2 = gagal fatal (extract crash total atau load gagal)
    """
    args = _parse_args()
    snapshot_date = config.get_snapshot_date(args.snapshot_date)
    reset = not args.no_reset

    # ── Logging setup ─────────────────────────────────────────────────────────
    etl_logger.configure_logging(
        log_dir=config.LOGS_DIR,
        snapshot_date=snapshot_date,
    )

    logger.info(
        "ETL pipeline dimulai — snapshot_date=%s, skip_extract=%s, reset=%s",
        snapshot_date,
        args.skip_extract,
        reset,
    )

    started_at = datetime.now(timezone.utc)
    t_start = time.monotonic()

    extract_summary: dict[str, Any] = {}
    load_summary: dict[str, Any] = {}
    errors: list[str] = []
    exit_code = 0

    # ── Phase 1: Extract ──────────────────────────────────────────────────────
    if args.skip_extract:
        logger.info("--skip-extract aktif: fase extract dilewati, pakai raw_data existing")
        extract_summary = {
            "skipped_phase": True,
            "ok": 0,
            "skipped": 0,
            "skipped_tickers": [],
        }
    else:
        try:
            extract_summary = _run_extract_phase(snapshot_date)
            if extract_summary["skipped"] > 0:
                exit_code = max(exit_code, 1)
                errors.append(
                    f"extract: {extract_summary['skipped']} ticker skipped — "
                    f"{extract_summary['skipped_tickers']}"
                )
        except Exception as exc:
            logger.error("extract phase gagal fatal: %s", exc, exc_info=True)
            errors.append(f"extract fatal: {exc}")
            exit_code = 2
            # Tanpa raw_data, load tidak bisa berjalan — tulis summary langsung
            _write_final_summary(
                snapshot_date=snapshot_date,
                started_at=started_at,
                t_start=t_start,
                extract_summary=extract_summary,
                load_summary=load_summary,
                errors=errors,
                exit_code=exit_code,
                reset=reset,
            )
            return exit_code

    # ── Phase 2: Load (transform + aggregate + upsert) ───────────────────────
    try:
        load_summary = _run_load_phase(snapshot_date, reset=reset)
    except Exception as exc:
        logger.error("load phase gagal fatal: %s", exc, exc_info=True)
        errors.append(f"load fatal: {exc}")
        exit_code = 2

    # ── Finalize ──────────────────────────────────────────────────────────────
    _write_final_summary(
        snapshot_date=snapshot_date,
        started_at=started_at,
        t_start=t_start,
        extract_summary=extract_summary,
        load_summary=load_summary,
        errors=errors,
        exit_code=exit_code,
        reset=reset,
    )

    if exit_code == 0:
        logger.info(
            "ETL pipeline SUKSES — %d dokumen di-load ke KB snapshot=%s",
            load_summary.get("documents", 0),
            snapshot_date,
        )
    elif exit_code == 1:
        logger.warning(
            "ETL pipeline PARSIAL — %d dokumen di-load, %d ticker skipped. "
            "Lihat errors di etl_run_%s.json",
            load_summary.get("documents", 0),
            extract_summary.get("skipped", 0),
            snapshot_date,
        )
    else:
        logger.error(
            "ETL pipeline GAGAL — lihat etl_run_%s.json untuk detail",
            snapshot_date,
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
