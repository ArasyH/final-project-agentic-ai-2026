"""
Extract raw data dari Sectors.app API v2 untuk semua ticker IDX30.

Posisi dalam pipeline ETL:
    Sectors.app API → extract.py → etl/raw_data/{ticker}_{snapshot_date}.json
    → transform.py / aggregate.py

Setiap file JSON output berisi:
    - ``fundamentals``      : Company Report penuh (dict dari Sectors.app)
    - ``daily_prices_30d``  : Harga harian 30 hari terakhir (list dari
                              Daily Transaction endpoint)
    - ``ticker``            : Simbol saham
    - ``snapshot_date``     : Tanggal ETL run ISO 8601

``daily_prices_30d`` dibutuhkan oleh:
    - ``transform.build_price_snapshot`` → field "perubahan 30 hari" per saham
    - ``aggregate._load_all_metrics``    → ranking penurunan harga 30 hari
    - ``aggregate.build_sector_doc``     → kinerja harga 30 hari per sektor

CSV Audit Log:
    Setiap extract_ticker menulis satu baris ke
    ``etl/logs/api_calls.csv`` (kolom: timestamp, symbol, endpoints_called,
    company_report_status, daily_tx_status). Log lengkap per-request
    (timestamp, latency, response_size) akan diimplementasikan di Task 8
    (etl/etl_logger.py).

Referensi:
    §5  — Sectors.app API endpoints + rate handling
    §7.5 — Error handling: 4xx skip, 5xx/429 retry, ConnectionError retry
    §16 #2 — Daily Transaction hanya 30 hari (bukan time series panjang)
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from etl import config
from etl.api_client import SectorsAPIError, fetch_sectors, sleep_between_tickers

logger = logging.getLogger(__name__)

# ── Konstanta ─────────────────────────────────────────────────────────────────

_DAILY_TX_LOOKBACK_DAYS: int = 30
"""Jumlah hari lookback untuk Daily Transaction endpoint (§16 #2)."""


# ── Private helpers ───────────────────────────────────────────────────────────

def _get_csv_writer(log_path: Path) -> tuple[Any, Any]:
    """Buka (atau buat) api_calls.csv dan kembalikan (file_handle, csv.writer).

    Header ditulis hanya jika file baru. Caller bertanggung jawab memanggil
    ``file_handle.close()`` setelah selesai.

    Args:
        log_path: Path lengkap ke file CSV log.

    Returns:
        Tuple ``(file_handle, csv.writer)``.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    fh = log_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(
            ["timestamp", "symbol", "endpoint", "status"]
        )
    return fh, writer


def _log_api_call(
    writer: Any,
    symbol: str,
    endpoint: str,
    status: str,
) -> None:
    """Tulis satu baris audit ke api_calls.csv.

    Args:
        writer: csv.writer object.
        symbol: Ticker saham, mis. ``"BBCA"``.
        endpoint: Endpoint yang dipanggil, mis. ``"company_report"``.
        status: ``"ok"`` atau deskripsi error singkat.
    """
    writer.writerow([date.today().isoformat(), symbol, endpoint, status])


# ── Fetch functions ───────────────────────────────────────────────────────────

def fetch_company_report(ticker: str) -> dict[str, Any]:
    """Ambil Company Report lengkap untuk satu ticker dari Sectors.app.

    Memanggil ``GET /v2/company/report/{ticker}/``. Response berisi semua
    section: overview, valuation, future, financials, dividend, management,
    ownership, peers.

    Args:
        ticker: Simbol saham IDX30, mis. ``"BBCA"``.

    Returns:
        Response JSON sebagai dict.

    Raises:
        SectorsAPIError: Jika HTTP error non-retryable atau retry exhausted.
    """
    return fetch_sectors(f"company/report/{ticker}/")


def fetch_daily_transaction_30d(
    ticker: str,
    end_date: str,
) -> list[dict[str, Any]]:
    """Ambil harga harian 30 hari terakhir dari Daily Transaction endpoint.

    Memanggil ``GET /v2/transaction/daily/{ticker}/?start=...&end=...``
    dengan 1 request saja (30 hari < 90 hari limit per §5). Response bisa
    berupa list langsung atau dict dengan key ``"data"``.

    Args:
        ticker: Simbol saham IDX30.
        end_date: Tanggal akhir ISO 8601, biasanya snapshot_date. Tanggal
            mulai dihitung mundur ``_DAILY_TX_LOOKBACK_DAYS`` hari.

    Returns:
        List dict harga harian. Setiap entry biasanya berisi ``date``,
        ``close``, ``open``, ``high``, ``low``, ``volume``. List kosong
        jika tidak ada data untuk rentang tersebut.

    Raises:
        SectorsAPIError: Jika HTTP error non-retryable atau retry exhausted.
    """
    start = (
        date.fromisoformat(end_date) - timedelta(days=_DAILY_TX_LOOKBACK_DAYS)
    ).isoformat()

    # Endpoint aktual: /v2/daily/{ticker}/ (bukan transaction/daily/ seperti di §5)
    # Diverifikasi 2026-05-28 — 404 pada transaction/daily/, OK pada daily/.
    raw = fetch_sectors(
        f"daily/{ticker}/",
        params={"start": start, "end": end_date},
    )

    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return raw.get("data", [])
    return []


# ── Per-ticker extract ────────────────────────────────────────────────────────

def extract_ticker(
    ticker: str,
    snapshot_date: str,
    csv_writer: Any | None = None,
) -> None:
    """Ekstrak Company Report + harga 30 hari untuk satu ticker, simpan ke JSON.

    Output: ``{config.RAW_DATA_DIR}/{ticker}_{snapshot_date}.json``

    Jika Company Report gagal (SectorsAPIError), ticker di-skip dan error
    di-log — tidak crash ETL (§7.5). Jika hanya Daily Transaction gagal,
    file tetap disimpan dengan ``daily_prices_30d: []`` dan warning di-log.

    Args:
        ticker: Simbol saham IDX30.
        snapshot_date: Tanggal snapshot ISO 8601, mis. ``"2026-05-28"``.
        csv_writer: Opsional csv.writer untuk audit log. Jika None, tidak
            ada CSV logging untuk call ini.

    Raises:
        Tidak ada — semua error di-log dan di-handle internal.
    """
    logger.info("extract_ticker: mulai %s snapshot=%s", ticker, snapshot_date)

    # ── Company Report (wajib) ─────────────────────────────────────────────
    try:
        company_report = fetch_company_report(ticker)
        logger.info("extract_ticker: company_report %s OK", ticker)
        if csv_writer:
            _log_api_call(csv_writer, ticker, "company_report", "ok")
    except SectorsAPIError as exc:
        logger.error(
            "extract_ticker: company_report %s GAGAL — %s. Ticker di-skip.",
            ticker, exc,
        )
        if csv_writer:
            _log_api_call(csv_writer, ticker, "company_report", f"error: {exc}")
        return

    # ── Daily Transaction 30 hari (opsional) ──────────────────────────────
    daily_prices: list[dict[str, Any]] = []
    try:
        daily_prices = fetch_daily_transaction_30d(ticker, snapshot_date)
        logger.info(
            "extract_ticker: daily_transaction %s OK — %d records",
            ticker, len(daily_prices),
        )
        if csv_writer:
            _log_api_call(csv_writer, ticker, "daily_transaction", "ok")
    except SectorsAPIError as exc:
        logger.warning(
            "extract_ticker: daily_transaction %s GAGAL — %s. "
            "change_30d tidak tersedia untuk ticker ini.",
            ticker, exc,
        )
        if csv_writer:
            _log_api_call(csv_writer, ticker, "daily_transaction", f"error: {exc}")
        # daily_prices tetap [] — extract lanjut dengan data fundamentals saja

    # ── Simpan JSON ────────────────────────────────────────────────────────
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {
        "ticker": ticker,
        "snapshot_date": snapshot_date,
        "extracted_at": snapshot_date,
        "fundamentals": company_report,
        "daily_prices_30d": daily_prices,
    }

    json_path = config.RAW_DATA_DIR / f"{ticker}_{snapshot_date}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(
        "extract_ticker: saved %s (%d fundamentals keys, %d price records)",
        json_path.name,
        len(company_report),
        len(daily_prices),
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def extract_all(snapshot_date: str) -> dict[str, Any]:
    """Ekstrak semua 30 ticker IDX30 dan simpan ke etl/raw_data/.

    Urutan per ticker:
    1. ``fetch_company_report`` — wajib; jika gagal, ticker di-skip
    2. ``fetch_daily_transaction_30d`` — opsional; gagal = daily_prices_30d=[]
    3. Simpan JSON → ``{RAW_DATA_DIR}/{ticker}_{snapshot_date}.json``
    4. ``sleep_between_tickers()`` (``config.API_SLEEP_BETWEEN_TICKERS_SEC``)

    CSV audit log ditulis ke ``config.LOGS_DIR/api_calls.csv``.

    Args:
        snapshot_date: Tanggal snapshot ISO 8601, mis. ``"2026-05-28"``.

    Returns:
        Dict summary: ``{"snapshot_date": s, "ok": N, "skipped": M,
        "skipped_tickers": [...]}``.
    """
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = config.LOGS_DIR / "api_calls.csv"
    fh, writer = _get_csv_writer(csv_path)

    ok_count = 0
    skipped: list[str] = []

    try:
        for i, ticker in enumerate(config.IDX30_TICKERS):
            json_path = config.RAW_DATA_DIR / f"{ticker}_{snapshot_date}.json"
            extract_ticker(ticker, snapshot_date, csv_writer=writer)

            if json_path.exists():
                ok_count += 1
            else:
                skipped.append(ticker)

            # Jeda antar ticker — skip untuk ticker terakhir
            if i < len(config.IDX30_TICKERS) - 1:
                sleep_between_tickers()
    finally:
        fh.close()

    summary: dict[str, Any] = {
        "snapshot_date": snapshot_date,
        "ok": ok_count,
        "skipped": len(skipped),
        "skipped_tickers": skipped,
    }

    logger.info(
        "extract_all selesai: %d ok, %d skipped untuk snapshot %s",
        ok_count, len(skipped), snapshot_date,
    )
    return summary
