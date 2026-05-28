"""
HTTP client wrapper untuk Sectors.app API v2.

Menyediakan satu fungsi publik utama (``fetch_sectors``) dengan
exponential backoff retry, dan satu helper (``sleep_between_tickers``)
untuk jeda antar-ticker di loop extract.

Backoff pola: ``API_BACKOFF_INITIAL_SEC * (2 ** attempt)``
→ 2 → 4 → 8 detik (untuk ``API_RETRY_MAX = 3``).

Referensi: §5 system prompt — Sectors.app API rate handling strategy.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from etl import config

logger = logging.getLogger(__name__)

_session: requests.Session | None = None


class SectorsAPIError(Exception):
    """Raised setelah retry exhausted atau error fatal (mis. 401 auth).

    Tidak perlu ditangkap di dalam api_client — propagate ke extract.py
    agar caller memutuskan apakah skip ticker atau abort run.
    """


def _get_session() -> requests.Session:
    """Return singleton requests.Session dengan Authorization header.

    Lazy-init: session dibuat hanya pada pemanggilan pertama dan
    digunakan ulang di semua request berikutnya. Header auth mengikuti
    konvensi Sectors.app: ``Authorization: <raw_api_key>`` (tanpa prefix
    ``Bearer``), sesuai pola yang dipakai di extract.py existing.

    Returns:
        requests.Session yang sudah dikonfigurasi dengan header auth.

    Raises:
        SectorsAPIError: Jika ``config.SECTORS_API_KEY`` adalah ``None``.
    """
    global _session
    if _session is None:
        if config.SECTORS_API_KEY is None:
            raise SectorsAPIError("SECTORS_API_KEY tidak di-set di environment")
        _session = requests.Session()
        _session.headers.update({"Authorization": config.SECTORS_API_KEY})
    return _session


def fetch_sectors(
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Kirim GET request ke Sectors.app API v2 dengan exponential backoff.

    URL dibentuk dari ``config.SECTORS_BASE_URL`` + ``endpoint``.
    Retry dilakukan untuk HTTP 429, 5xx, timeout, dan connection error.
    Error 4xx selain 429 (terutama 401, 404) langsung raise tanpa retry.

    Args:
        endpoint: Path setelah base URL, mis. ``"company/report/BBCA/"``.
            Leading slash diabaikan.
        params: Query parameters opsional, mis.
            ``{"start": "2026-01-01", "end": "2026-04-30"}``.

    Returns:
        Parsed JSON response sebagai ``dict``.

    Raises:
        SectorsAPIError: Jika semua retry exhausted, atau response 4xx
            non-retryable diterima, atau API key tidak di-set.
    """
    session = _get_session()
    url = f"{config.SECTORS_BASE_URL}/{endpoint.lstrip('/')}"

    for attempt in range(config.API_RETRY_MAX):
        t_start = time.monotonic()
        try:
            response = session.get(url, params=params, timeout=config.API_TIMEOUT_SEC)
            response.raise_for_status()

            latency_ms = round((time.monotonic() - t_start) * 1000)
            logger.info(
                "fetch_sectors success",
                extra={
                    "endpoint": endpoint,
                    "attempt": attempt,
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                },
            )
            return response.json()

        except requests.HTTPError as exc:
            latency_ms = round((time.monotonic() - t_start) * 1000)
            status = response.status_code
            logger.info(
                "fetch_sectors http_error",
                extra={
                    "endpoint": endpoint,
                    "attempt": attempt,
                    "status_code": status,
                    "latency_ms": latency_ms,
                },
            )
            # Non-retryable: 4xx selain 429 (401 auth, 404 not found, dll.)
            if status != 429 and status < 500:
                raise SectorsAPIError(
                    f"HTTP {status} tidak di-retry: {endpoint}"
                ) from exc
            # Retryable: 429 rate limit atau 5xx server error
            if attempt < config.API_RETRY_MAX - 1:
                delay = config.API_BACKOFF_INITIAL_SEC * (2 ** attempt)
                time.sleep(delay)
            else:
                raise SectorsAPIError(
                    f"Retry exhausted setelah {config.API_RETRY_MAX} attempt, "
                    f"HTTP {status}: {endpoint}"
                ) from exc

        except (requests.Timeout, requests.ConnectionError) as exc:
            latency_ms = round((time.monotonic() - t_start) * 1000)
            logger.info(
                "fetch_sectors connection_error",
                extra={
                    "endpoint": endpoint,
                    "attempt": attempt,
                    "status_code": "N/A",
                    "latency_ms": latency_ms,
                },
            )
            if attempt < config.API_RETRY_MAX - 1:
                delay = config.API_BACKOFF_INITIAL_SEC * (2 ** attempt)
                time.sleep(delay)
            else:
                raise SectorsAPIError(
                    f"Koneksi gagal setelah {config.API_RETRY_MAX} attempt: {endpoint}"
                ) from exc

    raise SectorsAPIError(f"Unexpected: retry loop exhausted untuk {endpoint}")


def sleep_between_tickers() -> None:
    """Tidur selama ``config.API_SLEEP_BETWEEN_TICKERS_SEC`` detik.

    Dipanggil oleh extract.py di antara request antar-ticker untuk
    menghormati rate limit Sectors.app. Nilai dikontrol via config.py
    agar bisa diubah tanpa menyentuh kode extract.
    """
    time.sleep(config.API_SLEEP_BETWEEN_TICKERS_SEC)
