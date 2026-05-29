"""
Logging utilities untuk pipeline ETL IDX30.

Menyediakan dua fungsi publik yang dipakai oleh ``etl/run.py``:

    ``configure_logging(log_dir, snapshot_date)``
        Setup root logger dengan dua handler:
        - StreamHandler → stdout (level INFO)
        - FileHandler   → ``etl/logs/etl_run_{snapshot_date}.log`` (level DEBUG)

    ``write_run_summary(snapshot_date, data)``
        Tulis run summary ke ``etl/logs/etl_run_{snapshot_date}.json``.
        File ini adalah audit trail Bab III: merekam snapshot_date, timing,
        jumlah dokumen per fase, ticker yang di-skip, error list, dan exit code.

Gap yang diketahui (Known Limitation):
    Spec §7.4 minta full per-request ``api_calls.csv`` dengan kolom tambahan
    ``response_size_bytes`` dan integer ``status_code``. Implementasi penuh butuh
    perubahan ``api_client.py`` agar mengembalikan response metadata. Kolom
    tersebut tidak diimplementasikan dalam task ini — eskalasi ke peneliti jika
    dibutuhkan untuk Bab III. Saat ini ``api_calls.csv`` ditulis langsung oleh
    ``extract.py`` dengan kolom: timestamp, symbol, endpoint, status (string).

Referensi:
    §7.4 — Reproducibility: etl/logs/etl_run_{snapshot_date}.json
    §10  — Definition of Done: logging pakai logging module standar Python
    §11  — Setiap dokumen output diverifikasi (audit trail pendukung)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# ── Konstanta format ──────────────────────────────────────────────────────────

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
"""Format log: timestamp ISO 8601 + level + nama logger + pesan."""

_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"
"""Format datetime di log — ISO 8601 tanpa timezone suffix."""


# ── Public functions ──────────────────────────────────────────────────────────


def configure_logging(
    log_dir: Path | str,
    snapshot_date: str,
    *,
    stdout_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """Konfigurasi root logger untuk satu ETL run.

    Pasang dua handler ke root logger:
    - ``StreamHandler`` → stdout, level ``stdout_level`` (default INFO)
    - ``FileHandler``   → ``{log_dir}/etl_run_{snapshot_date}.log``,
      level ``file_level`` (default DEBUG — lebih verbose untuk audit)

    Idempotent: jika root logger sudah punya handler (mis. dipanggil dua kali
    dalam sesi yang sama), semua handler lama di-close dan di-remove terlebih
    dahulu sebelum handler baru dipasang.

    Args:
        log_dir: Direktori untuk menyimpan file log. Dibuat jika belum ada.
        snapshot_date: Tanggal snapshot ISO 8601, mis. ``"2026-05-28"``.
            Dipakai sebagai suffix nama file log.
        stdout_level: Level minimum untuk handler stdout. Default ``logging.INFO``.
        file_level: Level minimum untuk handler file. Default ``logging.DEBUG``.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()

    # Bersihkan handler lama agar tidak double-log jika dipanggil ulang
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)

    root.setLevel(logging.DEBUG)  # level root minimal = DEBUG; filter di handler

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    # Handler 1: stdout (INFO ke atas)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(stdout_level)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    # Handler 2: file log (DEBUG ke atas — lebih lengkap untuk audit)
    log_file = log_dir / f"etl_run_{snapshot_date}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Logging dikonfigurasi: stdout(INFO) + file(%s)",
        log_file.name,
    )


def write_run_summary(
    snapshot_date: str,
    data: dict[str, Any],
) -> Path:
    """Tulis run summary ke ``etl/logs/etl_run_{snapshot_date}.json``.

    File ini adalah audit trail untuk Bab III metodologi penelitian.
    Merekam: snapshot_date, timing (started_at, finished_at, duration_seconds),
    jumlah dokumen per fase, ticker yang di-skip, error list, dan exit code.

    Idempotent: jika file sudah ada (mis. re-run di hari yang sama),
    di-overwrite — konsisten dengan Idempotent Full Refresh (§16 #4).

    Args:
        snapshot_date: Tanggal snapshot ISO 8601. Dipakai sebagai suffix
            nama file output.
        data: Dict yang akan di-serialize ke JSON. Diharapkan berisi:
            ``snapshot_date``, ``started_at``, ``finished_at``,
            ``duration_seconds``, ``exit_code``, ``phases``
            (dengan sub-dict ``extract`` dan ``load``), ``errors``.

    Returns:
        ``Path`` absolut ke file JSON yang ditulis.

    Raises:
        OSError: Jika direktori tidak bisa dibuat atau file tidak bisa ditulis.
    """
    from etl import config  # import lokal untuk hindari circular dependency saat import

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.LOGS_DIR / f"etl_run_{snapshot_date}.json"

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)

    logging.getLogger(__name__).info(
        "Run summary ditulis: %s  (exit_code=%s, docs=%s, duration=%.1fs)",
        out_path.name,
        data.get("exit_code"),
        data.get("phases", {}).get("load", {}).get("documents"),
        data.get("duration_seconds", 0.0),
    )
    return out_path
