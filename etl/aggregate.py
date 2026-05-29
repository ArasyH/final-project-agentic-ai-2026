"""
Build cross-saham aggregate documents untuk kategori ranking dan sector_query.

Posisi dalam pipeline ETL:
    etl/raw_data/*.json → aggregate.py → list[DocOutput] → load.py → ChromaDB

Input  : raw JSON files di ``config.RAW_DATA_DIR/{ticker}_{snapshot_date}.json``
         (file yang sama dengan transform.py; tidak ada HTTP call, tidak ada
         ChromaDB import di modul ini).
Output : list[DocOutput]
         - category="aggregate_ranking"  (8 dokumen, 1 per metrik ranking)
         - category="aggregate_sector"   (1 per sub-sektor yang ditemukan di data)

Motivasi H4 (Incorrect Inference):
    Tanpa aggregate docs, Generator harus menyimpulkan ranking dari sample
    partial hasil retrieval — rentan H4. Precomputed ranking docs memastikan
    jawaban berbasis data lengkap 30 saham, bukan inferensi dari subset.

Referensi evaluation_dataset.json (app/data/evaluation_dataset.json):
    Ranking  Q025-Q034 (10 pertanyaan; Q032 coverage=none → tidak butuh doc)
    Sector   Q041-Q050 (10 pertanyaan)

CATATAN Q034 (penurunan harga 30 hari, coverage="full"):
    ``aggregate_ranking_top_daily_change`` menggunakan ``daily_close_change``
    (perubahan harian 1 hari) bukan 30 hari, karena extract.py saat ini tidak
    menarik daily transaction endpoint. Konten dokumen jujur menyebut
    "perubahan harian". Peneliti: konfirmasi apakah 30-hari data dari daily
    transaction endpoint diperlukan (§14 eskalasi wajib).

Referensi:
    §4  — Kategori dataset: ranking, sector_query
    §6  — Naming convention dokumen + metadata ChromaDB
    §15 — ChromaDB Metadata Schema (7 field)
    §16 #4 — H4 mitigation via precomputed aggregate
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from etl import config
from etl.schemas import DocMetadata, DocOutput

logger = logging.getLogger(__name__)


# ── Format helpers (identik dengan transform.py — duplikat disengaja agar ──────
#    aggregate.py tidak bergantung pada private API transform.py)             ──

def _fmt_idr(value: Any, fallback: str = "n/a") -> str:
    """Format angka ke IDR ringkas, konvensi Indonesia (titik ribuan, koma desimal)."""
    try:
        v = float(value)
        if v >= 1e12:
            return "Rp " + f"{v / 1e12:.1f}".replace(".", ",") + " T"
        if v >= 1e9:
            return "Rp " + f"{v / 1e9:.1f}".replace(".", ",") + " M"
        if v >= 1e6:
            return "Rp " + f"{v / 1e6:.1f}".replace(".", ",") + " jt"
        return "Rp " + f"{int(v):,}".replace(",", ".")
    except (TypeError, ValueError):
        return fallback


def _fmt_pct(value: Any, fallback: str = "n/a") -> str:
    """Format float 0-1 ke persen 2 desimal (koma), mis. 0.2043 → '20,43%'."""
    try:
        return f"{float(value) * 100:.2f}".replace(".", ",") + "%"
    except (TypeError, ValueError):
        return fallback


def _fmt_ratio(value: Any, fallback: str = "n/a") -> str:
    """Format rasio valuasi ke 2 desimal (koma), mis. 12.34 → '12,34x'."""
    try:
        return f"{float(value):.2f}".replace(".", ",") + "x"
    except (TypeError, ValueError):
        return fallback


# ── Private helpers ───────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    """Konversi nama sektor ke slug untuk doc_id, mis. 'Oil, Gas & Coal' → 'oil_gas_coal'."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _load_all_metrics(snapshot_date: str) -> dict[str, dict[str, Any]]:
    """Muat dan ekstrak metrik agregat dari raw JSON semua ticker IDX30.

    Membaca ``{config.RAW_DATA_DIR}/{ticker}_{snapshot_date}.json`` untuk
    setiap ticker. Ticker dengan file tidak ditemukan di-skip tanpa error
    (sama dengan perilaku ``transform_all``). Error parsing JSON / KeyError
    di-log lalu di-skip.

    Args:
        snapshot_date: Tanggal snapshot ISO 8601, mis. ``"2026-05-28"``.

    Returns:
        Dict ``{ticker: metrics}`` di mana ``metrics`` berisi field:
        ``sector``, ``sub_sector``, ``market_cap``, ``last_close_price``,
        ``daily_change``, ``pe``, ``pb``, ``yield_ttm``, ``roe_2024``,
        ``revenue_2024``, ``revenue_2023``, ``rev_growth_2024``,
        ``total_debt_2024``.
    """
    results: dict[str, dict[str, Any]] = {}

    for ticker in config.IDX30_TICKERS:
        json_path: Path = config.RAW_DATA_DIR / f"{ticker}_{snapshot_date}.json"
        try:
            with json_path.open(encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
            raw: dict[str, Any] = data["fundamentals"]
        except FileNotFoundError:
            continue
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("aggregate._load_all_metrics: skip %s — %s", ticker, exc)
            continue

        ov = raw.get("overview", {})
        val = raw.get("valuation", {})
        fin = raw.get("financials", {})
        div = raw.get("dividend", {})

        # ── P/E dan P/B dari historical_valuation tahun terakhir ─────────
        hist_val: list[dict] = val.get("historical_valuation", [])
        latest_val: dict = {}
        if hist_val and isinstance(hist_val, list):
            try:
                latest_val = max(hist_val, key=lambda x: x.get("year", 0))
            except (TypeError, ValueError):
                pass

        pe: Any = latest_val.get("pe") or val.get("pe_ratio") or val.get("forward_pe")
        pb: Any = latest_val.get("pb") or val.get("pb_ratio")

        # ── Data keuangan tahunan 2024 & 2023 ─────────────────────────────
        annual: list[dict] = (
            fin.get("historical_financials") or fin.get("annual_data", [])
        )
        year_data: dict[str, dict] = {
            str(e.get("year", "")): e
            for e in annual
            if e.get("year")
        }
        entry_2024 = year_data.get("2024", {})
        entry_2023 = year_data.get("2023", {})

        revenue_2024: Any = entry_2024.get("revenue")
        revenue_2023: Any = entry_2023.get("revenue")
        total_assets_2024: Any = entry_2024.get("total_assets")
        total_equity_2024: Any = entry_2024.get("total_equity")
        earnings_2024: Any = entry_2024.get("earnings") or entry_2024.get("net_income")

        # Total debt 2024 ≈ total_assets − total_equity (total liabilitas)
        total_debt_2024: float | None = None
        if total_assets_2024 is not None and total_equity_2024 is not None:
            try:
                total_debt_2024 = float(total_assets_2024) - float(total_equity_2024)
            except (TypeError, ValueError):
                pass

        # Revenue growth 2024 vs 2023
        rev_growth_2024: float | None = None
        if revenue_2024 is not None and revenue_2023 is not None:
            try:
                rv24, rv23 = float(revenue_2024), float(revenue_2023)
                if rv23:
                    rev_growth_2024 = (rv24 - rv23) / rv23
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        # ROE 2024 dari historical_financial_ratio
        roe_2024: Any = None
        ratio_list: list[dict] = fin.get("historical_financial_ratio", [])
        for r in ratio_list:
            if str(r.get("year", "")) == "2024":
                roe_2024 = r.get("profitability", {}).get("roe")
                break
        # Fallback ke entry langsung (mock)
        if roe_2024 is None:
            roe_2024 = entry_2024.get("roe")

        # change_30d dari daily_prices_30d (Opsi B — konfirmasi peneliti 2026-05-28)
        daily_prices: list[dict[str, Any]] = data.get("daily_prices_30d", [])
        change_30d: float | None = None
        if len(daily_prices) >= 2:
            try:
                sorted_p = sorted(daily_prices, key=lambda x: x.get("date", ""))
                first_c = float(sorted_p[0]["close"])
                last_c = float(sorted_p[-1]["close"])
                if first_c:
                    change_30d = (last_c - first_c) / first_c
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                pass

        results[ticker] = {
            "sector": ov.get("sector", ""),
            "sub_sector": (
                ov.get("sub_sector") or ov.get("sub_industry") or ov.get("sector", "")
            ),
            "market_cap": ov.get("market_cap"),
            "last_close_price": ov.get("last_close_price"),
            "daily_change": ov.get("daily_close_change"),
            "change_30d": change_30d,
            "pe": pe,
            "pb": pb,
            "yield_ttm": div.get("yield_ttm"),
            "roe_2024": roe_2024,
            "revenue_2024": revenue_2024,
            "revenue_2023": revenue_2023,
            "rev_growth_2024": rev_growth_2024,
            "total_debt_2024": total_debt_2024,
            "earnings_2024": earnings_2024,
        }

    return results


def _make_aggregate_metadata(
    category: str,
    doc_id: str,
    snapshot_date: str,
    sector: str = "",
    period: str = "",
) -> DocMetadata:
    """Buat DocMetadata untuk aggregate doc (symbol kosong, source_endpoint None).

    Args:
        category: ``"aggregate_ranking"`` atau ``"aggregate_sector"``.
        doc_id: ID unik, mis. ``"aggregate_ranking_top_marketcap"``.
        snapshot_date: Tanggal ETL run ISO 8601.
        sector: Nama sektor untuk aggregate_sector doc. Kosong untuk ranking.
        period: Periode data (``""`` untuk aggregate).

    Returns:
        DocMetadata tervalidasi Pydantic.
    """
    return DocMetadata(
        category=category,  # type: ignore[arg-type]
        symbol="",
        sector=sector,
        period=period,
        snapshot_date=snapshot_date,
        doc_id=doc_id,
        source_endpoint=None,
    )


# ── Ranking builders ──────────────────────────────────────────────────────────

def build_ranking_top_marketcap(
    metrics: dict[str, dict[str, Any]],
    snapshot_date: str,
) -> DocOutput:
    """Bangun doc ranking 10 saham IDX30 berdasarkan market cap terbesar.

    Menjawab Q025: "Apa 3 saham IDX30 dengan kapitalisasi pasar terbesar?"
    Data: overview.market_cap.

    Args:
        metrics: Output ``_load_all_metrics()``.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        DocOutput dengan ``category="aggregate_ranking"``.
    """
    ranked = sorted(
        [(t, m["market_cap"]) for t, m in metrics.items() if m["market_cap"] is not None],
        key=lambda x: float(x[1]),
        reverse=True,
    )[:10]

    if not ranked:
        entries = "data tidak tersedia"
    else:
        entries = " ".join(
            f"{i+1}) {t} {_fmt_idr(v)}" for i, (t, v) in enumerate(ranked)
        )

    content = (
        f"IDX30 market cap (kapitalisasi pasar) terbesar per {snapshot_date}: "
        f"{entries}."
    ).strip()

    return DocOutput(
        content=content,
        metadata=_make_aggregate_metadata(
            "aggregate_ranking",
            "aggregate_ranking_top_marketcap",
            snapshot_date,
        ),
    )


def build_ranking_top_roe_2024(
    metrics: dict[str, dict[str, Any]],
    snapshot_date: str,
) -> DocOutput:
    """Bangun doc ranking 10 saham IDX30 berdasarkan ROE tertinggi tahun 2024.

    Menjawab Q026: "Saham IDX30 mana yang memiliki ROE tertinggi pada tahun 2024?"
    Data: financials.historical_financial_ratio[year=2024].profitability.roe.

    Args:
        metrics: Output ``_load_all_metrics()``.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        DocOutput dengan ``category="aggregate_ranking"``.
    """
    ranked = sorted(
        [(t, m["roe_2024"]) for t, m in metrics.items() if m["roe_2024"] is not None],
        key=lambda x: float(x[1]),
        reverse=True,
    )[:10]

    if not ranked:
        entries = "data tidak tersedia"
    else:
        entries = " ".join(
            f"{i+1}) {t} {_fmt_pct(v)}" for i, (t, v) in enumerate(ranked)
        )

    content = (
        f"IDX30 ROE (Return on Equity) tertinggi tahun 2024: "
        f"{entries}."
    ).strip()

    return DocOutput(
        content=content,
        metadata=_make_aggregate_metadata(
            "aggregate_ranking",
            "aggregate_ranking_top_roe_2024",
            snapshot_date,
        ),
    )


def build_ranking_top_per_lowest(
    metrics: dict[str, dict[str, Any]],
    snapshot_date: str,
) -> DocOutput:
    """Bangun doc ranking 10 saham IDX30 berdasarkan P/E TTM terendah.

    Menjawab Q027: "Saham IDX30 mana yang memiliki P/E TTM terendah?"
    Data: valuation.historical_valuation (tahun terakhir, field pe).
    Hanya saham dengan P/E > 0 yang dimasukkan (P/E negatif = rugi, misleading).

    Args:
        metrics: Output ``_load_all_metrics()``.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        DocOutput dengan ``category="aggregate_ranking"``.
    """
    ranked = sorted(
        [
            (t, m["pe"])
            for t, m in metrics.items()
            if m["pe"] is not None and float(m["pe"]) > 0
        ],
        key=lambda x: float(x[1]),
    )[:10]

    if not ranked:
        entries = "data tidak tersedia"
    else:
        entries = " ".join(
            f"{i+1}) {t} {_fmt_ratio(v)}" for i, (t, v) in enumerate(ranked)
        )

    content = (
        f"IDX30 P/E (Price-to-Earnings) TTM (Trailing Twelve Months) "
        f"terendah per {snapshot_date}: {entries}."
    ).strip()

    return DocOutput(
        content=content,
        metadata=_make_aggregate_metadata(
            "aggregate_ranking",
            "aggregate_ranking_top_per_lowest",
            snapshot_date,
        ),
    )


def build_ranking_top_dividend_yield(
    metrics: dict[str, dict[str, Any]],
    snapshot_date: str,
) -> DocOutput:
    """Bangun doc ranking 10 saham IDX30 berdasarkan dividend yield TTM tertinggi.

    Menjawab Q028: "Apa 3 saham IDX30 dengan dividend yield TTM tertinggi?"
    Data: dividend.yield_ttm.

    Args:
        metrics: Output ``_load_all_metrics()``.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        DocOutput dengan ``category="aggregate_ranking"``.
    """
    ranked = sorted(
        [
            (t, m["yield_ttm"])
            for t, m in metrics.items()
            if m["yield_ttm"] is not None
        ],
        key=lambda x: float(x[1]),
        reverse=True,
    )[:10]

    if not ranked:
        entries = "data tidak tersedia"
    else:
        entries = " ".join(
            f"{i+1}) {t} {_fmt_pct(v)}" for i, (t, v) in enumerate(ranked)
        )

    content = (
        f"IDX30 dividend yield TTM (Trailing Twelve Months) tertinggi "
        f"per {snapshot_date}: {entries}."
    ).strip()

    return DocOutput(
        content=content,
        metadata=_make_aggregate_metadata(
            "aggregate_ranking",
            "aggregate_ranking_top_dividend_yield",
            snapshot_date,
        ),
    )


def build_ranking_top_revenue_growth(
    metrics: dict[str, dict[str, Any]],
    snapshot_date: str,
) -> DocOutput:
    """Bangun doc ranking 10 saham IDX30 berdasarkan revenue growth 2024 vs 2023.

    Menjawab Q030: "Saham IDX30 mana yang mencatat pertumbuhan pendapatan
    tahunan tertinggi pada tahun 2024?"
    Data: dihitung dari historical_financials (revenue_2024 - revenue_2023) / revenue_2023.

    Args:
        metrics: Output ``_load_all_metrics()``.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        DocOutput dengan ``category="aggregate_ranking"``.
    """
    ranked = sorted(
        [
            (t, m["rev_growth_2024"])
            for t, m in metrics.items()
            if m["rev_growth_2024"] is not None
        ],
        key=lambda x: float(x[1]),
        reverse=True,
    )[:10]

    if not ranked:
        entries = "data tidak tersedia"
    else:
        entries = " ".join(
            f"{i+1}) {t} {_fmt_pct(v)}" for i, (t, v) in enumerate(ranked)
        )

    content = (
        f"IDX30 pertumbuhan pendapatan (revenue growth) tertinggi 2024 vs 2023: "
        f"{entries}."
    ).strip()

    return DocOutput(
        content=content,
        metadata=_make_aggregate_metadata(
            "aggregate_ranking",
            "aggregate_ranking_top_revenue_growth",
            snapshot_date,
        ),
    )


def build_ranking_top_pb_lowest(
    metrics: dict[str, dict[str, Any]],
    snapshot_date: str,
) -> DocOutput:
    """Bangun doc ranking 10 saham IDX30 berdasarkan P/B MRQ terendah.

    Menjawab Q031: "Apa 5 saham IDX30 dengan rasio P/B terendah berdasarkan
    kuartal terakhir?"
    Data: valuation.historical_valuation (tahun terakhir, field pb).
    Hanya saham dengan P/B > 0 yang dimasukkan.

    Args:
        metrics: Output ``_load_all_metrics()``.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        DocOutput dengan ``category="aggregate_ranking"``.
    """
    ranked = sorted(
        [
            (t, m["pb"])
            for t, m in metrics.items()
            if m["pb"] is not None and float(m["pb"]) > 0
        ],
        key=lambda x: float(x[1]),
    )[:10]

    if not ranked:
        entries = "data tidak tersedia"
    else:
        entries = " ".join(
            f"{i+1}) {t} {_fmt_ratio(v)}" for i, (t, v) in enumerate(ranked)
        )

    content = (
        f"IDX30 P/B (Price-to-Book) MRQ (Most Recent Quarter) "
        f"terendah per {snapshot_date}: {entries}."
    ).strip()

    return DocOutput(
        content=content,
        metadata=_make_aggregate_metadata(
            "aggregate_ranking",
            "aggregate_ranking_top_pb_lowest",
            snapshot_date,
        ),
    )


def build_ranking_top_total_debt(
    metrics: dict[str, dict[str, Any]],
    snapshot_date: str,
) -> DocOutput:
    """Bangun doc ranking 10 saham IDX30 berdasarkan total utang (liabilitas) 2024.

    Menjawab Q033: "Apa 3 saham IDX30 dengan total utang terbesar pada akhir
    tahun fiskal 2024?"
    Total utang ≈ total_assets − total_equity (total liabilitas) per 2024.

    Args:
        metrics: Output ``_load_all_metrics()``.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        DocOutput dengan ``category="aggregate_ranking"``.
    """
    ranked = sorted(
        [
            (t, m["total_debt_2024"])
            for t, m in metrics.items()
            if m["total_debt_2024"] is not None
        ],
        key=lambda x: float(x[1]),
        reverse=True,
    )[:10]

    if not ranked:
        entries = "data tidak tersedia"
    else:
        entries = " ".join(
            f"{i+1}) {t} {_fmt_idr(v)}" for i, (t, v) in enumerate(ranked)
        )

    content = (
        f"IDX30 total liabilitas (utang) terbesar akhir tahun fiskal 2024: "
        f"{entries}."
    ).strip()

    return DocOutput(
        content=content,
        metadata=_make_aggregate_metadata(
            "aggregate_ranking",
            "aggregate_ranking_top_total_debt",
            snapshot_date,
        ),
    )


def build_ranking_top_daily_change(
    metrics: dict[str, dict[str, Any]],
    snapshot_date: str,
) -> DocOutput:
    """Bangun doc ranking saham IDX30 berdasarkan perubahan harga 30 hari.

    Menjawab Q034: "Saham IDX30 mana yang mengalami penurunan harga terbesar
    dalam 30 hari terakhir?" (expected_kb_coverage=full).

    Menggunakan ``change_30d`` (dihitung dari ``daily_prices_30d`` yang
    di-fetch oleh ``extract.fetch_daily_transaction_30d``). Jika ticker
    tidak memiliki data 30-hari (daily_prices_30d kosong), ticker tersebut
    tidak dimasukkan ke ranking.

    Ranking ascending: paling negatif di atas (penurunan terbesar).

    Args:
        metrics: Output ``_load_all_metrics()``.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        DocOutput dengan ``category="aggregate_ranking"``.
    """
    # Prioritas: change_30d (30-hari). Tidak ada fallback ke daily_change
    # untuk menjaga akurasi — konten harus jujur "30 hari".
    ranked = sorted(
        [
            (t, m["change_30d"])
            for t, m in metrics.items()
            if m.get("change_30d") is not None
        ],
        key=lambda x: float(x[1]),
    )[:10]

    if not ranked:
        entries = "data tidak tersedia (daily_prices_30d kosong)"
    else:
        entries = " ".join(
            f"{i+1}) {t} {_fmt_pct(v)}" for i, (t, v) in enumerate(ranked)
        )

    content = (
        f"IDX30 perubahan harga 30 hari terbesar (naik/turun) per {snapshot_date}: "
        f"{entries}."
    ).strip()

    return DocOutput(
        content=content,
        metadata=_make_aggregate_metadata(
            "aggregate_ranking",
            "aggregate_ranking_top_daily_change",
            snapshot_date,
        ),
    )


# ── Sector builder ────────────────────────────────────────────────────────────

def build_sector_doc(
    sub_sector: str,
    ticker_metrics: list[tuple[str, dict[str, Any]]],
    snapshot_date: str,
) -> DocOutput:
    """Bangun doc ringkasan satu sub-sektor IDX30.

    Field wajib (semua harus masuk, tidak boleh di-drop):
    - Daftar anggota ticker
    - Market cap terbesar (saham tunggal)
    - **Total market cap sektor** (jumlah semua anggota) — Q043
    - ROE rata-rata 2024
    - Dividend yield TTM rata-rata
    - **Dividend yield TTM tertinggi** (ticker + nilai) — Q044
    - Perubahan harga 30 hari rata-rata — Q047
    - Revenue 2024 rata-rata — Q050

    Token budget ≤128 hard limit (§16 #6). Strategi:
    1. Coba susun semua field dengan ekspansi penuh (mis. "ROE (Return on
       Equity) rata-rata").
    2. Jika >128T, rebuild dengan short form: hilangkan ekspansi kurung
       (mis. "ROE rata-rata", "Dividend yield TTM rata-rata") — ranking docs
       tidak terpengaruh, hanya sector doc yang bersangkutan.
    3. Jika masih >128T setelah short form → raise ``ValueError`` agar
       caller log dan bisa split di task berikutnya.

    Args:
        sub_sector: Nama sub-sektor, mis. ``"Banks"`` atau ``"Oil, Gas & Coal"``.
        ticker_metrics: List pasangan ``(ticker, metrics_dict)`` anggota sektor.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        DocOutput dengan ``category="aggregate_sector"``,
        ``sector=sub_sector``.

    Raises:
        ValueError: Jika token count melebihi MAX_DOC_TOKENS bahkan setelah
            short form. Caller wajib log dan laporkan ke peneliti.
    """
    from etl.transform import count_tokens  # reuse tokenizer singleton

    tickers = [t for t, _ in ticker_metrics]
    ticker_list = ", ".join(tickers)

    # ── Hitung semua komponen nilai ───────────────────────────────────────
    mc_pairs = [
        (t, float(m["market_cap"]))
        for t, m in ticker_metrics
        if m["market_cap"] is not None
    ]
    top_mc_t, top_mc_v = max(mc_pairs, key=lambda x: x[1]) if mc_pairs else ("", 0)
    total_mc = sum(v for _, v in mc_pairs)

    roe_vals = [float(m["roe_2024"]) for _, m in ticker_metrics if m["roe_2024"] is not None]
    roe_avg = sum(roe_vals) / len(roe_vals) if roe_vals else None

    dy_pairs = [
        (t, float(m["yield_ttm"]))
        for t, m in ticker_metrics
        if m["yield_ttm"] is not None
    ]
    dy_avg = sum(v for _, v in dy_pairs) / len(dy_pairs) if dy_pairs else None
    top_dy_t, top_dy_v = max(dy_pairs, key=lambda x: x[1]) if dy_pairs else ("", 0)

    chg30_vals = [float(m["change_30d"]) for _, m in ticker_metrics if m.get("change_30d") is not None]
    chg30_avg = sum(chg30_vals) / len(chg30_vals) if chg30_vals else None

    rev_vals = [float(m["revenue_2024"]) for _, m in ticker_metrics if m["revenue_2024"] is not None]
    rev_avg = sum(rev_vals) / len(rev_vals) if rev_vals else None

    # ── Bangun semua fragmen string ────────────────────────────────────────
    def _assemble(short: bool) -> str:
        """Rakit konten. short=True → drop ekspansi kurung di field ini saja."""
        roe_label = "ROE rata-rata" if short else "ROE (Return on Equity) rata-rata"
        dy_label  = "Dividend yield TTM rata-rata" if short else "Dividend yield TTM (Trailing Twelve Months) rata-rata"

        parts = [f"Sektor {sub_sector} IDX30 ({len(tickers)} saham): {ticker_list}."]
        if mc_pairs:
            parts.append(f" Market cap terbesar: {top_mc_t} {_fmt_idr(top_mc_v)}.")
        if total_mc > 0:
            parts.append(f" Total market cap sektor: {_fmt_idr(total_mc)}.")
        if roe_avg is not None:
            parts.append(f" {roe_label} 2024: {_fmt_pct(roe_avg)}.")
        if dy_avg is not None:
            parts.append(f" {dy_label}: {_fmt_pct(dy_avg)}.")
        if dy_pairs:
            parts.append(f" Dividend yield TTM tertinggi: {top_dy_t} {_fmt_pct(top_dy_v)}.")
        if chg30_avg is not None:
            parts.append(f" Perubahan harga 30 hari rata-rata: {_fmt_pct(chg30_avg)}.")
        if rev_avg is not None:
            parts.append(f" Revenue 2024 rata-rata: {_fmt_idr(rev_avg)}.")
        return "".join(parts).strip()

    # Coba full form dulu
    content = _assemble(short=False)
    if count_tokens(content) > config.MAX_DOC_TOKENS:
        # Fallback ke short form (drop ekspansi kurung)
        content = _assemble(short=True)
        if count_tokens(content) > config.MAX_DOC_TOKENS:
            raise ValueError(
                f"build_sector_doc '{sub_sector}': {count_tokens(content)}T "
                f"melebihi {config.MAX_DOC_TOKENS}T bahkan setelah short form. "
                f"Perlu split doc — eskalasi ke peneliti."
            )

    doc_id = f"aggregate_sector_{_slug(sub_sector)}"
    return DocOutput(
        content=content,
        metadata=_make_aggregate_metadata(
            "aggregate_sector",
            doc_id,
            snapshot_date,
            sector=sub_sector,
        ),
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def aggregate_all(snapshot_date: str) -> list[DocOutput]:
    """Buat semua aggregate docs (ranking + sector) untuk snapshot tertentu.

    Urutan:
    1. ``_load_all_metrics(snapshot_date)`` — muat raw JSON semua ticker
    2. Build 8 ranking docs (cover Q025-Q034 kecuali Q032 coverage=none)
    3. Group ticker berdasarkan ``sub_sector``; build 1 sector doc per grup
    4. Kembalikan flat list semua DocOutput

    Jika tidak ada raw JSON yang ditemukan (extract belum dijalankan),
    fungsi ini mengembalikan list kosong dan mencatat WARNING.

    Args:
        snapshot_date: Tanggal snapshot ISO 8601, mis. ``"2026-05-28"``.

    Returns:
        List ``DocOutput`` siap di-upsert ke ChromaDB.
        Jumlah tipikal: 8 ranking + N sector docs (N = jumlah sub-sektor unik
        di data, biasanya 8-12 untuk IDX30).
    """
    metrics = _load_all_metrics(snapshot_date)

    if not metrics:
        logger.warning(
            "aggregate_all: tidak ada raw JSON untuk snapshot %s. "
            "Jalankan extract.py terlebih dahulu.",
            snapshot_date,
        )
        return []

    docs: list[DocOutput] = []

    # ── Ranking docs ──────────────────────────────────────────────────────
    ranking_builders = [
        build_ranking_top_marketcap,
        build_ranking_top_roe_2024,
        build_ranking_top_per_lowest,
        build_ranking_top_dividend_yield,
        build_ranking_top_revenue_growth,
        build_ranking_top_pb_lowest,
        build_ranking_top_total_debt,
        build_ranking_top_daily_change,
    ]
    for builder in ranking_builders:
        try:
            doc = builder(metrics, snapshot_date)
            docs.append(doc)
            logger.info(
                "aggregate_all: built %s (%d chars)",
                doc.metadata.doc_id,
                len(doc.content),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("aggregate_all: builder %s gagal — %s", builder.__name__, exc)

    # ── Sector docs (dinamis dari sub_sector) ─────────────────────────────
    sector_groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for ticker, m in metrics.items():
        sub = m.get("sub_sector") or m.get("sector") or "Unknown"
        sector_groups.setdefault(sub, []).append((ticker, m))

    for sub_sector, ticker_metrics_list in sorted(sector_groups.items()):
        if sub_sector in ("", "Unknown"):
            continue
        try:
            doc = build_sector_doc(sub_sector, ticker_metrics_list, snapshot_date)
            docs.append(doc)
            logger.info(
                "aggregate_all: built sector doc %s (%d chars)",
                doc.metadata.doc_id,
                len(doc.content),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "aggregate_all: sector %s gagal — %s", sub_sector, exc
            )

    logger.info(
        "aggregate_all: total %d aggregate docs untuk snapshot %s",
        len(docs),
        snapshot_date,
    )
    return docs
