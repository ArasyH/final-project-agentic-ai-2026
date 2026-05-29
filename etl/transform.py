"""
Transform raw Company Report JSON → list[DocOutput] per kategori.

Posisi dalam pipeline ETL:
    raw JSON (etl/raw_data/) → transform.py → list[DocOutput] (in-memory)
    → load.py → ChromaDB

Tidak ada I/O disk output, tidak ada HTTP call, tidak ada ChromaDB import.
Output DocOutput di-upsert oleh load.py dengan metadata 7-field penuh.

Setiap dokumen output ≤ 128 token (hard limit embedder MiniLM-L12-v2).
Konten Bahasa Indonesia, faktual, tanpa interpretasi atau opini.
Singkatan diperluas pada kemunculan pertama per dokumen (§16 #13).

Referensi:
    §6  — Document Output Schema + naming convention
    §15 — ChromaDB Metadata Schema (7 field)
    §16 #10 — ANNUAL_HISTORY_YEARS = 5
    §16 #11 — QUARTERLY_HISTORY_QUARTERS = 6
    §16 #13 — ekspansi singkatan kemunculan pertama
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from etl import config
from etl.schemas import DocMetadata, DocOutput
from sentence_transformers import SentenceTransformer


# ── Tokenizer singleton ───────────────────────────────────────────────────────

_tokenizer: Any = None


def _get_tokenizer() -> Any:
    """Kembalikan singleton tokenizer dari EMBEDDER_MODEL.

    Lazy-init: ``SentenceTransformer`` dimuat hanya pada pemanggilan pertama
    dan di-cache di module-level ``_tokenizer``. Tokenizer yang dikembalikan
    adalah ``PreTrainedTokenizerFast`` WordPiece dari
    ``paraphrase-multilingual-MiniLM-L12-v2``.

    Returns:
        PreTrainedTokenizerFast yang dipakai oleh EMBEDDER_MODEL.
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = SentenceTransformer(config.EMBEDDER_MODEL).tokenizer
    return _tokenizer


def count_tokens(text: str) -> int:
    """Hitung jumlah token (termasuk [CLS] dan [SEP]) untuk teks.

    Menggunakan tokenizer yang sama dengan embedder sehingga hitungan
    akurat — dokumen dengan ``count_tokens(content) > MAX_DOC_TOKENS``
    akan ter-truncate saat di-embed dan bagian akhirnya tidak bisa
    di-retrieve.

    Args:
        text: Teks yang akan dihitung tokennya.

    Returns:
        Jumlah token int termasuk special tokens.
    """
    return len(_get_tokenizer().encode(text, add_special_tokens=True))


# ── Format helpers (private) ──────────────────────────────────────────────────

def _fmt_idr(value: Any, fallback: str = "data tidak tersedia") -> str:
    """Format angka ke representasi IDR ringkas dengan konvensi Indonesia.

    Scale otomatis ke T (triliun ≥1e12), M (miliar ≥1e9), atau jt (juta ≥1e6).
    Pemisah desimal: koma. Pemisah ribuan: titik (§13).

    Args:
        value: Nilai numerik yang akan di-cast ke float.
        fallback: String yang dikembalikan jika parse gagal.

    Returns:
        String IDR, mis. ``"Rp 112,0 T"``, ``"Rp 5,2 M"``, ``"Rp 9.450"``.
    """
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


def _fmt_pct(value: Any, fallback: str = "data tidak tersedia") -> str:
    """Format float 0-1 ke persentase dua desimal konvensi Indonesia.

    Args:
        value: Float antara 0–1 (mis. 0.2043).
        fallback: String jika parse gagal.

    Returns:
        String persen dengan koma desimal, mis. ``"20,43%"`` (§13).
    """
    try:
        return f"{float(value) * 100:.2f}".replace(".", ",") + "%"
    except (TypeError, ValueError):
        return fallback


# ── Daily price helper ───────────────────────────────────────────────────────

def _compute_change_30d(
    daily_prices: list[dict[str, Any]],
) -> float | None:
    """Hitung perubahan harga relatif dari daftar harga harian 30 hari.

    Rumus: ``(close_terakhir - close_pertama) / close_pertama``.
    Data diurutkan ascending by ``date`` untuk memastikan urutan benar.
    Mengembalikan ``None`` jika data kurang dari 2 titik, atau parsing gagal.

    Args:
        daily_prices: List dict harga harian dari Daily Transaction endpoint.
            Setiap entry harus memiliki field ``date`` (str ISO 8601) dan
            ``close`` (numeric).

    Returns:
        Float perubahan relatif (mis. ``-0.0312`` untuk turun 3,12%) atau
        ``None`` jika tidak bisa dihitung.
    """
    if len(daily_prices) < 2:
        return None
    try:
        sorted_prices = sorted(daily_prices, key=lambda x: x.get("date", ""))
        first_close = float(sorted_prices[0]["close"])
        last_close = float(sorted_prices[-1]["close"])
        if first_close == 0:
            return None
        return (last_close - first_close) / first_close
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


# ── Metadata helper ───────────────────────────────────────────────────────────

def _make_metadata(
    category: str,
    ticker: str,
    period: str,
    doc_id: str,
    snapshot_date: str,
    *,
    sector: str = "",
) -> DocMetadata:
    """Buat DocMetadata standar untuk dokumen per-saham.

    ``source_endpoint`` selalu ``"company_report"`` untuk semua dokumen
    yang dihasilkan transform.py. Aggregate docs dari aggregate.py
    menggunakan ``source_endpoint=None``.

    Args:
        category: Kategori dokumen sesuai Literal di DocMetadata.
        ticker: Simbol saham, mis. ``"BBCA"``.
        period: Periode data, mis. ``"2024"``, ``"Q1-2026"``,
            ``"snapshot"``, atau ``""`` untuk dokumen non-periodik.
        doc_id: ID unik untuk ChromaDB upsert (idempotent).
        snapshot_date: Tanggal ETL run ISO 8601, mis. ``"2026-05-28"``.
        sector: Sektor saham untuk metadata filtering. Default ``""``.

    Returns:
        DocMetadata yang sudah ter-validasi Pydantic.
    """
    return DocMetadata(
        category=category,  # type: ignore[arg-type]
        symbol=ticker,
        sector=sector,
        period=period,
        snapshot_date=snapshot_date,
        doc_id=doc_id,
        source_endpoint="company_report",
    )


# ── Builders ──────────────────────────────────────────────────────────────────

def build_profile(raw: dict[str, Any], ticker: str, snapshot_date: str) -> DocOutput:
    """Bangun dokumen profil perusahaan dari overview + management + ownership.

    Konten: nama lengkap perusahaan, sektor, sub-industri, tanggal listing,
    Direktur Utama (President Director), dan pemegang saham mayoritas.
    Tidak ada data keuangan atau harga di dokumen ini.

    Args:
        raw: Sections dict dari Company Report (``fundamentals`` sub-dict
            dari file JSON extract.py). Wajib punya key ``overview``.
        ticker: Simbol saham.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        DocOutput dengan ``category="profile"``, ``period=""``.

    Raises:
        ValueError: Jika ``count_tokens(content) > MAX_DOC_TOKENS``.
    """
    ov = raw.get("overview", {})
    mgmt = raw.get("management", {})
    own = raw.get("ownership", {})

    # company_name: top-level di actual API, atau di overview di mock
    company_name = (
        raw.get("company_name")
        or ov.get("company_name")
        or ticker
    )
    sector = ov.get("sector", "data tidak tersedia")
    sub = ov.get("sub_industry") or ov.get("sub_sector", "data tidak tersedia")
    listing_date = ov.get("listing_date", "data tidak tersedia")

    # Direktur Utama: cari posisi "President Director", fallback ke index 0
    executives: list[dict] = mgmt.get("key_executives", [])
    ceo = next(
        (e["name"] for e in executives if "President Director" in e.get("position", "")),
        executives[0]["name"] if executives else "data tidak tersedia",
    )

    # Pemegang saham terbesar
    # actual API: major_shareholders; mock: top_holders
    shareholders: list[dict] = (
        own.get("major_shareholders") or own.get("top_holders", [])
    )
    if shareholders:
        top = shareholders[0]
        top_name = top.get("name", "data tidak tersedia")
        # share_percentage: actual API = string "0.54942"; mock: float 54.94
        pct_raw = top.get("share_percentage") or top.get("pct")
        try:
            pct_f = float(pct_raw)
            # Jika > 1 berarti sudah dalam bentuk persen (mock), jika ≤ 1 berarti fraksi (actual)
            top_pct = (
                f"{pct_f:.2f}%".replace(".", ",")
                if pct_f > 1
                else f"{pct_f * 100:.2f}%".replace(".", ",")
            )
        except (TypeError, ValueError):
            top_pct = "data tidak tersedia"
        holder_str = f"Pemegang saham mayoritas: {top_name} {top_pct}."
    else:
        holder_str = "Pemegang saham mayoritas: data tidak tersedia."

    content = (
        f"{ticker} adalah {company_name}, sektor {sector}, "
        f"sub-industri {sub}. "
        f"Listed {listing_date}. "
        f"Direktur Utama: {ceo}. "
        f"{holder_str}"
    ).strip()

    token_count = count_tokens(content)
    if token_count > config.MAX_DOC_TOKENS:
        raise ValueError(
            f"profile {ticker}: {token_count} token melebihi batas "
            f"{config.MAX_DOC_TOKENS}"
        )

    doc_id = f"profile_{ticker}"
    return DocOutput(
        content=content,
        metadata=_make_metadata(
            "profile", ticker, "", doc_id, snapshot_date, sector=sector
        ),
    )


def build_price_snapshot(
    raw: dict[str, Any], ticker: str, snapshot_date: str
) -> DocOutput:
    """Bangun dokumen snapshot harga dari overview.

    Konten: harga penutupan terakhir, perubahan harian, perubahan 30 hari
    (jika tersedia dari ``overview.change_30d``), 52-week high/low, market cap.

    ``change_30d`` diinjeksikan ke ``overview`` oleh ``transform_ticker``
    sebelum fungsi ini dipanggil — dihitung dari ``daily_prices_30d`` di
    file JSON (hasil ``extract.fetch_daily_transaction_30d``).

    Menjawab Q006 "perubahan harga 30 hari" (expected_kb_coverage=full).

    Args:
        raw: Sections dict dari Company Report, dengan ``overview.change_30d``
            opsional (diinjeksikan oleh transform_ticker).
        ticker: Simbol saham.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        DocOutput dengan ``category="price_snapshot"``, ``period="snapshot"``.

    Raises:
        ValueError: Jika ``count_tokens(content) > MAX_DOC_TOKENS``.
    """
    ov = raw.get("overview", {})

    close_price = ov.get("last_close_price")
    close_date = ov.get("latest_close_date", snapshot_date)
    daily_change = ov.get("daily_close_change")
    market_cap = ov.get("market_cap")
    all_time = ov.get("all_time_price", {})

    # Harga penutupan
    price_str = (
        "Rp " + f"{int(close_price):,}".replace(",", ".")
        if isinstance(close_price, (int, float))
        else "data tidak tersedia"
    )

    # Perubahan harian
    if daily_change is not None:
        sign = "+" if daily_change >= 0 else ""
        daily_str = sign + f"{daily_change * 100:.2f}".replace(".", ",") + "%"
    else:
        daily_str = "data tidak tersedia"

    # 52-week high & low
    # Actual API: all_time_price.52_w_high = {date: price}
    # Mock: overview.52w_high = int langsung
    def _price_from_dict(d: dict[str, Any]) -> str:
        """Ambil harga dari format {date: price}."""
        if not d:
            return "data tidak tersedia"
        date_key, price_val = next(iter(d.items()))
        return "Rp " + f"{int(price_val):,}".replace(",", ".") + f" ({date_key})"

    w52h = all_time.get("52_w_high", {})
    w52l = all_time.get("52_w_low", {})
    high_str = _price_from_dict(w52h) if w52h else (
        "Rp " + f"{int(ov.get('52w_high', 0)):,}".replace(",", ".")
        if ov.get("52w_high")
        else "data tidak tersedia"
    )
    low_str = _price_from_dict(w52l) if w52l else (
        "Rp " + f"{int(ov.get('52w_low', 0)):,}".replace(",", ".")
        if ov.get("52w_low")
        else "data tidak tersedia"
    )

    mktcap_str = _fmt_idr(market_cap) if market_cap else "data tidak tersedia"

    # Perubahan 30 hari — diinjeksikan oleh transform_ticker dari daily_prices_30d
    change_30d = ov.get("change_30d")
    if change_30d is not None:
        sign_30 = "+" if change_30d >= 0 else ""
        change_30d_str = sign_30 + f"{change_30d * 100:.2f}".replace(".", ",") + "%"
        change_30d_part = f", perubahan 30 hari {change_30d_str}"
    else:
        change_30d_part = ""

    content = (
        f"{ticker} harga penutupan terakhir {price_str} per {close_date}, "
        f"perubahan harian {daily_str}{change_30d_part}. "
        f"52-week high {high_str}, 52-week low {low_str}. "
        f"Market cap {mktcap_str}."
    ).strip()

    token_count = count_tokens(content)
    if token_count > config.MAX_DOC_TOKENS:
        raise ValueError(
            f"price_snapshot {ticker}: {token_count} token melebihi batas "
            f"{config.MAX_DOC_TOKENS}"
        )

    sector = ov.get("sector", "")
    doc_id = f"price_snapshot_{ticker}_snapshot"
    return DocOutput(
        content=content,
        metadata=_make_metadata(
            "price_snapshot", ticker, "snapshot", doc_id, snapshot_date, sector=sector
        ),
    )


def build_valuation(
    raw: dict[str, Any], ticker: str, snapshot_date: str
) -> DocOutput:
    """Bangun dokumen valuasi dari valuation + overview + dividend.

    Konten: P/E, P/B, forward P/E, market cap, peringkat market cap,
    dividend yield, intrinsic value per tanggal snapshot.

    P/E dan P/B diambil dari ``historical_valuation`` tahun terakhir
    (actual API). Fallback ke ``pe_ratio`` dan ``pb_ratio`` langsung
    untuk kompatibilitas mock smoke test.

    Args:
        raw: Sections dict dari Company Report.
        ticker: Simbol saham.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        DocOutput dengan ``category="valuation"``, ``period="snapshot"``.

    Raises:
        ValueError: Jika ``count_tokens(content) > MAX_DOC_TOKENS``.
    """
    ov = raw.get("overview", {})
    val = raw.get("valuation", {})
    div = raw.get("dividend", {})

    # P/E dan P/B dari historical_valuation (actual API) atau langsung (mock)
    hist_val: list[dict] = val.get("historical_valuation", [])
    latest_ratio: dict = {}
    if hist_val and isinstance(hist_val, list):
        latest_ratio = max(hist_val, key=lambda x: x.get("year", 0))

    pe = latest_ratio.get("pe") or val.get("pe_ratio") or val.get("forward_pe")
    pb = latest_ratio.get("pb") or val.get("pb_ratio")
    forward_pe = val.get("forward_pe")
    intrinsic = val.get("intrinsic_value")

    market_cap = ov.get("market_cap")
    market_cap_rank = ov.get("market_cap_rank")
    div_yield = div.get("yield_ttm") or val.get("dividend_yield")
    close_date = ov.get("latest_close_date", snapshot_date)

    def _fmt_ratio(v: Any) -> str:
        """Format rasio valuasi ke 2 desimal."""
        try:
            return f"{float(v):.2f}".replace(".", ",")
        except (TypeError, ValueError):
            return "data tidak tersedia"

    pe_str = _fmt_ratio(pe)
    pb_str = _fmt_ratio(pb)
    fpe_str = _fmt_ratio(forward_pe)
    mktcap_str = _fmt_idr(market_cap) if market_cap else "data tidak tersedia"
    rank_str = f", peringkat ke-{market_cap_rank}" if market_cap_rank else ""
    dy_str = _fmt_pct(div_yield) if div_yield is not None else "data tidak tersedia"
    intrinsic_str = (
        "Rp " + f"{int(intrinsic):,}".replace(",", ".")
        if isinstance(intrinsic, (int, float))
        else "data tidak tersedia"
    )

    content = (
        f"{ticker} valuasi per {close_date}: "
        f"P/E (Price-to-Earnings) {pe_str}, P/B (Price-to-Book) {pb_str}. "
        f"Forward P/E {fpe_str}, market cap {mktcap_str}{rank_str}. "
        f"Dividend yield (imbal hasil dividen) {dy_str}, "
        f"intrinsic value {intrinsic_str}."
    ).strip()

    token_count = count_tokens(content)
    if token_count > config.MAX_DOC_TOKENS:
        raise ValueError(
            f"valuation {ticker}: {token_count} token melebihi batas "
            f"{config.MAX_DOC_TOKENS}"
        )

    sector = ov.get("sector", "")
    doc_id = f"valuation_{ticker}_snapshot"
    return DocOutput(
        content=content,
        metadata=_make_metadata(
            "valuation", ticker, "snapshot", doc_id, snapshot_date, sector=sector
        ),
    )


def build_growth(
    raw: dict[str, Any], ticker: str, snapshot_date: str
) -> DocOutput:
    """Bangun dokumen estimasi pertumbuhan dari future.

    Konten: EPS growth dan revenue growth forecast tahun terdekat dari
    ``company_growth_forecasts``, dan konsensus analis dari
    ``analyst_rating_breakdown``.

    Args:
        raw: Sections dict dari Company Report.
        ticker: Simbol saham.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        DocOutput dengan ``category="growth"``, ``period="snapshot"``.

    Raises:
        ValueError: Jika ``count_tokens(content) > MAX_DOC_TOKENS``.
    """
    fut = raw.get("future", {})

    # Ambil forecast tahun terdekat dari company_growth_forecasts
    forecasts: list[dict] = fut.get("company_growth_forecasts", [])
    nearest: dict = {}
    if forecasts and isinstance(forecasts, list):
        nearest = min(forecasts, key=lambda x: x.get("estimate_year", 9999))

    # Fallback ke mock fields (eps_growth_1y, revenue_growth_1y)
    eps_growth = nearest.get("eps_growth") or fut.get("eps_growth_1y")
    rev_growth = nearest.get("revenue_growth") or fut.get("revenue_growth_1y")
    estimate_year = nearest.get("estimate_year", "")

    eps_str = (
        f"{float(eps_growth) * 100:.2f}".replace(".", ",") + "%"
        if eps_growth is not None
        else "data tidak tersedia"
    )
    rev_str = (
        f"{float(rev_growth) * 100:.2f}".replace(".", ",") + "%"
        if rev_growth is not None
        else "data tidak tersedia"
    )
    year_str = f" tahun {estimate_year}" if estimate_year else ""

    # API mengembalikan null (None) jika tidak ada data analis — pakai {} sebagai fallback
    analyst: dict = fut.get("analyst_rating_breakdown") or {}
    n_analyst = analyst.get("n_analyst", 0)
    strong_buy = analyst.get("strong_buy", 0)
    buy = analyst.get("buy", 0)
    hold = analyst.get("hold", 0)
    sell = analyst.get("sell", 0)
    updated = str(analyst.get("updated_on", "data tidak tersedia"))
    # Potong ke tanggal saja jika ada timestamp
    if len(updated) > 10 and updated[10] == " ":
        updated = updated[:10]

    content = (
        f"{ticker} estimasi pertumbuhan{year_str}: "
        f"EPS growth (pertumbuhan laba per saham) {eps_str}, "
        f"revenue growth {rev_str}. "
        f"Konsensus {n_analyst} analis: {strong_buy} strong buy, "
        f"{buy} buy, {hold} hold, {sell} sell. "
        f"Diperbarui {updated}."
    ).strip()

    token_count = count_tokens(content)
    if token_count > config.MAX_DOC_TOKENS:
        raise ValueError(
            f"growth {ticker}: {token_count} token melebihi batas "
            f"{config.MAX_DOC_TOKENS}"
        )

    sector = raw.get("overview", {}).get("sector", "")
    doc_id = f"growth_{ticker}_snapshot"
    return DocOutput(
        content=content,
        metadata=_make_metadata(
            "growth", ticker, "snapshot", doc_id, snapshot_date, sector=sector
        ),
    )


def build_financials_annual(
    raw: dict[str, Any], ticker: str, snapshot_date: str
) -> list[DocOutput]:
    """Bangun dokumen keuangan tahunan, 1 DocOutput per tahun.

    Mengambil ``ANNUAL_HISTORY_YEARS`` (5) tahun terakhir dari
    ``historical_financials``. Setiap dokumen berisi: revenue, laba bersih,
    EPS, ROE, ROA, net margin, total aset, ekuitas.

    EPS diambil dari ``historical_eps`` (dict year → {eps}), atau dihitung
    dari earnings / outstanding_shares jika tidak tersedia.

    Dual-field support: ``historical_financials`` (actual API) atau
    ``annual_data`` (smoke test mock) — whichever is non-empty.

    Args:
        raw: Sections dict dari Company Report.
        ticker: Simbol saham.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        List DocOutput dengan ``category="financials_annual"``,
        ``period=str(year)`` (mis. ``"2024"``). List kosong jika tidak ada
        data.

    Raises:
        ValueError: Jika content salah satu tahun melebihi MAX_DOC_TOKENS.
    """
    fin: dict = raw.get("financials", {})

    # Dual-field: actual API field atau mock field
    annual_list: list[dict] = (
        fin.get("historical_financials") or fin.get("annual_data", [])
    )
    ratio_list: list[dict] = fin.get("historical_financial_ratio", [])
    eps_raw_dict: dict = fin.get("historical_eps", {})

    # Build ratio lookup: {year_str: profitability_dict}
    ratio_map: dict[str, dict] = {
        str(r.get("year", "")): r.get("profitability", {})
        for r in ratio_list
        if r.get("year")
    }

    # Build EPS lookup: handle {year: {eps: val}} (actual) or {year: val} (dict of float)
    eps_map: dict[str, float] = {}
    for y_str, v in eps_raw_dict.items():
        if isinstance(v, dict):
            eps_map[y_str] = float(v.get("eps", 0) or 0)
        elif isinstance(v, (int, float)):
            eps_map[y_str] = float(v)

    # Sort ascending by year, ambil ANNUAL_HISTORY_YEARS terakhir
    try:
        sorted_annual = sorted(annual_list, key=lambda x: x.get("year", 0))
    except (TypeError, AttributeError):
        return []

    selected = sorted_annual[-config.ANNUAL_HISTORY_YEARS :]
    sector = raw.get("overview", {}).get("sector", "")
    docs: list[DocOutput] = []

    for entry in selected:
        year = entry.get("year")
        if not year:
            continue
        year_str = str(year)

        revenue = entry.get("revenue")
        # actual: earnings; mock: net_income
        earnings = entry.get("earnings") or entry.get("net_income")
        total_assets = entry.get("total_assets")
        total_equity = entry.get("total_equity")
        outstanding = entry.get("outstanding_shares")

        # EPS: from historical_eps, atau fallback ke entry.eps (mock),
        # atau hitung dari earnings/outstanding_shares
        eps_val = (
            eps_map.get(year_str)
            or entry.get("eps")
            or (
                earnings / outstanding
                if isinstance(earnings, (int, float))
                and isinstance(outstanding, (int, float))
                and outstanding
                else None
            )
        )
        eps_str = (
            "Rp " + f"{eps_val:.0f}".replace(".", ",")
            if isinstance(eps_val, (int, float))
            else "data tidak tersedia"
        )

        # Rasio dari historical_financial_ratio, fallback ke entry langsung (mock)
        prof = ratio_map.get(year_str, {})
        roe = prof.get("roe") or entry.get("roe")
        roa = prof.get("roa") or entry.get("roa")
        net_margin = (
            prof.get("net_profit_margin")
            or entry.get("net_margin")
            or entry.get("net_profit_margin")
        )

        content = (
            f"{ticker} tahun {year}: "
            f"revenue {_fmt_idr(revenue)}, laba bersih {_fmt_idr(earnings)}, "
            f"EPS (Earnings Per Share) {eps_str}. "
            f"ROE (Return on Equity) {_fmt_pct(roe)}, "
            f"ROA (Return on Assets) {_fmt_pct(roa)}, "
            f"net margin {_fmt_pct(net_margin)}. "
            f"Total aset {_fmt_idr(total_assets)}, "
            f"ekuitas {_fmt_idr(total_equity)}."
        ).strip()

        token_count = count_tokens(content)
        if token_count > config.MAX_DOC_TOKENS:
            raise ValueError(
                f"financials_annual {ticker} {year}: {token_count} token "
                f"melebihi batas {config.MAX_DOC_TOKENS}"
            )

        doc_id = f"financials_annual_{ticker}_{year_str}"
        docs.append(
            DocOutput(
                content=content,
                metadata=_make_metadata(
                    "financials_annual",
                    ticker,
                    year_str,
                    doc_id,
                    snapshot_date,
                    sector=sector,
                ),
            )
        )

    return docs


def build_financials_quarterly(
    raw: dict[str, Any], ticker: str, snapshot_date: str
) -> list[DocOutput]:
    """Bangun dokumen keuangan kuartalan, 1 DocOutput per kuartal.

    Mengambil ``QUARTERLY_HISTORY_QUARTERS`` (6) kuartal terakhir dari
    ``historical_financials_quarterly`` (dict keyed by "Q1-2026").
    Setiap dokumen berisi: revenue, laba bersih, EBITDA, total aset,
    ekuitas. Net interest income dan non-interest income disertakan jika
    tersedia (sektor perbankan).

    Dual-field: ``historical_financials_quarterly`` (actual API, dict)
    atau ``quarterly_data`` (mock, list dengan field ``quarter``).

    Args:
        raw: Sections dict dari Company Report.
        ticker: Simbol saham.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        List DocOutput dengan ``category="financials_quarterly"``,
        ``period="Q{n}-{YYYY}"`` (mis. ``"Q1-2026"``). List kosong jika
        tidak ada data.

    Raises:
        ValueError: Jika content salah satu kuartal melebihi MAX_DOC_TOKENS.
    """
    fin: dict = raw.get("financials", {})

    # Dual-field: dict (actual) atau list (mock)
    quarterly_raw = fin.get("historical_financials_quarterly") or fin.get(
        "quarterly_data"
    )

    if isinstance(quarterly_raw, list):
        # Mock format: [{"quarter": "Q1-2026", "revenue": ..., ...}]
        quarterly_dict: dict[str, dict] = {
            item["quarter"]: item
            for item in quarterly_raw
            if isinstance(item, dict) and "quarter" in item
        }
    elif isinstance(quarterly_raw, dict):
        quarterly_dict = quarterly_raw
    else:
        return []

    # Sort kuartal kronologis (tahun ASC, Q ASC)
    def _q_sort_key(label: str) -> tuple[int, int]:
        parts = label.split("-")
        try:
            return (int(parts[1]), int(parts[0][1:]))
        except (IndexError, ValueError):
            return (0, 0)

    sorted_quarters = sorted(quarterly_dict.keys(), key=_q_sort_key)
    selected = sorted_quarters[-config.QUARTERLY_HISTORY_QUARTERS :]

    sector = raw.get("overview", {}).get("sector", "")
    docs: list[DocOutput] = []

    for q_label in selected:
        qdata = quarterly_dict[q_label]

        revenue = qdata.get("revenue")
        # actual: earnings; mock: net_income
        earnings = qdata.get("earnings") or qdata.get("net_income")
        ebitda = qdata.get("ebitda")
        total_assets = qdata.get("total_assets")
        total_equity = qdata.get("total_equity")
        net_interest = qdata.get("net_interest_income")
        non_interest = qdata.get("non_interest_income")

        content = (
            f"{ticker} {q_label}: "
            f"revenue {_fmt_idr(revenue)}, laba bersih {_fmt_idr(earnings)}, "
            f"EBITDA {_fmt_idr(ebitda)}. "
            f"Total aset {_fmt_idr(total_assets)}, "
            f"ekuitas {_fmt_idr(total_equity)}."
        )

        # Tambahkan kolom perbankan hanya jika tersedia dan masih dalam budget
        if net_interest and non_interest:
            banking_line = (
                f" Net interest income {_fmt_idr(net_interest)}, "
                f"non-interest income {_fmt_idr(non_interest)}."
            )
            candidate = (content + banking_line).strip()
            if count_tokens(candidate) <= config.MAX_DOC_TOKENS:
                content = candidate
            else:
                content = content.strip()
        else:
            content = content.strip()

        token_count = count_tokens(content)
        if token_count > config.MAX_DOC_TOKENS:
            raise ValueError(
                f"financials_quarterly {ticker} {q_label}: {token_count} token "
                f"melebihi batas {config.MAX_DOC_TOKENS}"
            )

        doc_id = f"financials_quarterly_{ticker}_{q_label}"
        docs.append(
            DocOutput(
                content=content,
                metadata=_make_metadata(
                    "financials_quarterly",
                    ticker,
                    q_label,
                    doc_id,
                    snapshot_date,
                    sector=sector,
                ),
            )
        )

    return docs


def build_dividend(
    raw: dict[str, Any], ticker: str, snapshot_date: str
) -> DocOutput:
    """Bangun dokumen dividen dari dividend section.

    Konten: dividend yield TTM, payout ratio, dividend TTM per saham,
    tanggal ex-dividend terakhir, dan historis 3 tahun terakhir.

    Dual-field: ``historical_dividends`` (actual API, dict year → total)
    atau ``history`` (mock, list [{year, amount}]).

    Args:
        raw: Sections dict dari Company Report.
        ticker: Simbol saham.
        snapshot_date: Tanggal ETL run ISO 8601.

    Returns:
        DocOutput dengan ``category="dividend"``, ``period=""``.

    Raises:
        ValueError: Jika ``count_tokens(content) > MAX_DOC_TOKENS``.
    """
    div: dict = raw.get("dividend", {})

    yield_ttm = div.get("yield_ttm")
    payout = div.get("payout_ratio")
    dividend_ttm = div.get("dividend_ttm")
    last_ex = div.get("last_ex_dividend_date", "data tidak tersedia")

    # Format TTM dividend per saham
    div_ttm_str = (
        "Rp " + f"{dividend_ttm:,.0f}".replace(",", ".")
        if isinstance(dividend_ttm, (int, float))
        else "data tidak tersedia"
    )

    # Historis dividen — last 3 tahun
    hist_div = div.get("historical_dividends") or div.get("history")
    history_parts: list[str] = []

    if isinstance(hist_div, dict):
        # Actual API: {year_str: {total_dividend, breakdown, ...}}
        for y in sorted(hist_div.keys(), reverse=True)[:3]:
            total = hist_div[y].get("total_dividend")
            if total is not None:
                history_parts.append(
                    f"{y} Rp " + f"{total:,.0f}".replace(",", ".")
                )
    elif isinstance(hist_div, list):
        # Mock: [{year: 2024, amount: 270}, ...]
        sorted_hist = sorted(hist_div, key=lambda x: x.get("year", 0), reverse=True)
        for item in sorted_hist[:3]:
            y = item.get("year")
            amt = item.get("amount")
            if y is not None and amt is not None:
                history_parts.append(
                    f"{y} Rp " + f"{amt:,.0f}".replace(",", ".")
                )

    history_str = (
        " Historis: " + ", ".join(history_parts) + " per saham."
        if history_parts
        else ""
    )

    content = (
        f"{ticker} dividen: "
        f"yield TTM (Trailing Twelve Months) {_fmt_pct(yield_ttm)}, "
        f"payout ratio {_fmt_pct(payout)}, "
        f"dividend TTM {div_ttm_str} per saham. "
        f"Ex-dividend date terakhir {last_ex}."
        f"{history_str}"
    ).strip()

    token_count = count_tokens(content)
    if token_count > config.MAX_DOC_TOKENS:
        raise ValueError(
            f"dividend {ticker}: {token_count} token melebihi batas "
            f"{config.MAX_DOC_TOKENS}"
        )

    sector = raw.get("overview", {}).get("sector", "")
    doc_id = f"dividend_{ticker}"
    return DocOutput(
        content=content,
        metadata=_make_metadata(
            "dividend", ticker, "", doc_id, snapshot_date, sector=sector
        ),
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def transform_ticker(ticker: str, snapshot_date: str) -> list[DocOutput]:
    """Baca JSON satu ticker dan jalankan semua 7 builder.

    Membaca ``{RAW_DATA_DIR}/{ticker}_{snapshot_date}.json``, mengekstrak
    sub-dict ``fundamentals``, lalu memanggil semua builder. Dokumen
    aggregate (ranking, sektor) bukan scope fungsi ini — lihat aggregate.py.

    Args:
        ticker: Simbol saham IDX30.
        snapshot_date: Tanggal snapshot ISO 8601, mis. ``"2026-05-28"``.

    Returns:
        Flat list DocOutput semua kategori, 13–15 dokumen per ticker
        (1 profile + 1 price_snapshot + 1 valuation + 1 growth +
        ≤5 financials_annual + ≤6 financials_quarterly + 1 dividend).

    Raises:
        FileNotFoundError: Jika file JSON tidak ditemukan.
        KeyError: Jika key ``"fundamentals"`` tidak ada di JSON.
        ValueError: Jika salah satu builder menghasilkan doc > MAX_DOC_TOKENS.
    """
    json_path: Path = config.RAW_DATA_DIR / f"{ticker}_{snapshot_date}.json"
    with json_path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    raw: dict[str, Any] = data["fundamentals"]

    # Injeksi change_30d ke overview dari daily_prices_30d (Opsi B).
    # Dihitung di sini agar build_price_snapshot tidak perlu signature baru.
    daily_prices: list[dict[str, Any]] = data.get("daily_prices_30d", [])
    change_30d: float | None = _compute_change_30d(daily_prices)
    if change_30d is not None:
        raw.setdefault("overview", {})["change_30d"] = change_30d

    docs: list[DocOutput] = []
    docs.append(build_profile(raw, ticker, snapshot_date))
    docs.append(build_price_snapshot(raw, ticker, snapshot_date))
    docs.append(build_valuation(raw, ticker, snapshot_date))
    docs.append(build_growth(raw, ticker, snapshot_date))
    docs.extend(build_financials_annual(raw, ticker, snapshot_date))
    docs.extend(build_financials_quarterly(raw, ticker, snapshot_date))
    docs.append(build_dividend(raw, ticker, snapshot_date))
    return docs


def transform_all(snapshot_date: str) -> dict[str, list[DocOutput]]:
    """Jalankan transform untuk semua 30 ticker IDX30.

    Ticker yang file JSON-nya tidak ditemukan (extract belum dijalankan)
    di-skip tanpa error. Exception lain (mis. ``ValueError`` token overflow,
    ``KeyError`` struktur JSON berubah) di-propagate ke caller agar
    tidak tersembunyi.

    Args:
        snapshot_date: Tanggal snapshot ISO 8601.

    Returns:
        Dict ``{ticker: list[DocOutput]}``, hanya berisi ticker yang
        berhasil di-transform.
    """
    results: dict[str, list[DocOutput]] = {}
    for ticker in config.IDX30_TICKERS:
        try:
            results[ticker] = transform_ticker(ticker, snapshot_date)
        except FileNotFoundError:
            # File belum di-extract — expected, bukan error
            pass
    return results
