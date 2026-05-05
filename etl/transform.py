# Membaca data mentah json dan menghasilkan narasi teks per emiten yang akan menjadi knowledge base
# TASK 2
import json
import os
import pandas as pd
from datetime import date
from dotenv import load_dotenv

load_dotenv()
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR")

# transform.py — versi diperbaiki
import json
import os
import math
import pandas as pd
from datetime import date
from dotenv import load_dotenv

load_dotenv()
RAW_DATA_DIR       = os.getenv("RAW_DATA_DIR")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR")

def _safe(val, fmt=None, fallback="N/A"):
    """Konversi nilai ke string dengan aman — handle NaN, None, dan format angka."""
    if val is None:
        return fallback
    if isinstance(val, float) and math.isnan(val):
        return fallback
    if fmt == "rp":
        try:
            return f"Rp {float(val):,.0f}"
        except (TypeError, ValueError):
            return fallback
    if fmt == "pct":
        try:
            return f"{float(val):.2f}%"
        except (TypeError, ValueError):
            return fallback
    return str(val) if val != "" else fallback


def calculate_eda(price_df: pd.DataFrame) -> dict:
    """Hitung metrik EDA. Return dict dengan fallback N/A jika data tidak cukup."""
    # Guard: minimal butuh 2 baris untuk pct_change
    if price_df.empty or len(price_df) < 2:
        return {
            "latest_price":   None,
            "change_30d_pct": None,
            "volatility_30d": None,
            "ma_7":           None,
            "ma_30":          None,
            "high_52w":       None,
            "low_52w":        None,
        }

    price_df = price_df.copy()
    price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
    price_df["date"]  = pd.to_datetime(price_df["date"], errors="coerce")
    price_df = price_df.dropna(subset=["close", "date"]).sort_values("date")

    if price_df.empty:
        return {k: None for k in
                ["latest_price","change_30d_pct","volatility_30d","ma_7","ma_30","high_52w","low_52w"]}

    latest_price  = price_df["close"].iloc[-1]
    # Gunakan min(30, len-1) agar tidak IndexError kalau data < 30 hari
    lookback      = min(30, len(price_df) - 1)
    price_past    = price_df["close"].iloc[-lookback - 1]

    change_30d    = round((latest_price - price_past) / price_past * 100, 2) \
                    if price_past != 0 else None

    vol_series    = price_df["close"].pct_change().tail(30)
    volatility    = round(float(vol_series.std()) * 100, 2) \
                    if vol_series.std() and not math.isnan(vol_series.std()) else None

    return {
        "latest_price":   latest_price,
        "change_30d_pct": change_30d,
        "volatility_30d": volatility,
        "ma_7":           round(float(price_df["close"].tail(7).mean()), 0),
        "ma_30":          round(float(price_df["close"].tail(30).mean()), 0),
        "high_52w":       float(price_df["close"].tail(252).max()),
        "low_52w":        float(price_df["close"].tail(252).min()),
    }


def _get_sections(data: dict) -> dict:
    """Return dict sections + tambahkan company_name & symbol dari top-level."""
    if "sections" in data:
        sections = data["sections"]
    else:
        sections = data.get("fundamentals", {})

    # Sisipkan company_name & symbol ke dalam sections
    # agar build_narrative bisa akses lewat sections["company_name"]
    sections["company_name"] = (
        sections.get("company_name")           # sudah ada di top-level sections
        or data.get("fundamentals", {}).get("company_name")
        or sections.get("symbol")
        or data.get("ticker", "")
    )
    sections["symbol"] = sections.get("symbol") or data.get("ticker", "")
    return sections


def build_narrative(ticker: str, eda: dict, sections: dict, quarterly: list) -> str:
    ov  = sections.get("overview",   {})
    val = sections.get("valuation",  {})
    fin = sections.get("financials", {})
    div = sections.get("dividend",   {})
    fut = sections.get("future",     {})

    # ── company_name dari top-level sections (bukan dari overview) ──
    company_name = sections.get("company_name") or ticker

    # ── sector/sub_sector ada di overview ──
    sector     = ov.get("sector",     val.get("sector",     "N/A"))
    sub_sector = ov.get("sub_sector", "N/A")
    industry   = ov.get("industry",   ov.get("sub_industry", "N/A"))
    employees  = ov.get("employee_num", "N/A")
    listing    = ov.get("listing_date", "N/A")
    website    = ov.get("website", "N/A")

    change = eda.get("change_30d_pct")
    import math
    if change is not None and not (isinstance(change, float) and math.isnan(change)):
        change_str = f"{'naik' if change > 0 else 'turun'} {abs(change):.2f}%"
    else:
        change_str = "N/A"

    quarterly_lines = ""
    for q in (quarterly or [])[:4]:
        period = q.get("period") or q.get("report_date", "")
        rev    = q.get("revenue", q.get("total_revenue", "N/A"))
        earn   = q.get("earnings", q.get("net_income", "N/A"))
        if period:
            quarterly_lines += f"\n  {period}: Revenue={rev}, Laba bersih={earn}"

    return f"""
Saham {ticker} — {date.today().isoformat()}

== Profil ==
Nama: {company_name}
Sektor: {sector} | Sub-sektor: {sub_sector}
Industri: {industry}
Karyawan: {employees} | Listing: {listing}
Website: {website}

== Harga & Teknikal ==
Harga terakhir: {_safe(eda.get('latest_price'), 'rp')}
Pergerakan 30 hari: {change_str}
Volatilitas 30 hari: {_safe(eda.get('volatility_30d'), 'pct')}
MA 7 hari: {_safe(eda.get('ma_7'), 'rp')} | MA 30 hari: {_safe(eda.get('ma_30'), 'rp')}
52-week high: {_safe(eda.get('high_52w'), 'rp')} | 52-week low: {_safe(eda.get('low_52w'), 'rp')}

== Valuasi ==
P/E TTM: {val.get('pe_ttm', 'N/A')} | P/B MRQ: {val.get('pb_mrq', 'N/A')}
ROE TTM: {val.get('roe_ttm', 'N/A')} | ROA TTM: {val.get('roa_ttm', 'N/A')}
Market cap: {ov.get('market_cap', val.get('market_cap', 'N/A'))}

== Keuangan Tahunan ==
Revenue: {fin.get('revenue', 'N/A')} | Laba: {fin.get('earnings', 'N/A')}
EBITDA: {fin.get('ebitda', 'N/A')} | Net margin: {fin.get('net_profit_margin', 'N/A')}

== Proyeksi Analis ==
EPS growth forecast: {fut.get('forecast_eps_growth', 'N/A')}
Revenue growth forecast: {fut.get('forecast_revenue_growth', 'N/A')}

== Dividen ==
Yield TTM: {div.get('yield_ttm', 'N/A')} | Payout ratio: {div.get('payout_ratio', 'N/A')}
Ex-div terakhir: {div.get('last_ex_dividend_date', 'N/A')}

== Keuangan Kuartalan =={quarterly_lines if quarterly_lines else ' Data tidak tersedia'}
""".strip()


def transform_all():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    today = date.today().isoformat()
    ok, skip, err = 0, 0, 0

    for filename in sorted(os.listdir(RAW_DATA_DIR)):
        if not filename.endswith(f"{today}.json") or "backup" in filename:
            skip += 1
            continue

        ticker   = filename.split("_")[0]
        filepath = os.path.join(RAW_DATA_DIR, filename)

        try:
            with open(filepath) as f:
                data = json.load(f)

            price_list = data.get("price_history", [])
            if not price_list:
                print(f"  SKIP {ticker}: price_history kosong")
                skip += 1
                continue

            price_df  = pd.DataFrame(price_list)
            eda       = calculate_eda(price_df)
            sections  = _get_sections(data)
            quarterly = data.get("quarterly", [])
            narrative = build_narrative(ticker, eda, sections, quarterly)

            out_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_{today}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(narrative)

            print(f"  ✓ {ticker}")
            ok += 1

        except Exception as e:
            print(f"  ✗ {ticker}: {e}")
            err += 1

    print(f"\nTransform selesai: {ok} OK, {skip} skip, {err} error")


if __name__ == "__main__":
    transform_all()