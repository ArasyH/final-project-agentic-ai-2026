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

def calculate_eda(price_df: pd.DataFrame) -> dict:
    """Hitung metrik EDA dari dataframe harga harian."""
    price_df["close"] = pd.to_numeric(price_df["close"])
    price_df["date"]  = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values("date")
    
    latest_price  = price_df["close"].iloc[-1]
    price_30d_ago = price_df["close"].iloc[-30] if len(price_df) >= 30 else price_df["close"].iloc[0]
    
    return {
        "latest_price":   latest_price,
        "change_30d_pct": round((latest_price - price_30d_ago) / price_30d_ago * 100, 2),
        "volabtility_30d": round(price_df["close"].pct_change().tail(30).std() * 100, 2),
        "ma_7":           round(price_df["close"].tail(7).mean(), 0),
        "ma_30":          round(price_df["close"].tail(30).mean(), 0),
        "high_52w":       price_df["close"].tail(252).max(),
        "low_52w":        price_df["close"].tail(252).min(),
    }

# transform.py — versi diperbaiki

def build_narrative(ticker: str, eda: dict, fundamentals: dict, quarterly: list) -> str:
    trend = "naik" if eda["change_30d_pct"] > 0 else "turun"
    
    # Ambil lebih banyak field dari fundamentals (Company Report kaya data)
    f = fundamentals
    
    # Susun ringkasan quarterly terbaru (maks 4 kuartal)
    quarterly_text = ""
    if quarterly:
        for q in quarterly[:4]:
            qdate = q.get("report_date") or q.get("period", "N/A")
            rev   = q.get("revenue", q.get("total_revenue", "N/A"))
            earn  = q.get("earnings", q.get("net_income", "N/A"))
            quarterly_text += f"\n  {qdate}: Revenue {rev}, Laba {earn}"

    return f"""
Saham {ticker} — Data per {date.today().isoformat()}

== Profil Perusahaan ==
Nama: {f.get('company_name', ticker)}
Sektor: {f.get('sector', 'N/A')} | Sub-sektor: {f.get('sub_sector', 'N/A')}
Industri: {f.get('industry', 'N/A')}
Market cap: {f.get('market_cap', 'N/A')}
Deskripsi: {str(f.get('overview', {}).get('description', f.get('business_description', '')))[:400]}

== Valuasi ==
P/E TTM: {f.get('pe_ttm', f.get('pe_ratio', 'N/A'))}
P/B MRQ: {f.get('pb_mrq', 'N/A')}
ROE TTM: {f.get('roe_ttm', 'N/A')}
ROA TTM: {f.get('roa_ttm', 'N/A')}
Dividen yield: {f.get('yield_ttm', 'N/A')}

== Pergerakan Harga ==
Harga terakhir: Rp {eda['latest_price']:,.0f}
Pergerakan 30 hari: {trend} {abs(eda['change_30d_pct'])}%
Volatilitas 30 hari: {eda['volatility_30d']}%
MA 7 hari: Rp {eda['ma_7']:,.0f} | MA 30 hari: Rp {eda['ma_30']:,.0f}
52-week high: Rp {eda['high_52w']:,.0f} | 52-week low: Rp {eda['low_52w']:,.0f}

== Keuangan Kuartalan (4 periode terakhir) =={quarterly_text if quarterly_text else ' Data tidak tersedia'}
""".strip()

def transform_all():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    today = date.today().isoformat()

    for filename in os.listdir(RAW_DATA_DIR):
        if not filename.endswith(f"{today}.json"):
            continue
        ticker = filename.split("_")[0]
        with open(os.path.join(RAW_DATA_DIR, filename)) as f:
            data = json.load(f)

        price_df  = pd.DataFrame(data["price_history"])
        if price_df.empty:
            print(f"  SKIP {ticker}: price_history kosong")
            continue
        
        eda       = calculate_eda(price_df)
        # Kirim quarterly juga ke build_narrative
        narrative = build_narrative(ticker, eda, data["fundamentals"], data.get("quarterly", []))

        out_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_{today}.txt")
        with open(out_path, "w") as f:
            f.write(narrative)
        print(f"Transformed {ticker}")