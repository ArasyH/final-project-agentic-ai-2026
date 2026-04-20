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
        "volatility_30d": round(price_df["close"].pct_change().tail(30).std() * 100, 2),
        "ma_7":           round(price_df["close"].tail(7).mean(), 0),
        "ma_30":          round(price_df["close"].tail(30).mean(), 0),
        "high_52w":       price_df["close"].tail(252).max(),
        "low_52w":        price_df["close"].tail(252).min(),
    }

def build_narrative(ticker: str, eda: dict, fundamentals: dict) -> str:
    """Bangun narasi teks yang akan di-embed ke vector database."""
    trend = "naik" if eda["change_30d_pct"] > 0 else "turun"
    
    return f"""
Saham {ticker} — Data per {date.today().isoformat()}

Harga terakhir: Rp {eda['latest_price']:,.0f}
Pergerakan 30 hari: {trend} {abs(eda['change_30d_pct'])}%
Volatilitas 30 hari: {eda['volatility_30d']}%
Moving average 7 hari: Rp {eda['ma_7']:,.0f}
Moving average 30 hari: Rp {eda['ma_30']:,.0f}
Tertinggi 52 minggu: Rp {eda['high_52w']:,.0f}
Terendah 52 minggu: Rp {eda['low_52w']:,.0f}

Fundamental:
Nama perusahaan: {fundamentals.get('company_name', ticker)}
Sektor: {fundamentals.get('sector', 'N/A')}
Market cap: {fundamentals.get('market_cap', 'N/A')}
P/E ratio: {fundamentals.get('pe_ratio', 'N/A')}
""".strip()

def transform_all():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    today = date.today().isoformat()
    
    for filename in os.listdir(RAW_DATA_DIR):
        if not filename.endswith(f"{today}.json"):
            continue
        
        ticker = filename.split("_")[0]
        filepath = os.path.join(RAW_DATA_DIR, filename)
        
        with open(filepath) as f:
            data = json.load(f)
        
        price_df     = pd.DataFrame(data["price_history"])
        eda          = calculate_eda(price_df)
        narrative    = build_narrative(ticker, eda, data["fundamentals"])
        
        out_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_{today}.txt")
        with open(out_path, "w") as f:
            f.write(narrative)
        
        print(f"Transformed {ticker}")

if __name__ == "__main__":
    transform_all()
    print("Transform selesai.")