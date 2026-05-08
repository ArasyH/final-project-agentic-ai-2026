# Extract data via sectors.app untuk semua 30 emiten IDX 30 dan menyimpan ke file JSON lokal
# TASK 1
import requests
import json
import os
import time
from datetime import date, timedelta
from dotenv import load_dotenv

# kinerja saham per tanggal 13 april 2026
IDX30_TICKERS = [
    "AADI", "ADRO", "AMRT", "ANTM", "ASII",
    "BBCA", "BBNI", "BBRI", "BMRI", "BRPT",
    "BUMI", "CPIN", "EMTK", "GOTO", "ICBP",
    "INCO", "INDF", "INKP", "ISAT", "JPFA",
    "KLBF", "MBMA", "MDKA", "MEDC", "PGAS",
    "PGEO", "PTBA", "TLKM", "UNTR", "UNVR"
]

load_dotenv()
SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
HEADERS = {"Authorization": SECTORS_API_KEY}
BASE_URL = "https://api.sectors.app/v2"

def _get(url: str) -> dict:
    resp = requests.get (url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_stock_price(ticker: str, start="2025-01-01") -> list:
    """Ambil data harga historis dengan chunking per 90 hari."""
    all_records, chunk = [], 90
    current_start = date.fromisoformat(start)
    end_date = date.today()

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk), end_date)
        url = f"{BASE_URL}/daily/{ticker}/?start={current_start}&end={current_end}"
        try:
            data = _get(url)
            #API bisa return list atau dict
            records = data if isinstance(data,list) else data.get("data", [])
            all_records.extend(records)
            print(f"[{ticker}] {current_start} -> {current_end}: {len(records)} records")
        except Exception as e:
            print()
        current_start = current_end + timedelta(days=1)
        time.sleep(5) # lebih pendek, per chunk 
        
    return all_records 

def fetch_company_report(ticker: str) -> dict:
    return _get(f"{BASE_URL}/company/report/{ticker}/")

# def fetch_quarterly_financials(ticker: str) -> list:
#     """Ambil data keuangan kuartalan. v2 endpoint."""
#     # Coba v2 dulu, fallback ke v1 jika gagal
#     for url in [
#         f"{BASE_URL}/financials/quarterly/{ticker}/",
#         f"https://api.sectors.app/v1/financials/quarterly/{ticker}/?approx=true",
#     ]:
#         try:
#             data = _get(url)
#             return data if isinstance(data, list) else data.get("data", [])
#         except Exception:
#             continue
#     return []

# def fetch_quarterly_dates(ticker: str) -> list:
#     """Ambil tanggal laporan kuartalan yang tersedia."""
#     for url in [
#         f"{BASE_URL}/company/{ticker}/quarterly-financials/dates/",
#         f"https://api.sectors.app/v1/company/get_quarterly_financial_dates/{ticker}/",
#     ]:
#         try:
#             data = _get(url)
#             return data if isinstance(data, list) else data.get("dates", [])
#         except Exception:
#             continue
#     return []

# ── extract_all: simpan semua 4 dataset ────────────────────────────────────
def extract_all():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    today = date.today().isoformat()

    for ticker in IDX30_TICKERS:
        print(f"\nExtracting {ticker}...")
        try:
            output = {
                "ticker":        ticker,
                "extracted_at":  today,
                "price_history": fetch_stock_price(ticker),    # list panjang
                "fundamentals":  fetch_company_report(ticker), # dict 
            }
            filepath = os.path.join(RAW_DATA_DIR, f"{ticker}_{today}.json")
            with open(filepath, "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"  [{ticker}] Saved → {filepath}")
            time.sleep(5)  # 5 detik antar ticker
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

if __name__ == "__main__":
    extract_all()
    print("Extract selesai.")