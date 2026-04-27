# Extract data via sectors.app untuk semua 30 emiten IDX 30 dan menyimpan ke file JSON lokal
# TASK 1
import requests
import json
import os
import time
from datetime import date
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


def fetch_stock_price(ticker: str) -> dict:
    """Ambil data harga historis 1 tahun terakhir."""
    url = f"{BASE_URL}/daily/{ticker}/?start=2025-01-01&end={date.today()}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()

def fetch_company_report(ticker: str) -> dict:
    """Ambil laporan fundamental perusahaan."""
    url = f"{BASE_URL}/company/report/{ticker}/"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()

def extract_all():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    today = date.today().isoformat()
    
    for ticker in IDX30_TICKERS:
        print(f"Extracting {ticker}...")
        try:
            price_data    = fetch_stock_price(ticker)
            company_data  = fetch_company_report(ticker)
            
            output = {
                "ticker":       ticker,
                "extracted_at": today,
                "price_history": price_data,
                "fundamentals":  company_data,
            }
            
            filepath = os.path.join(RAW_DATA_DIR, f"{ticker}_{today}.json")
            with open(filepath, "w") as f:
                json.dump(output, f, indent=2)
                
            time.sleep(10)  # Hindari rate limit
            
        except Exception as e:
            print(f"ERROR {ticker}: {e}")

if __name__ == "__main__":
    extract_all()
    print("Extract selesai.")