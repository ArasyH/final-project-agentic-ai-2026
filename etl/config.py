import requests
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("SECTORS_API_KEY")
HEADERS = {
    "Authorization": API_KEY  # Pastikan key 'Authorization' tertulis persis seperti ini
}

url = "https://api.sectors.app/v2/daily/AADI/?start=2025-01-01&end=2026-04-14"
response = requests.get(url, headers=HEADERS)

print(response.status_code) # Jika berhasil, ini akan mencetak 200, bukan 401