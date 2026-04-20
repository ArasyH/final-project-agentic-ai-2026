# test_chat.py
import httpx
import json

BASE_URL = "http://localhost:8000"

def ask(question: str, session_id: str = "test-session-001"):
    resp = httpx.post(
        f"{BASE_URL}/chat",
        json={"question": question, "session_id": session_id},
        timeout=30,
    )
    print(resp.text)
    resp.raise_for_status()
    data = resp.json()

    print(f"\n{'='*55}")
    print(f"Q  : {question}")
    print(f"A  : {data['answer']}")
    print(f"Src: {data['source'].upper()} | Score: {data['similarity']} | {data['latency_ms']}ms")
    print(f"{'='*55}")
    return data

if __name__ == "__main__":
    # Uji 1: pertanyaan baru → harus AGENT
    ask("Berapa harga saham BBCA saat ini?")

    # Uji 2: pertanyaan sama persis → harus CACHE
    ask("Berapa harga saham BBCA saat ini?")

    # Uji 3: variasi semantik → harus CACHE (inilah inti penelitianmu)
    ask("Harga Bank Central Asia sekarang berapa?")
    ask("Cek harga BCA dong")
    ask("BBCA closing price hari ini?")

    # Uji 4: pertanyaan berbeda topik → harus AGENT lagi
    ask("Bagaimana tren volatilitas saham TLKM bulan ini?")