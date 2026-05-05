# etl/check_db.py — jalankan: python check_db.py
import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ── Cek 1: apakah folder chroma_db lokal ada? ──
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
local_exists   = os.path.isdir(CHROMA_DB_PATH)
sqlite_exists  = os.path.isfile(os.path.join(CHROMA_DB_PATH, "chroma.sqlite3"))

print("=" * 50)
print("CEK LOKASI DATA CHROMADB")
print("=" * 50)
print(f"\n[1] Path lokal  : {os.path.abspath(CHROMA_DB_PATH)}")
print(f"    Folder ada  : {'✓ YA' if local_exists else '✗ TIDAK'}")
print(f"    SQLite ada  : {'✓ YA' if sqlite_exists else '✗ TIDAK'}")

# ── Cek 2: koneksi + hitung dokumen di local ──
print("\n[2] Koneksi PersistentClient (local):")
try:
    local_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collections  = local_client.list_collections()

    if not collections:
        print("    ✗ Tidak ada collection — data belum masuk local")
    else:
        for col in collections:
            count = col.count()
            print(f"    Collection: '{col.name}' → {count} dokumen")

except Exception as e:
    print(f"    ✗ Error: {e}")

# ── Cek 3: apakah load.py pakai CloudClient atau PersistentClient? ──
print("\n[3] Deteksi jenis client di load.py:")
load_path = os.path.join(os.path.dirname(__file__), "load.py")
if os.path.isfile(load_path):
    with open(load_path) as f:
        content = f.read()
    if "CloudClient" in content:
        print("    ⚠ load.py masih pakai CloudClient → data masuk CLOUD, bukan local")
        print("    → Ganti ke: chromadb.PersistentClient(path=CHROMA_DB_PATH)")
    elif "PersistentClient" in content:
        print("    ✓ load.py pakai PersistentClient → data masuk LOCAL")
    else:
        print("    ? Tidak bisa mendeteksi jenis client")

# ── Cek 4: sample isi dokumen (spot check) ──
print("\n[4] Sample dokumen dari collection 'stock_knowledge_base':")
try:
    col = local_client.get_collection("stock_knowledge_base")
    result = col.peek(limit=2)  # ambil 2 dokumen pertama

    for i, (doc_id, doc, meta) in enumerate(zip(
        result["ids"], result["documents"], result["metadatas"]
    )):
        print(f"\n  Dokumen #{i+1}")
        print(f"  ID       : {doc_id}")
        print(f"  Ticker   : {meta.get('ticker', 'N/A')} | Tanggal: {meta.get('date', 'N/A')}")
        print(f"  Preview  : {doc[:120]}...")

except Exception as e:
    print(f"  ✗ {e}")

print("\n" + "=" * 50)