# Membaca narasi teks, membuat embedding, dan memasukkannya ke vector databse (chromaDB)
# TASK 3
import os
from datetime import date
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
from dotenv import load_dotenv


load_dotenv()
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

def load_to_vector_db():
    today = date.today().isoformat()

    # Cek file tersedia
    txt_files = [f for f in os.listdir(PROCESSED_DATA_DIR)
                 if f.endswith(f"{today}.txt")]

    if not txt_files:
        print(f"⚠ Tidak ada file .txt untuk {today}")
        print(f"  Jalankan transform.py terlebih dahulu.")
        return

    print(f"Ditemukan {len(txt_files)} file — mulai loading ke ChromaDB...")
    print(f"Path  : {os.path.abspath(CHROMA_DB_PATH)}")
    print(f"Model : {EMBEDDING_MODEL_NAME}\n")

    # Embedding function — ChromaDB yang handle, bukan manual encode
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col    = client.get_or_create_collection(
        name="stock_knowledge_base",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Dokumen sebelum load: {col.count()}")

    ok, err = 0, 0
    for filename in sorted(txt_files):
        ticker = filename.split("_")[0]
        try:
            with open(os.path.join(PROCESSED_DATA_DIR, filename), encoding="utf-8") as f:
                text = f.read()

            col.upsert(
                ids=[f"{ticker}_{today}"],
                documents=[text],           # ChromaDB auto-embed
                metadatas=[{"ticker": ticker, "date": today}]
            )
            print(f"  ✓ {ticker}")
            ok += 1

        except Exception as e:
            print(f"  ✗ {ticker}: {e}")
            err += 1

    print(f"\nSelesai: {ok} berhasil, {err} error")
    print(f"Total dokumen sekarang: {col.count()}")

if __name__ == "__main__":
    load_to_vector_db()