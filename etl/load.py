# Membaca narasi teks, membuat embedding, dan memasukkannya ke vector databse (chromaDB)
# TASK 3

import os
from datetime import date
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv


load_dotenv()
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

def load_to_vector_db():
    today     = date.today().isoformat()
    model     = SentenceTransformer("all-MiniLM-L6-v2")
    client    = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Gunakan satu koleksi untuk knowledge base
    collection = client.get_or_create_collection(
        name="stock_knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    
    loaded, skipped = 0, 0
    for filename in os.listdir(PROCESSED_DATA_DIR):
        if not filename.endswith(f"{today}.txt"):
            skipped += 1
            continue
        ticker   = filename.split("_")[0]
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)

        with open(filepath) as f:
            text = f.read()

        embedding = model.encode(text).tolist()
        doc_id    = f"{ticker}_{today}"

        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "ticker": ticker,
                "date":   today,
                "type":   "daily_narrative"  # metadata tambahan untuk filter agent
            }]
        )
        print(f"  Loaded {ticker} → {doc_id}")
        loaded += 1

    print(f"\nLoad selesai: {loaded} dokumen, {skipped} dilewati")