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
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

def load_to_vector_db():
    today     = date.today().isoformat()
    model     = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    client    = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )
    
    # Gunakan satu koleksi untuk knowledge base
    collection = client.get_or_create_collection(
        name="stock_knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    
    for filename in os.listdir(PROCESSED_DATA_DIR):
        if not filename.endswith(f"{today}.txt"):
            continue
        
        ticker   = filename.split("_")[0]
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        
        with open(filepath) as f:
            text = f.read()
        
        embedding = model.encode(text).tolist()
        doc_id    = f"{ticker}_{today}"
        
        # Upsert: update jika sudah ada, insert jika belum
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"ticker": ticker, "date": today}]
        )
        print(f"Loaded {ticker} ke ChromaDB")

if __name__ == "__main__":
    load_to_vector_db()
    print("Load selesai.")