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

def load_fundamental(ticker: str, report_data: dict):
    """Load company report ke collection 'fundamental'."""
    client = get_chroma_client()
    col = client.get_or_create_collection("fundamental", embedding_function=embed_fn)
    
    # Buat teks yang semantically rich untuk di-embed
    text = f"""
    Ticker: {ticker}
    Nama: {report_data.get('company_name', '')}
    Sektor: {report_data.get('sector', '')}
    Sub-sektor: {report_data.get('sub_sector', '')}
    Revenue TTM: {report_data.get('total_revenue_mrq', 'N/A')}
    ROE TTM: {report_data.get('roe_ttm', 'N/A')}
    P/E TTM: {report_data.get('pe_ttm', 'N/A')}
    Deskripsi bisnis: {report_data.get('business_description', '')}
    """
    
    col.upsert(
        ids=[f"{ticker}_fundamental"],
        documents=[text.strip()],
        metadatas=[{
            "ticker": ticker,
            "type": "fundamental",
            "updated_at": date.today().isoformat(),
            **{k: str(v) for k, v in report_data.items() if isinstance(v, (str, int, float))}
        }]
    )

def load_quarterly(ticker: str, quarter: str, fin_data: dict):
    """Load quarterly financials. quarter format: '2025-Q1'"""
    client = get_chroma_client()
    col = client.get_or_create_collection("quarterly", embedding_function=embed_fn)
    
    text = f"""
    Ticker: {ticker} | Periode: {quarter}
    Revenue: {fin_data.get('revenue', 'N/A')}
    Laba bersih: {fin_data.get('earnings', 'N/A')}
    EBITDA: {fin_data.get('ebitda', 'N/A')}
    Total aset: {fin_data.get('total_assets', 'N/A')}
    Total utang: {fin_data.get('total_debt', 'N/A')}
    """
    
    col.upsert(
        ids=[f"{ticker}_{quarter}"],
        documents=[text.strip()],
        metadatas=[{
            "ticker": ticker,
            "quarter": quarter,
            "type": "quarterly",
            **{k: str(v) for k, v in fin_data.items() if isinstance(v, (str, int, float))}
        }]
    )

def load_daily(ticker: str, records: list):
    """Bulk upsert daily transactions."""
    if not records:
        return
    
    client = get_chroma_client()
    col = client.get_or_create_collection("daily_tx", embedding_function=embed_fn)
    
    ids, docs, metas = [], [], []
    for r in records:
        tx_date = r.get("date") or r.get("transaction_date")
        if not tx_date:
            continue
        
        ids.append(f"{ticker}_{tx_date}")
        docs.append(
            f"Ticker {ticker} pada {tx_date}: "
            f"Close {r.get('close', 'N/A')}, "
            f"Volume {r.get('volume', 'N/A')}, "
            f"Foreign buy {r.get('foreign_buy', 'N/A')}, "
            f"Foreign sell {r.get('foreign_sell', 'N/A')}"
        )
        metas.append({
            "ticker": ticker,
            "date": tx_date,
            "type": "daily",
            **{k: str(v) for k, v in r.items() if isinstance(v, (str, int, float))}
        })
    
    # Upsert batch 100 dokumen sekaligus
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        col.upsert(
            ids=ids[i:i+batch_size],
            documents=docs[i:i+batch_size],
            metadatas=metas[i:i+batch_size]
        )
    
    print(f"  [{ticker}] Loaded {len(ids)} daily records ke ChromaDB")

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