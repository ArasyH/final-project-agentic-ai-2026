"""
Upsert dokumen hasil transform ke ChromaDB knowledge base.

Posisi dalam pipeline ETL:
    transform.py → list[DocOutput] → load.py → ChromaDB collection

Idempotent Full Refresh: setiap doc di-upsert via ``doc_id`` sebagai
ChromaDB ID. Upsert pada doc_id yang sama tidak menghasilkan duplikat.
``load_all(reset=True)`` (default) menghapus collection lama lalu
recreate sebelum upsert — memastikan KB bersih dari dokumen stale.

Embedding function yang dipakai SAMA dengan ``app/services/retrieval_service.py``
(model ``paraphrase-multilingual-MiniLM-L12-v2``) dan jarak yang sama dengan
``app/services/cache_service.py`` (``hnsw:space: cosine``). Cosine similarity
antara embedding ETL vs LangChain HuggingFaceEmbeddings = 1.000000 (verified).

Referensi:
    §3  #4 — Idempotent Full Refresh
    §15 — ChromaDB Metadata Schema (7 field)
"""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from etl import config
from etl.schemas import DocOutput
from etl.aggregate import aggregate_all
from etl.transform import transform_all


# ── Module-level singletons ───────────────────────────────────────────────────

_client: Any = None
_embedding_fn: Any = None


# ── Private helpers ───────────────────────────────────────────────────────────

def _get_embedding_function() -> Any:
    """Kembalikan singleton SentenceTransformerEmbeddingFunction.

    Menggunakan ``config.EMBEDDER_MODEL`` (paraphrase-multilingual-MiniLM-L12-v2).
    Model yang sama dengan ``app/services/retrieval_service.py`` →
    vektor embedding identik (cosine similarity = 1.0, verified).

    Returns:
        SentenceTransformerEmbeddingFunction yang sudah diinisialisasi.
    """
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.EMBEDDER_MODEL
            )
        )
    return _embedding_fn


def _get_client() -> Any:
    """Kembalikan singleton chromadb.PersistentClient.

    Path diambil dari ``config.CHROMA_DB_PATH`` agar konsisten dengan
    ``app/config.py`` yang membaca dari env var yang sama (``CHROMA_DB_PATH``).
    Lazy-init: client dibuat hanya pada pemanggilan pertama.

    Returns:
        chromadb.PersistentClient yang sudah terhubung ke CHROMA_DB_PATH.
    """
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
    return _client


def _sanitize_metadata(doc: DocOutput) -> dict[str, Any]:
    """Konversi DocMetadata ke dict ChromaDB yang aman (tanpa nilai None).

    ChromaDB v1.5.x raise ``ValueError`` jika ada metadata value berupa
    ``None``. Field ``source_endpoint`` bisa ``None`` untuk aggregate docs.
    Strategi: ganti ``None`` dengan string kosong ``""`` agar semua key
    tetap hadir — filter di app/ tetap bisa menggunakan field ini.

    Args:
        doc: DocOutput yang metadata-nya akan disanitasi.

    Returns:
        Dict metadata siap upsert ke ChromaDB, semua value berupa str.
    """
    raw = doc.metadata.model_dump()
    return {k: ("" if v is None else str(v)) for k, v in raw.items()}


# ── Public functions ──────────────────────────────────────────────────────────

def get_or_create_collection(reset: bool = False) -> Any:
    """Ambil atau buat collection ChromaDB untuk knowledge base.

    Collection dibuat dengan:
    - Nama: ``config.CHROMA_COLLECTION_NAME`` (``"stock_knowledge_base"``)
    - Embedding function: ``SentenceTransformerEmbeddingFunction`` (config.EMBEDDER_MODEL)
    - Distance metric: cosine (``hnsw:space: cosine``) — match dengan
      ``app/services/cache_service.py`` dan sesuai model sentence-transformers.

    Jika ``reset=True``, collection lama di-delete terlebih dahulu lalu
    di-recreate dari scratch. Ini adalah operasi **destruktif** —
    semua dokumen lama hilang. Gunakan hanya saat Idempotent Full Refresh
    yang disengaja.

    Args:
        reset: Jika ``True``, hapus collection existing dan recreate.
            Default ``False`` (upsert-over ke collection existing).

    Returns:
        chromadb Collection object.
    """
    client = _get_client()
    ef = _get_embedding_function()
    name = config.CHROMA_COLLECTION_NAME

    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            # Collection belum ada — expected, bukan error
            pass

    return client.get_or_create_collection(
        name=name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents(docs: list[DocOutput], collection: Any) -> int:
    """Batch upsert list DocOutput ke ChromaDB collection.

    Setiap dokumen di-upsert dengan:
    - ``ids``       = ``doc.metadata.doc_id``
    - ``documents`` = ``doc.content`` (teks yang akan di-embed)
    - ``metadatas`` = dict 7-field yang sudah disanitasi (tanpa None)

    Upsert (bukan insert) — dokumen dengan ``doc_id`` yang sama
    di-overwrite, bukan ditambahkan. Ini menjamin idempotency.

    Batch size 100 untuk efisiensi. Untuk 30 saham × 15 doc ≈ 450 doc
    + ~15 aggregate = ~465 doc, batch ini cukup dalam 5 call.

    Args:
        docs: List DocOutput dari transform.py.
        collection: chromadb Collection object dari
            ``get_or_create_collection()``.

    Returns:
        Jumlah dokumen yang di-upsert (int).
    """
    if not docs:
        return 0

    batch_size = 100
    total = 0

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        collection.upsert(
            ids=[d.metadata.doc_id for d in batch],
            documents=[d.content for d in batch],
            metadatas=[_sanitize_metadata(d) for d in batch],
        )
        total += len(batch)

    return total


def load_all(snapshot_date: str, reset: bool = True) -> dict[str, int]:
    """Orchestrate transform → flatten → load untuk semua ticker IDX30.

    Urutan:
    1. ``transform_all(snapshot_date)`` — hasilkan dict {ticker: list[DocOutput]}
    2. Flatten semua DocOutput ke satu list
    3. ``get_or_create_collection(reset=reset)``
    4. ``load_documents(flat_docs, collection)``

    Default ``reset=True`` untuk Idempotent Full Refresh: collection lama
    dihapus dan dibuat ulang sebelum upsert, sehingga tidak ada dokumen
    stale dari snapshot sebelumnya yang tersisa.

    **PERINGATAN**: ``reset=True`` akan menghapus semua dokumen existing
    di collection ``config.CHROMA_COLLECTION_NAME``. Jalankan ke DB asli
    hanya setelah konfirmasi peneliti.

    Args:
        snapshot_date: Tanggal snapshot ISO 8601, mis. ``"2026-05-28"``.
        reset: Jika ``True`` (default), wipe + recreate collection sebelum
            load. Jika ``False``, upsert-over ke collection existing.

    Returns:
        Dict summary: ``{"tickers": N, "documents": M, "aggregate_docs": K}``
        di mana N = jumlah ticker berhasil di-transform, M = total dokumen
        di-upsert (per-saham + aggregate), K = jumlah aggregate docs.
    """
    ticker_docs: dict[str, list[DocOutput]] = transform_all(snapshot_date)
    agg_docs: list[DocOutput] = aggregate_all(snapshot_date)

    flat_docs: list[DocOutput] = [
        doc
        for docs_per_ticker in ticker_docs.values()
        for doc in docs_per_ticker
    ] + agg_docs

    collection = get_or_create_collection(reset=reset)
    total_docs = load_documents(flat_docs, collection)

    return {
        "tickers": len(ticker_docs),
        "documents": total_docs,
        "aggregate_docs": len(agg_docs),
    }
