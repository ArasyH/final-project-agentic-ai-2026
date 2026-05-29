"""
Data contracts internal pipeline ETL IDX30.

Mendefinisikan dua model Pydantic v2 yang menjadi kontrak tipe bersama
antara transform.py, aggregate.py, dan load.py:

- ``DocMetadata``: 7-field metadata yang di-upsert ke ChromaDB per dokumen.
- ``DocOutput``: pasangan (content, metadata) yang dihasilkan setiap fungsi
  transform dan aggregate sebelum dikirim ke load.

Referensi: §15 ChromaDB Metadata Schema (system prompt).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocMetadata(BaseModel):
    """Metadata ChromaDB untuk satu dokumen knowledge base.

    Setiap dokumen yang di-upsert ke collection ``stock_knowledge_base``
    harus memiliki metadata persis sesuai skema ini. Field ``doc_id``
    dipakai sebagai ID upsert agar pipeline bersifat idempotent.

    Attributes:
        category: Jenis dokumen — menentukan topik yang dikandung.
        symbol: Ticker saham (mis. ``"BBCA"``). Kosong untuk dokumen
            aggregate lintas-saham.
        sector: Sektor saham (mis. ``"Financials"``). Kosong untuk
            aggregate_ranking.
        period: Periode data (mis. ``"2024"``, ``"Q1-2026"``,
            ``"snapshot"``). Kosong untuk aggregate.
        snapshot_date: Tanggal ETL run dalam format ISO 8601
            ``YYYY-MM-DD``. Dipakai untuk traceability H3.
        doc_id: Identifier unik dokumen — dipakai sebagai ChromaDB
            upsert ID agar full refresh bersifat idempotent.
        source_endpoint: Endpoint Sectors.app asal data. ``None`` jika
            dokumen diturunkan dari aggregate (bukan fetch langsung).
    """

    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "profile",
        "valuation",
        "financials_annual",
        "financials_quarterly",
        "dividend",
        "growth",
        "price_snapshot",
        "aggregate_ranking",
        "aggregate_sector",
    ]
    symbol: str = Field(default="")
    sector: str = Field(default="")
    period: str = Field(default="")
    snapshot_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    doc_id: str = Field(min_length=1)
    source_endpoint: Literal["company_report", "daily_transaction"] | None = None


class DocOutput(BaseModel):
    """Pasangan (konten teks, metadata) satu dokumen output ETL.

    Dihasilkan oleh setiap fungsi di transform.py dan aggregate.py,
    lalu dikonsumsi oleh load.py untuk di-upsert ke ChromaDB.
    ``content`` adalah teks yang akan di-embed; ``metadata`` adalah
    skema tujuh field yang menyertai embedding di vector store.

    Attributes:
        content: Teks dokumen yang akan di-embed. Tidak boleh kosong.
        metadata: Metadata ChromaDB lengkap sesuai ``DocMetadata``.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    metadata: DocMetadata
