from __future__ import annotations
# app/schemas.py
from typing import Any, Literal
from pydantic import BaseModel, Field

ExperimentMode = Literal[
    "mode_1_llm_only",
    "mode_2_rag_only",
    "mode_3_rag_jc",
    "mode_4_rag_jc_cache",
]

# H1=unsupported_numeric_claim, H2=fabricated_financial_metric,
# H3=stale_timestamp_misrepresentation, H4=incorrect_inference
HallucinationFlag = Literal["H1", "H2", "H3", "H4"]

ValidatorStatus = Literal["passed", "failed", "skipped"]
CacheStatus = Literal["hit", "miss", "stale", "bypassed"]


class SourceItem(BaseModel):
    """Satu dokumen sumber yang dikembalikan oleh retriever."""

    source_id: str
    title: str | None = None
    snippet: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    """Potongan teks evidence yang dipakai untuk generate jawaban."""

    content: str
    source_id: str | None = None


class InternalResponse(BaseModel):
    """Output internal yang dikembalikan oleh setiap mode runner.

    Field `hallucination_flags` berisi subset dari {"H1","H2","H3","H4"};
    diisi oleh GuardrailsService (H1, H3) dan CriticAgent (H2, H4).
    """

    answer: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    timestamp: str
    confidence: float = 0.0
    validator_status: ValidatorStatus = "skipped"
    cache_status: CacheStatus = "miss"
    mode: ExperimentMode
    hallucination_flags: list[HallucinationFlag] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """Payload POST /chat dari klien."""

    question: str
    session_id: str | None = None
    mode: ExperimentMode | None = None
    question_id: str | None = None


class ChatResponse(BaseModel):
    """Response publik API /chat yang dikembalikan ke klien."""

    answer: str
    mode: ExperimentMode
    source: str
    similarity: float
    latency_ms: float
    session_id: str
    validator_status: ValidatorStatus
    cache_status: CacheStatus
    sources: list[SourceItem] = Field(default_factory=list)
    confidence: float = 0.0
