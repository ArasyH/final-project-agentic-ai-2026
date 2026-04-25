from __future__ import annotations
# output terstruktur dari internal JSON
# app/schemas.py
from typing import Any, Literal
from pydantic import BaseModel, Field

ExperimentMode = Literal[
    "mode_1_baseline_llm",
    "mode_2_rag_only",
    "mode_3_full_agentic",
]

ValidatorStatus = Literal["passed", "failed", "skipped"]
CacheStatus = Literal["hit", "miss", "stale", "bypassed"]

class SourceItem(BaseModel):
    source_id: str
    title: str | None = None
    snippet: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class EvidenceItem(BaseModel):
    content: str
    source_id: str | None = None

class InternalResponse(BaseModel):
    answer: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    timestamp: str
    confidence: float = 0.0
    validator_status: ValidatorStatus = "skipped"
    cache_status: CacheStatus = "miss"
    mode: ExperimentMode
    metadata: dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    mode: ExperimentMode | None = None

class ChatResponse(BaseModel):
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