from __future__ import annotations
"""Domain Guardrails — 3 aturan deterministik untuk mitigasi halusinasi.

Mapping:
- H1 (Unsupported Numeric Claim) → numeric_traceability rule
- H3 (Stale Timestamp Misrepresentation) → timestamp_freshness rule
- No Investment Recommendation → pattern-based block (domain rule)

Threshold H3 (MAX_EVIDENCE_AGE_HOURS) configurable via app/config.py
(K2 decision: opsi (c), default 30 jam untuk akomodasi weekend BEI).

CATATAN: H2 + H4 NYA di Critic Agent (LLM-based). Service ini sengaja
TIDAK memeriksa H2/H4 untuk menjaga separation of concerns: deterministik
vs LLM-based.
"""
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.config import MAX_EVIDENCE_AGE_HOURS

# ----- Pattern definitions (K3 decision: opsi (i), inline starter list) -----

# Rekomendasi investasi terlarang. Case-insensitive regex.
INVESTMENT_RECOMMENDATION_PATTERNS: list[str] = [
    r"\brekomendasi(kan)?\b",
    r"\bsebaiknya (beli|jual|tahan)\b",
    r"\bwajib (beli|jual)\b",
    r"\bsaya sarankan (beli|jual)\b",
    r"\btahan saja\b",
    r"\bpotensi cuan\b",
    r"\bbuy on weakness\b",
    r"\btarget harga.*?\d",
    r"\bharus (beli|jual)\b",
    r"\b(I|you) (should|recommend|suggest) (buy|sell|invest|hold)\b",
    r"\bgood (buy|investment)\b",
    r"\bstrong buy\b",
    r"\bmust buy\b",
]

# Penanda waktu di answer yang menggugurkan flag H3.
TIMESTAMP_DISCLOSURE_PATTERNS: list[str] = [
    r"per tanggal",
    r"data hingga",
    r"as of",
    r"hingga (tanggal|hari)",
    r"\d{1,2}[/\-\s]\w+[/\-\s]\d{4}",  # "12 Mei 2026" atau "12/05/2026"
    r"\d{4}-\d{2}-\d{2}",  # ISO date
]

# Pattern token numerik (untuk H1 + H3 trigger detection).
NUMERIC_TOKEN_PATTERN = re.compile(r"\d+(?:[\.,]\d+)*")


# ----- Output schema -----

class GuardrailResult(BaseModel):
    """Hasil 3 rule deterministik. Struktur kompatibel dengan Critic verdict."""
    H1_unsupported_numeric: dict
    H3_stale_timestamp: dict
    no_investment_recommendation: dict
    overall_status: str  # "passed" | "failed"

    @property
    def hallucination_flags(self) -> list[str]:
        """Subset H1/H3 yang ter-flag oleh guardrails (untuk telemetry merge)."""
        flags: list[str] = []
        if self.H1_unsupported_numeric.get("flag"):
            flags.append("H1")
        if self.H3_stale_timestamp.get("flag"):
            flags.append("H3")
        return flags


# ----- Helper functions -----

def _normalize_number(token: str) -> str:
    """Strip thousand separators (. dan ,) untuk perbandingan token numerik.

    Catatan: pendekatan konservatif — kehilangan info desimal, tapi cukup
    untuk traceability check level thesis (false-positive lebih aman daripada
    false-negative untuk H1).
    """
    return token.replace(".", "").replace(",", "")


def _check_numeric_traceability(answer: str, evidence_text: str) -> dict:
    """H1: setiap angka di answer harus appear di evidence (substring/normalized)."""
    answer_numbers = NUMERIC_TOKEN_PATTERN.findall(answer)
    evidence_numbers = NUMERIC_TOKEN_PATTERN.findall(evidence_text)
    evidence_normalized: set[str] = {_normalize_number(n) for n in evidence_numbers}

    unsupported: list[str] = []
    for num in answer_numbers:
        if len(num) < 2:  # skip single-digit (likely list index/year fragment)
            continue
        if num in evidence_text:
            continue
        if _normalize_number(num) in evidence_normalized:
            continue
        unsupported.append(num)

    if unsupported:
        return {
            "flag": True,
            "rationale": f"Angka tidak ter-trace di evidence: {unsupported}",
            "evidence": {"unsupported_numbers": unsupported},
        }
    return {
        "flag": False,
        "rationale": "Semua angka pada jawaban ter-trace di evidence.",
        "evidence": {"checked_numbers": answer_numbers},
    }


def _check_timestamp_freshness(
    answer: str,
    evidence_timestamps: list[datetime],
    max_age_hours: int,
    now: datetime,
) -> dict:
    """H3: AND-condition antara claim numerik, evidence basi, tanpa disclosure."""
    has_numeric_claim = bool(NUMERIC_TOKEN_PATTERN.search(answer))
    if not has_numeric_claim:
        return {
            "flag": False,
            "rationale": "Answer tidak mengandung claim numerik.",
            "evidence": {"has_numeric_claim": False},
        }

    has_disclosure = any(
        re.search(pat, answer, re.IGNORECASE)
        for pat in TIMESTAMP_DISCLOSURE_PATTERNS
    )
    if has_disclosure:
        return {
            "flag": False,
            "rationale": "Answer mengandung disclaimer/timestamp.",
            "evidence": {"has_disclosure": True},
        }

    if not evidence_timestamps:
        return {
            "flag": False,
            "rationale": "Evidence tidak punya timestamp untuk diperiksa.",
            "evidence": {"timestamps_count": 0},
        }

    most_recent = max(evidence_timestamps)
    age_hours = (now - most_recent).total_seconds() / 3600

    if age_hours > max_age_hours:
        return {
            "flag": True,
            "rationale": (
                f"Evidence terbaru berumur {age_hours:.1f} jam "
                f"(> {max_age_hours} jam threshold) tetapi answer tidak "
                f"mencantumkan disclaimer waktu."
            ),
            "evidence": {
                "max_age_hours_threshold": max_age_hours,
                "most_recent_age_hours": round(age_hours, 2),
            },
        }
    return {
        "flag": False,
        "rationale": f"Evidence terbaru cukup segar ({age_hours:.1f} jam).",
        "evidence": {"most_recent_age_hours": round(age_hours, 2)},
    }


def _check_investment_recommendation(answer: str) -> dict:
    """Block jika answer mengandung pola rekomendasi investasi."""
    matches: list[str] = [
        pat for pat in INVESTMENT_RECOMMENDATION_PATTERNS
        if re.search(pat, answer, re.IGNORECASE)
    ]
    if matches:
        return {
            "flag": True,
            "rationale": f"Answer mengandung pola rekomendasi: {matches[:3]}",
            "evidence": {"matched_patterns": matches},
        }
    return {
        "flag": False,
        "rationale": "Tidak ada pola rekomendasi investasi terdeteksi.",
        "evidence": {"matched_patterns": []},
    }


def _extract_evidence_timestamps(evidence: list[dict]) -> list[datetime]:
    """Parse timestamps dari evidence[i]['metadata']['timestamp'|'date']."""
    out: list[datetime] = []
    for e in evidence:
        metadata = e.get("metadata") or {}
        ts_str = metadata.get("timestamp") or metadata.get("snapshot_date")
        if not ts_str:
            continue
        try:
            parsed = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            out.append(parsed)
        except (ValueError, TypeError):
            continue
    return out


# ----- Service class -----

class GuardrailsService:
    """3 deterministic rules: H1 + H3 + no-investment-recommendation."""

    def __init__(self, max_evidence_age_hours: int | None = None) -> None:
        """
        Args:
            max_evidence_age_hours: override untuk H3 staleness threshold.
                Default: ambil dari config (MAX_EVIDENCE_AGE_HOURS).
        """
        self.max_age_hours: int = (
            max_evidence_age_hours
            if max_evidence_age_hours is not None
            else MAX_EVIDENCE_AGE_HOURS
        )

    def check(
        self,
        answer: str,
        evidence: list[dict],
        now: datetime | None = None,
    ) -> GuardrailResult:
        """Run all 3 deterministic rules.

        Args:
            answer: teks jawaban yang akan diperiksa.
            evidence: list of dict, minimal key 'content'.
                Optional 'metadata' dict dengan key 'timestamp' atau 'date'
                (ISO 8601) untuk H3.
            now: clock override untuk reproducibility test. Default: UTC now.

        Returns:
            GuardrailResult dengan 3 flag + overall_status.
        """
        clock = now or datetime.now(timezone.utc)
        evidence_text = " ".join(e.get("content", "") for e in evidence)
        evidence_timestamps = _extract_evidence_timestamps(evidence)

        h1 = _check_numeric_traceability(answer, evidence_text)
        h3 = _check_timestamp_freshness(
            answer, evidence_timestamps, self.max_age_hours, clock,
        )
        nir = _check_investment_recommendation(answer)

        any_failed = h1["flag"] or h3["flag"] or nir["flag"]

        return GuardrailResult(
            H1_unsupported_numeric=h1,
            H3_stale_timestamp=h3,
            no_investment_recommendation=nir,
            overall_status="failed" if any_failed else "passed",
        )