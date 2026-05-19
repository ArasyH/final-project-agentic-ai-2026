from __future__ import annotations
"""Critic Agent — LLM-based validation untuk H2 + H4.

Menggunakan Llama-3.3-70B-Versatile (Groq) dengan structured JSON output.
Output schema FROZEN (lihat system prompt §16, Keputusan #1).

Mapping:
- H2 (Fabricated Financial Metric): metrik finansial yang tidak ada di evidence
- H4 (Incorrect Inference): kesimpulan tidak konsisten secara logis dengan evidence

Catatan: Critic juga memeriksa H1 + H3 sebagai cross-check, namun deteksi
otoritatif untuk H1 + H3 ada di Domain Guardrails (deterministik).
"""
import json
import re
from typing import Any, Literal

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from app.config import GROQ_CRITIC_MODEL
from app.services.llm_service import build_critic_llm

CRITIC_TEMPERATURE: float = 0.0
PROMPT_VERSION: str = "critic_v1"


class CategoryFlag(BaseModel):
    """Satu kategori halusinasi: flag boolean + rationale string (Bahasa Indonesia)."""
    flag: bool = Field(description="True jika kategori halusinasi terdeteksi")
    rationale: str = Field(description="Penjelasan singkat dalam Bahasa Indonesia")


class CriticVerdict(BaseModel):
    """Output struktur frozen Critic Agent (system prompt §16).

    Catatan: `model` dan `temperature` di-fill dari config server-side, bukan
    dari LLM, untuk audit trail yang reliable.
    """
    H1_unsupported_numeric: CategoryFlag
    H2_fabricated_metric: CategoryFlag
    H3_stale_timestamp: CategoryFlag
    H4_incorrect_inference: CategoryFlag
    overall_verdict: Literal["pass", "fail"]
    model: str = GROQ_CRITIC_MODEL
    temperature: float = CRITIC_TEMPERATURE


CRITIC_PROMPT_V1 = """Anda adalah Critic Agent yang memvalidasi jawaban tentang pasar saham IDX30.

Tugas: periksa apakah JAWABAN konsisten dengan EVIDENCE, dan flag 4 kategori halusinasi.

KATEGORI HALUSINASI:
- H1 (Unsupported Numeric Claim): angka di jawaban tidak ada di evidence.
- H2 (Fabricated Financial Metric): metrik finansial (PER, PBV, ROE, EPS, dll) yang disebut di jawaban tetapi tidak muncul di evidence.
- H3 (Stale Timestamp Misrepresentation): klaim numerik disajikan tanpa konteks waktu yang sesuai.
- H4 (Incorrect Inference): kesimpulan tidak konsisten secara logis dengan evidence (contoh: evidence menunjukkan harga turun, jawaban bilang naik).

EVIDENCE:
{evidence}

PERTANYAAN:
{question}

JAWABAN UNTUK DIVALIDASI:
{answer}

INSTRUKSI OUTPUT:
Balas HANYA dengan JSON valid (tanpa preamble, tanpa markdown fence). Schema:
{{
  "H1_unsupported_numeric": {{"flag": <bool>, "rationale": "<id-string non-empty>"}},
  "H2_fabricated_metric": {{"flag": <bool>, "rationale": "<id-string non-empty>"}},
  "H3_stale_timestamp": {{"flag": <bool>, "rationale": "<id-string non-empty>"}},
  "H4_incorrect_inference": {{"flag": <bool>, "rationale": "<id-string non-empty>"}},
  "overall_verdict": "pass"
}}

Aturan:
- rationale wajib non-empty walaupun flag = false (untuk audit trail).
- overall_verdict akan dihitung server-side, kamu bisa selalu menulis "pass".
"""


class CriticAgent:
    """LLM-based critic (Llama-3.3-70B) untuk H1/H2/H3/H4 validation.

    Args:
        llm: Optional ChatGroq instance. Jika None, dibuat via build_critic_llm().
             Parameter ini ada untuk dependency injection (testing dengan mock).
    """

    def __init__(self, llm: Any = None) -> None:
        self._llm = llm or build_critic_llm(temperature=CRITIC_TEMPERATURE)
        self._prompt = PromptTemplate.from_template(CRITIC_PROMPT_V1)
        self.prompt_version: str = PROMPT_VERSION

    def validate(
        self,
        question: str,
        answer: str,
        evidence: list[dict],
    ) -> CriticVerdict:
        """Run critic LLM dan return parsed structured verdict.

        Args:
            question: pertanyaan asli pengguna.
            answer: jawaban yang akan divalidasi.
            evidence: list of dict dengan minimal key 'content'.

        Returns:
            CriticVerdict. Jika LLM error / JSON invalid, return fail-safe
            verdict (semua flag=true, overall_verdict="fail") supaya mode runner
            tidak crash.
        """
        evidence_blob = "\n\n".join(
            f"[evidence-{i}] {e.get('content', '')}" for i, e in enumerate(evidence)
        ) or "(tidak ada evidence)"

        prompt = self._prompt.format(
            evidence=evidence_blob,
            question=question,
            answer=answer,
        )

        try:
            raw = self._llm.invoke(prompt).content
            return self._parse(raw)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            return self._failsafe_verdict(reason=f"parse_error: {type(exc).__name__}")
        except Exception as exc:
            return self._failsafe_verdict(reason=f"llm_error: {type(exc).__name__}")

    def _parse(self, raw: str) -> CriticVerdict:
        """Parse JSON output dari critic LLM. Strip markdown fence kalau ada."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)

        # Override overall_verdict dari flags (jangan trust LLM untuk derived field)
        flags = [
            data.get("H1_unsupported_numeric", {}).get("flag", False),
            data.get("H2_fabricated_metric", {}).get("flag", False),
            data.get("H3_stale_timestamp", {}).get("flag", False),
            data.get("H4_incorrect_inference", {}).get("flag", False),
        ]
        data["overall_verdict"] = "fail" if any(flags) else "pass"

        return CriticVerdict(**data)

    def _failsafe_verdict(self, reason: str) -> CriticVerdict:
        """Fail-safe: kalau critic tidak bisa dijalankan, treat sebagai fail."""
        return CriticVerdict(
            H1_unsupported_numeric=CategoryFlag(flag=True, rationale=reason),
            H2_fabricated_metric=CategoryFlag(flag=True, rationale=reason),
            H3_stale_timestamp=CategoryFlag(flag=True, rationale=reason),
            H4_incorrect_inference=CategoryFlag(flag=True, rationale=reason),
            overall_verdict="fail",
        )