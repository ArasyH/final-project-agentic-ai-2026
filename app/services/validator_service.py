from __future__ import annotations
# inti mitigasi halusinasi pada mode 3 (FULL AGENTIC)
# app/services/validator_service.py
import re

class ValidatorService:
    def validate(
        self,
        answer: str,
        evidence: list[dict],
        expected_tickers: list[str],
    ) -> dict:
        issues = []
        evidence_blob = " ".join(item.get("content", "") for item in evidence)

        if not evidence:
            issues.append("missing_evidence")
            return {"status": "failed", "issues": issues, "confidence_penalty": 0.5}

        # ── Cek ticker: terima juga nama perusahaan sebagai pengganti ticker ──
        TICKER_ALIASES = {
            "BBCA": ["bbca", "bank central asia", "bca"],
            "BBRI": ["bbri", "bank rakyat indonesia", "bri"],
            "BMRI": ["bmri", "bank mandiri", "mandiri"],
            "BBNI": ["bbni", "bank negara indonesia", "bni"],
            "TLKM": ["tlkm", "telkom"],
            "UNVR": ["unvr", "unilever"],
        }
        answer_lower = answer.lower()
        for ticker in expected_tickers:
            aliases = TICKER_ALIASES.get(ticker, [ticker.lower()])
            if not any(alias in answer_lower for alias in aliases):
                issues.append(f"ticker_not_mentioned:{ticker}")

        # ── Cek angka: normalisasi dulu sebelum bandingkan ──
        def normalize_num(s: str) -> str:
            """Hapus semua pemisah ribuan dan desimal, sisakan digit."""
            return re.sub(r"[.,\s]", "", s)

        # Kumpulkan semua angka dari evidence (sudah dinormalisasi)
        evidence_nums = {
            normalize_num(n)
            for n in re.findall(r"\d[\d.,]*", evidence_blob)
            if len(normalize_num(n)) >= 3   # abaikan angka 1-2 digit
        }

        # Cek angka di jawaban
        answer_nums = re.findall(r"\d[\d.,]*", answer)
        unsupported = []
        for num in answer_nums:
            norm = normalize_num(num)
            if len(norm) < 3:
                continue  # abaikan angka kecil
            if norm not in evidence_nums:
                unsupported.append(num)

        # Hanya fail kalau LEBIH DARI SEPARUH angka tidak didukung evidence
        # (toleransi untuk pembulatan dan format berbeda dari LLM)
        if unsupported and len(unsupported) > len(answer_nums) / 2:
            issues.append(f"unsupported_numbers:{','.join(unsupported[:3])}")

        status = "passed" if not issues else "failed"
        return {
            "status": status,
            "issues": issues,
            "confidence_penalty": min(0.4, 0.1 * len(issues)),
        }