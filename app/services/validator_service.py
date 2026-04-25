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

        if not evidence:
            issues.append("missing_evidence")

        for ticker in expected_tickers:
            if ticker not in answer and expected_tickers:
                issues.append(f"ticker_not_mentioned:{ticker}")

        numbers_in_answer = re.findall(r"\d[\d\.,]*", answer)
        evidence_blob = " ".join(item.get("content", "") for item in evidence)

        for num in numbers_in_answer:
            if num not in evidence_blob:
                issues.append(f"unsupported_number:{num}")

        status = "passed" if not issues else "failed"
        return {
            "status": status,
            "issues": issues,
            "confidence_penalty": min(0.5, 0.1 * len(issues)),
        }