from app.services.guardrails_service import GuardrailsService


def test_guardrails_passed_when_number_supported():
    svc = GuardrailsService()
    result = svc.check(
        answer="Harga BBCA adalah 9125.",
        evidence=[{"content": "Harga penutupan BBCA adalah 9125."}],
    )
    assert result.overall_status == "passed"
    assert "H1" not in result.hallucination_flags


def test_guardrails_failed_when_number_not_supported():
    svc = GuardrailsService()
    result = svc.check(
        answer="Harga BBCA adalah 9999.",
        evidence=[{"content": "Harga penutupan BBCA adalah 9125."}],
    )
    assert result.overall_status == "failed"
    assert "H1" in result.hallucination_flags
