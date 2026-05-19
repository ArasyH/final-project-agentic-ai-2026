from datetime import datetime, timezone

from app.services.guardrails_service import GuardrailsService


def test_passed_when_all_clean():
    svc = GuardrailsService()
    result = svc.check(
        answer="Harga BBCA pada penutupan 8 Mei 2026 adalah 9125 rupiah.",
        evidence=[{
            "content": "Harga penutupan BBCA tanggal 8 Mei 2026: 9125.",
            "metadata": {"timestamp": "2026-05-08T16:00:00+00:00"},
        }],
        now=datetime(2026, 5, 8, 18, 0, tzinfo=timezone.utc),
    )
    assert result.overall_status == "passed"
    assert result.hallucination_flags == []


def test_h1_flagged_when_number_not_in_evidence():
    svc = GuardrailsService()
    result = svc.check(
        answer="Harga BBCA adalah 9999.",
        evidence=[{"content": "Harga penutupan BBCA adalah 9125."}],
    )
    assert "H1" in result.hallucination_flags
    assert result.H1_unsupported_numeric["flag"] is True


def test_h3_flagged_when_evidence_stale_and_no_disclosure():
    svc = GuardrailsService(max_evidence_age_hours=24)
    result = svc.check(
        answer="Harga BBCA adalah 9125 rupiah.",
        evidence=[{
            "content": "Harga BBCA: 9125.",
            "metadata": {"timestamp": "2026-05-01T16:00:00+00:00"},
        }],
        now=datetime(2026, 5, 8, 18, 0, tzinfo=timezone.utc),
    )
    assert "H3" in result.hallucination_flags


def test_h3_passes_with_explicit_disclosure():
    svc = GuardrailsService(max_evidence_age_hours=24)
    result = svc.check(
        answer="Harga BBCA per tanggal 1 Mei 2026 adalah 9125.",
        evidence=[{
            "content": "Harga BBCA: 9125.",
            "metadata": {"timestamp": "2026-05-01T16:00:00+00:00"},
        }],
        now=datetime(2026, 5, 8, 18, 0, tzinfo=timezone.utc),
    )
    assert "H3" not in result.hallucination_flags


def test_no_investment_recommendation_blocked():
    svc = GuardrailsService()
    result = svc.check(
        answer="Saya rekomendasikan beli BBCA sekarang.",
        evidence=[{"content": "Harga BBCA: 9125."}],
    )
    assert result.no_investment_recommendation["flag"] is True
    assert result.overall_status == "failed"


def test_short_numbers_skipped_for_h1():
    """Single-digit numbers (e.g., '1', '0') tidak di-flag H1."""
    svc = GuardrailsService()
    result = svc.check(
        answer="Saham 1 ini bagus.",
        evidence=[{"content": "Tidak ada angka."}],
    )
    assert result.H1_unsupported_numeric["flag"] is False