from app.schemas import InternalResponse


def test_fallback_shape_mode3():
    """InternalResponse valid untuk fail-safe path mode_3_rag_jc."""
    resp = InternalResponse(
        answer="Maaf, sistem tidak dapat menghasilkan jawaban saat ini.",
        evidence=[],
        sources=[],
        tickers=[],
        timestamp="2026-04-23T00:00:00Z",
        confidence=0.2,
        validator_status="failed",
        cache_status="bypassed",
        mode="mode_3_rag_jc",
    )
    assert resp.answer.startswith("Maaf")
    assert resp.validator_status == "failed"
    assert resp.confidence == 0.2
    assert resp.hallucination_flags == []


def test_fallback_shape_mode4():
    """InternalResponse valid untuk fail-safe path mode_4_rag_jc_cache."""
    resp = InternalResponse(
        answer="Maaf, sistem tidak dapat menghasilkan jawaban saat ini.",
        evidence=[],
        sources=[],
        tickers=[],
        timestamp="2026-04-23T00:00:00Z",
        confidence=0.2,
        validator_status="failed",
        cache_status="miss",
        mode="mode_4_rag_jc_cache",
    )
    assert resp.validator_status == "failed"
    assert resp.mode == "mode_4_rag_jc_cache"
