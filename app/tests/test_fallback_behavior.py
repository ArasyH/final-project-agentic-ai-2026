from app.schemas import InternalResponse

def test_fallback_shape():
    resp = InternalResponse(
        answer="Data tidak cukup untuk menjawab pertanyaan ini.",
        evidence=[],
        sources=[],
        tickers=[],
        timestamp="2026-04-23T00:00:00Z",
        confidence=0.2,
        validator_status="failed",
        cache_status="miss",
        mode="mode_3_full_agentic",
    )
    assert resp.answer.startswith("Data tidak cukup")
    assert resp.validator_status == "failed"