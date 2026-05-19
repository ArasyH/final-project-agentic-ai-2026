from unittest.mock import MagicMock, patch

from app.modes.mode_4_rag_jc_cache import run_mode_4
from app.schemas import InternalResponse

_CACHED_ANSWER = "Harga BBCA penutupan terakhir adalah 9125 per lembar."

_HIT_DICT = {
    "hit": True,
    "status": "hit",
    "score": 0.93,
    "answer": _CACHED_ANSWER,
    "intent": "price_lookup",
    "evidence_summary": [{"content": "BBCA close: 9125.", "source_id": "kb_0"}],
    "source_metadata": [{"source_id": "kb_0", "snippet": "BBCA close: 9125."}],
    "timestamp": "2026-05-19T08:00:00+00:00",
}

_MISS_DICT = {"hit": False, "status": "miss", "score": 0.12}


def _make_passed_response() -> InternalResponse:
    return InternalResponse(
        answer=_CACHED_ANSWER,
        evidence=[],
        sources=[],
        tickers=["BBCA"],
        timestamp="2026-05-19T09:00:00+00:00",
        confidence=0.85,
        validator_status="passed",
        cache_status="miss",
        mode="mode_4_rag_jc_cache",
        hallucination_flags=[],
    )


def _make_failed_response() -> InternalResponse:
    return InternalResponse(
        answer="Harga BBCA adalah 9999.",
        evidence=[],
        sources=[],
        tickers=["BBCA"],
        timestamp="2026-05-19T09:00:00+00:00",
        confidence=0.50,
        validator_status="failed",
        cache_status="miss",
        mode="mode_4_rag_jc_cache",
        hallucination_flags=["H1"],
    )


def test_cache_hit_bypasses_pipeline():
    """Cache hit harus langsung return tanpa memanggil pipeline sama sekali."""
    with (
        patch("app.modes.mode_4_rag_jc_cache.CacheService") as MockCache,
        patch("app.modes.mode_4_rag_jc_cache._run_rag_jc_pipeline") as mock_pipeline,
        patch("app.modes.mode_4_rag_jc_cache.TelemetryService"),
    ):
        mock_cache_instance = MockCache.return_value
        mock_cache_instance.lookup.return_value = _HIT_DICT

        result = run_mode_4("Berapa harga BBCA?", "s1")

        assert result.cache_status == "hit"
        assert result.mode == "mode_4_rag_jc_cache"
        assert result.answer == _CACHED_ANSWER
        assert result.validator_status == "passed"
        mock_pipeline.assert_not_called()
        mock_cache_instance.store.assert_not_called()


def test_cache_miss_passing_validator_stores():
    """Cache miss + validator passed → pipeline dipanggil dan hasil di-store."""
    with (
        patch("app.modes.mode_4_rag_jc_cache.CacheService") as MockCache,
        patch("app.modes.mode_4_rag_jc_cache._run_rag_jc_pipeline") as mock_pipeline,
        patch("app.modes.mode_4_rag_jc_cache.TelemetryService"),
    ):
        mock_cache_instance = MockCache.return_value
        mock_cache_instance.lookup.return_value = _MISS_DICT
        mock_pipeline.return_value = _make_passed_response()

        result = run_mode_4("Berapa harga BBCA?", "s1")

        assert result.cache_status == "miss"
        assert result.validator_status == "passed"
        mock_pipeline.assert_called_once()
        mock_cache_instance.store.assert_called_once()


def test_cache_miss_failing_validator_not_stored():
    """Cache miss + validator failed → pipeline dipanggil tapi hasil tidak di-store."""
    with (
        patch("app.modes.mode_4_rag_jc_cache.CacheService") as MockCache,
        patch("app.modes.mode_4_rag_jc_cache._run_rag_jc_pipeline") as mock_pipeline,
        patch("app.modes.mode_4_rag_jc_cache.TelemetryService"),
    ):
        mock_cache_instance = MockCache.return_value
        mock_cache_instance.lookup.return_value = _MISS_DICT
        mock_pipeline.return_value = _make_failed_response()

        result = run_mode_4("Berapa harga BBCA?", "s1")

        assert result.cache_status == "miss"
        assert result.validator_status == "failed"
        assert "H1" in result.hallucination_flags
        mock_pipeline.assert_called_once()
        mock_cache_instance.store.assert_not_called()
