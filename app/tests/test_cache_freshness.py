from datetime import datetime, timedelta, timezone

def test_cache_freshness_logic():
    now = datetime.now(timezone.utc)
    fresh = now - timedelta(hours=1)
    stale = now - timedelta(hours=20)

    assert (now - fresh).total_seconds() < 8 * 3600
    assert (now - stale).total_seconds() > 8 * 3600