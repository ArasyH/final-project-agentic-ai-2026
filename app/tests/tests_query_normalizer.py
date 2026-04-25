from app.services.query_normalizer import normalize_query

def test_synonym_to_ticker():
    result = normalize_query("harga Bank Central Asia hari ini")
    assert "BBCA" in result.detected_tickers
    assert "bbca" in result.normalized_query

def test_typo_normalization():
    result = normalize_query("harga bbac sekarang")
    assert "bbca" in result.normalized_query