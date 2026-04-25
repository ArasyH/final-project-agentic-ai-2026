from app.services.validator_service import ValidatorService

def test_validator_passed_when_number_supported():
    validator = ValidatorService()
    result = validator.validate(
        answer="Harga BBCA adalah 9125.",
        evidence=[{"content": "Harga penutupan BBCA adalah 9125."}],
        expected_tickers=["BBCA"],
    )
    assert result["status"] == "passed"

def test_validator_failed_when_number_not_supported():
    validator = ValidatorService()
    result = validator.validate(
        answer="Harga BBCA adalah 9999.",
        evidence=[{"content": "Harga penutupan BBCA adalah 9125."}],
        expected_tickers=["BBCA"],
    )
    assert result["status"] == "failed"