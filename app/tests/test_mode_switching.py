from app.services.orchestrator_service import OrchestratorService

def test_invalid_mode_raises():
    orch = OrchestratorService()
    try:
        orch.run("test", "invalid_mode")
        assert False
    except ValueError:
        assert True