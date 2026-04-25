from __future__ import annotations
# pusat routing mode
# app/services/orchestrator_service.py
from app.schemas import InternalResponse
from app.modes.mode_1_baseline_llm import run_mode_1
from app.modes.mode_2_rag_only import run_mode_2
from app.modes.mode_3_full_agentic import run_mode_3

class OrchestratorService:
    def run(self, question: str, mode: str) -> InternalResponse:
        if mode == "mode_1_baseline_llm":
            return run_mode_1(question)
        if mode == "mode_2_rag_only":
            return run_mode_2(question)
        if mode == "mode_3_full_agentic":
            return run_mode_3(question)
        raise ValueError(f"Unsupported mode: {mode}")