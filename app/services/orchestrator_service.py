from __future__ import annotations
# app/services/orchestrator_service.py
from app.schemas import ExperimentMode, InternalResponse
from app.modes.mode_1_baseline_llm import run_mode_1
from app.modes.mode_2_rag_only import run_mode_2


class OrchestratorService:
    """Routes pertanyaan ke mode runner yang sesuai berdasarkan ExperimentMode.

    session_id diterima di sini untuk keperluan Langfuse trace (task #10).
    Propagasi ke runner individual ditunda sampai signature runner diupdate (task #5).
    """

    def run(self, question: str, session_id: str, mode: ExperimentMode) -> InternalResponse:
        """Jalankan mode eksperimen yang ditentukan.

        Args:
            question: pertanyaan asli pengguna.
            session_id: ID sesi untuk Langfuse trace; belum dipropagasikan ke
                runner sampai task #5 selesai.
            mode: salah satu dari 4 ExperimentMode (Literal).

        Returns:
            InternalResponse dari mode runner.

        Raises:
            NotImplementedError: mode_3_rag_jc dan mode_4_rag_jc_cache belum
                diimplementasi.
        """
        if mode == "mode_1_llm_only":
            # run_mode_1 signature belum terima session_id — diupdate di task #5
            return run_mode_1(question)
        if mode == "mode_2_rag_only":
            # run_mode_2 signature belum terima session_id — diupdate di task #5
            return run_mode_2(question)
        if mode == "mode_3_rag_jc":
            raise NotImplementedError("mode_3_rag_jc belum diimplementasi — task #8")
        if mode == "mode_4_rag_jc_cache":
            raise NotImplementedError("mode_4_rag_jc_cache belum diimplementasi — task #9")
        raise ValueError(f"Unsupported mode: {mode}")
