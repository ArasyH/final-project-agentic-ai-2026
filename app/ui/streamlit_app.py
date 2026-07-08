from __future__ import annotations

import sys
from pathlib import Path

# Pastikan root project (parent dari app/) ada di sys.path supaya
# `streamlit run app/ui/streamlit_app.py` bisa import `app.*`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import time
import uuid
from typing import Any

import streamlit as st

from app.schemas import ExperimentMode, InternalResponse
from app.ui.components import (
    render_critic_panel,
    render_sources_panel,
    render_telemetry_panel,
)
from app.ui.kb_bootstrap import ensure_kb_ready, kb_document_count


MODE_LABELS: dict[str, ExperimentMode] = {
    "Mode 1 — LLM only": "mode_1_llm_only",
    "Mode 2 — RAG only": "mode_2_rag_only",
    "Mode 3 — RAG + Judge & Critic": "mode_3_rag_jc",
    "Mode 4 — RAG + J&C + Semantic Cache": "mode_4_rag_jc_cache",
}
DEFAULT_MODE_LABEL = "Mode 4 — RAG + J&C + Semantic Cache"


@st.cache_resource(show_spinner=False)
def _get_orchestrator() -> Any:
    """Init OrchestratorService sekali per container lifetime."""
    from app.services.orchestrator_service import OrchestratorService
    return OrchestratorService()


@st.cache_resource(show_spinner=False)
def _bootstrap_kb() -> tuple[bool, str]:
    """Cek KB & jalankan ETL kalau kosong. Cache di level container."""
    return ensure_kb_ready(on_progress=lambda msg: None)


def _init_session_state() -> None:
    """Inisialisasi state chat, session_id, mode terpilih."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "mode_label" not in st.session_state:
        st.session_state.mode_label = DEFAULT_MODE_LABEL


def _render_sidebar() -> ExperimentMode:
    """Render sidebar dan return mode terpilih."""
    with st.sidebar:
        st.title("Konfigurasi")

        st.subheader("Mode eksperimen")
        selected_label = st.radio(
            label="Pilih mode",
            options=list(MODE_LABELS.keys()),
            index=list(MODE_LABELS.keys()).index(st.session_state.mode_label),
            label_visibility="collapsed",
            key="mode_radio",
        )
        st.session_state.mode_label = selected_label

        st.caption(_mode_description(MODE_LABELS[selected_label]))

        st.markdown("---")
        st.subheader("Sesi")
        st.code(st.session_state.session_id, language="text")
        st.caption(f"{len(st.session_state.messages)} pesan di riwayat.")

        if st.button("Reset chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

        st.markdown("---")
        st.subheader("Knowledge Base")
        kb_count = kb_document_count()
        st.metric("Dokumen di KB", kb_count)

    return MODE_LABELS[selected_label]


def _mode_description(mode: ExperimentMode) -> str:
    if mode == "mode_1_llm_only":
        return "LLM saja, tanpa retrieval / cache / critic. Baseline."
    if mode == "mode_2_rag_only":
        return "Retrieval top-3 + LLM. Tanpa validator."
    if mode == "mode_3_rag_jc":
        return "RAG + Generator ReAct + Domain Guardrails + Critic Agent."
    if mode == "mode_4_rag_jc_cache":
        return "Mode 3 + Semantic Cache (threshold 0.85, TTL 8 jam)."
    return ""


def _render_chat_history() -> None:
    """Render pesan-pesan yang sudah ada di st.session_state.messages."""
    for entry in st.session_state.messages:
        role = entry["role"]
        with st.chat_message(role):
            st.markdown(entry["content"])
            if role == "assistant" and "response" in entry:
                _render_response_panels(entry["response"], entry["wall_latency_ms"])


def _render_response_panels(response: InternalResponse, wall_latency_ms: float) -> None:
    render_sources_panel(response)
    render_critic_panel(response)
    render_telemetry_panel(response, wall_latency_ms)


def _handle_user_input(mode: ExperimentMode) -> None:
    """Ambil input user, panggil orchestrator, tampilkan balasan."""
    user_input = st.chat_input("Tanyakan tentang saham IDX30...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Memproses..._")

        orchestrator = _get_orchestrator()
        question_id = f"UI-{uuid.uuid4().hex[:8]}"
        started = time.perf_counter()

        try:
            response = orchestrator.run(
                question=user_input,
                session_id=st.session_state.session_id,
                question_id=question_id,
                mode=mode,
            )
        except Exception as exc:
            placeholder.error(f"Terjadi error: {type(exc).__name__}: {exc}")
            return

        wall_latency_ms = (time.perf_counter() - started) * 1000
        placeholder.markdown(response.answer)
        _render_response_panels(response, wall_latency_ms)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "response": response,
        "wall_latency_ms": wall_latency_ms,
    })


def main() -> None:
    st.set_page_config(
        page_title="Agentic AI IDX30 — Demo",
        page_icon="📈",
        layout="wide",
    )

    _init_session_state()

    st.title("Agentic AI untuk Saham IDX30")
    st.caption(
        "Demo sistem RAG + Judge & Critic + Semantic Cache. "
        "Pilih mode eksperimen di sidebar untuk membandingkan perilaku."
    )

    with st.spinner("Mengecek knowledge base..."):
        ok, msg = _bootstrap_kb()
    if not ok:
        st.error(
            f"Bootstrap KB gagal: `{msg}`. "
            "Pastikan `SECTORS_API_KEY` dan `CHROMA_DB_PATH` sudah diset, "
            "atau jalankan `python -m etl.run` di lingkungan lokal terlebih dulu."
        )
        st.stop()

    mode = _render_sidebar()
    _render_chat_history()
    _handle_user_input(mode)


if __name__ == "__main__":
    main()
