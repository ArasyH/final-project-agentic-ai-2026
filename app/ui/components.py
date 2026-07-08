from __future__ import annotations
"""Render helper untuk Streamlit chat UI.

Modul ini memisahkan render logic 3 panel (sumber, critic, telemetri) supaya
`streamlit_app.py` tetap ringkas dan mudah dibaca.
"""

from typing import Any

import streamlit as st

from app.schemas import InternalResponse


HALLUCINATION_LABELS: dict[str, str] = {
    "H1": "H1 — Unsupported Numeric Claim",
    "H2": "H2 — Fabricated Financial Metric",
    "H3": "H3 — Stale Timestamp Misrepresentation",
    "H4": "H4 — Incorrect Inference",
}


def render_sources_panel(response: InternalResponse) -> None:
    """Panel expander yang menampilkan sumber & evidence yang dipakai jawaban."""
    n_sources = len(response.sources)
    n_evidence = len(response.evidence)

    with st.expander(f"Sumber & Evidence ({n_sources} sumber, {n_evidence} evidence)"):
        if n_sources == 0 and n_evidence == 0:
            st.caption("Tidak ada sumber yang di-retrieve untuk jawaban ini.")
            return

        if n_evidence > 0:
            st.markdown("**Evidence (potongan teks yang dipakai generator):**")
            for i, item in enumerate(response.evidence, start=1):
                source_ref = f" — `{item.source_id}`" if item.source_id else ""
                st.markdown(f"**[{i}]**{source_ref}")
                st.markdown(
                    f"> {item.content[:600]}{'...' if len(item.content) > 600 else ''}"
                )

        if n_sources > 0:
            st.markdown("**Sumber (metadata):**")
            for i, src in enumerate(response.sources, start=1):
                title = src.title or src.source_id or f"sumber-{i}"
                st.markdown(f"- `{src.source_id}` — {title}")
                if src.snippet:
                    st.caption(src.snippet[:240])


def render_critic_panel(response: InternalResponse) -> None:
    """Panel expander yang menampilkan verdict Critic + rationale per H1-H4."""
    verdict_str = response.metadata.get("critic_verdict", "n/a")
    flags = response.hallucination_flags
    critic_details: dict[str, Any] = response.metadata.get("critic_details", {})
    guardrails_details: dict[str, Any] = response.metadata.get("guardrails_details", {})

    header = f"Verdict Critic: {verdict_str.upper()}"
    if flags:
        header += f" — flag: {', '.join(flags)}"

    with st.expander(header):
        if not critic_details and not guardrails_details:
            st.caption(
                "Mode ini tidak menjalankan Critic/Guardrails "
                "(hanya mode_3_rag_jc & mode_4_rag_jc_cache)."
            )
            return

        st.markdown(f"**Overall verdict:** `{verdict_str}`")
        st.markdown(f"**Validator status:** `{response.validator_status}`")
        st.markdown("---")

        # H1-H4: gabungkan critic + guardrails untuk H1 & H3
        for code in ("H1", "H2", "H3", "H4"):
            label = HALLUCINATION_LABELS[code]
            critic_row = critic_details.get(code, {})
            critic_flag = critic_row.get("flag")
            critic_rationale = critic_row.get("rationale", "")

            icon = "⚠️" if code in flags else "✅"
            st.markdown(f"**{icon} {label}**")

            if critic_flag is not None:
                st.markdown(
                    f"- Critic: `flag={critic_flag}` — {critic_rationale or '_no rationale_'}"
                )

            # H1 & H3 punya guardrails deterministik
            if code in ("H1", "H3"):
                guard_key = (
                    "H1" if code == "H1" else "H3"
                )
                guard_row = guardrails_details.get(guard_key, {})
                if guard_row:
                    guard_flag = guard_row.get("flag")
                    guard_rationale = guard_row.get("rationale", "")
                    st.markdown(
                        f"- Guardrails (deterministik): `flag={guard_flag}` — {guard_rationale}"
                    )
            st.markdown("")


def render_telemetry_panel(response: InternalResponse, wall_latency_ms: float) -> None:
    """Panel expander yang menampilkan metrik latency, cache, evidence, confidence."""
    with st.expander("Telemetri"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Latency total", f"{wall_latency_ms:.0f} ms")
        with col2:
            st.metric("Cache", response.cache_status)
        with col3:
            st.metric("Evidence", len(response.evidence))
        with col4:
            st.metric("Confidence", f"{response.confidence:.2f}")

        st.markdown("---")
        st.markdown(f"**Mode:** `{response.mode}`")
        st.markdown(f"**Timestamp:** `{response.timestamp}`")
        st.markdown(f"**Validator status:** `{response.validator_status}`")

        if response.tickers:
            st.markdown(f"**Tickers terdeteksi:** {', '.join(response.tickers)}")

        extra_keys = {
            k: v
            for k, v in response.metadata.items()
            if k not in {"critic_details", "guardrails_details"}
        }
        if extra_keys:
            st.markdown("**Metadata tambahan:**")
            st.json(extra_keys, expanded=False)
