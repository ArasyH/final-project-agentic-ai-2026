from __future__ import annotations
# app/services/telemetry_service.py
import time
from contextlib import contextmanager
from typing import Any, Generator

from langfuse import Langfuse

from app.config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY


class TelemetryService:
    """Langfuse wrapper dengan latency measurement per pipeline stage.

    Pola pemakaian:
        trace = telemetry.start_trace(session_id, question, mode, question_id)
        with telemetry.measure_latency(trace, "total"):
            with telemetry.measure_latency(trace, "generation"):
                ...
            telemetry._record_latency(trace, "retrieval", 0.0)  # stage skipped
        telemetry.end_trace(trace, metadata={...})

    Latency dikumpulkan in-memory per trace (keyed by id(trace_handle)),
    di-flush ke Langfuse oleh end_trace().
    """

    def __init__(self) -> None:
        self.client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        self._latencies: dict[int, dict[str, float]] = {}

    def start_trace(
        self,
        session_id: str,
        question: str,
        mode: str,
        question_id: str = "",
    ) -> Any:
        """Buat Langfuse trace baru.

        Args:
            session_id: ID sesi pengguna.
            question: pertanyaan asli (masuk sebagai trace input).
            mode: ExperimentMode string untuk metadata + tag.
            question_id: ID pertanyaan untuk paired comparison lintas mode.

        Returns:
            StatefulTraceClient dari Langfuse SDK.
        """
        return self.client.trace(
            name="chat-query",
            session_id=session_id,
            input=question,
            metadata={"mode": mode, "question_id": question_id},
            tags=["idx30", mode],
        )

    def event(
        self,
        trace: Any,
        name: str,
        metadata: dict | None = None,
        input_data: Any = None,
        output_data: Any = None,
    ) -> None:
        """Log satu event ke trace.

        Args:
            trace: Langfuse trace object dari start_trace().
            name: nama event (e.g. "cache_hit", "generator_failed").
            metadata: metadata tambahan untuk event ini.
            input_data: input event (opsional).
            output_data: output event (opsional).
        """
        trace.event(
            name=name,
            input=input_data,
            output=output_data,
            metadata=metadata or {},
        )

    @contextmanager
    def measure_latency(
        self, trace_handle: Any, stage_name: str
    ) -> Generator[None, None, None]:
        """Context manager: ukur wall-clock duration (ms) satu pipeline stage.

        Hasil ukuran di-accumulate di `_latencies`; di-flush ke Langfuse
        saat end_trace() dipanggil.

        Args:
            trace_handle: Langfuse trace object dari start_trace().
            stage_name: label stage ("total", "retrieval", "generation", "critic").
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self._record_latency(trace_handle, stage_name, duration_ms)

    def _record_latency(
        self, trace_handle: Any, stage_name: str, duration_ms: float
    ) -> None:
        """Simpan latency untuk satu stage secara langsung (tanpa context manager).

        Dipakai untuk stage yang di-skip (nilai 0.0) agar semua 4 field
        selalu hadir di end_trace metadata.

        Args:
            trace_handle: Langfuse trace object dari start_trace().
            stage_name: label stage.
            duration_ms: durasi dalam milidetik; 0.0 untuk stage yang di-skip.
        """
        key = id(trace_handle)
        if key not in self._latencies:
            self._latencies[key] = {}
        self._latencies[key][stage_name] = duration_ms

    def end_trace(self, trace_handle: Any, metadata: dict) -> None:
        """Finalisasi trace: merge latency measurements ke metadata, update Langfuse.

        Metadata yang di-update mencakup 11 field wajib §15:
        mode, question_id, cache_status, validator_status, latency_ms_total,
        latency_ms_retrieval, latency_ms_generation, latency_ms_critic,
        hallucination_flags, evidence_count, confidence.

        Args:
            trace_handle: Langfuse trace object dari start_trace().
            metadata: dict dengan field wajib §15 (kecuali latency_ms_* yang
                diisi dari _latencies).
        """
        key = id(trace_handle)
        latencies = self._latencies.pop(key, {})
        full_metadata = {
            **metadata,
            "latency_ms_total": latencies.get("total", 0.0),
            "latency_ms_retrieval": latencies.get("retrieval", 0.0),
            "latency_ms_generation": latencies.get("generation", 0.0),
            "latency_ms_critic": latencies.get("critic", 0.0),
        }
        trace_handle.update(metadata=full_metadata)

    def flush(self) -> None:
        """Flush semua pending events ke Langfuse (panggil saat shutdown)."""
        self.client.flush()
