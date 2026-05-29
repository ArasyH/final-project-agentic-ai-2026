from __future__ import annotations
# app/services/telemetry_service.py
import time
from contextlib import contextmanager
from typing import Any, Generator

from langfuse import Langfuse

from app.config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY


class TelemetryService:
    """Langfuse v3 wrapper dengan latency measurement per pipeline stage.

    Langfuse v3 (>= 3.0) adalah OTEL-native. API `langfuse.trace()` dari v2
    sudah dihapus. Pengganti yang dipakai di sini:
      - `langfuse.start_span()` → buat root span (= trace baru jika tidak ada
        OTEL context aktif). Perlu `span.end()` saat selesai.
      - `span.update_trace()` → set session_id dan tags di trace-level.
      - `span.create_event()` → child event di bawah span.
      - `span.update(metadata=...)` → update metadata span sebelum di-end.

    API eksternal (start_trace, end_trace, event, measure_latency,
    _record_latency, flush) tidak berubah — mode runner tidak perlu dimodifikasi.

    Pola pemakaian:
        trace = telemetry.start_trace(session_id, question, mode, question_id)
        with telemetry.measure_latency(trace, "total"):
            with telemetry.measure_latency(trace, "generation"):
                ...
            telemetry._record_latency(trace, "retrieval", 0.0)
        telemetry.end_trace(trace, metadata={...})
    """

    def __init__(self) -> None:
        self._client = Langfuse(
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
        """Buat Langfuse v3 root span (= trace baru).

        Berbeda dari v2 `langfuse.trace()`, v3 menggunakan `start_span()` yang
        mengembalikan `LangfuseSpan`. Span ini menjadi root observation dari trace
        baru (karena tidak ada OTEL context aktif saat request masuk).

        Args:
            session_id: ID sesi pengguna — diset via `span.update_trace()`.
            question: pertanyaan asli (masuk sebagai span input + trace-level input).
            mode: ExperimentMode string untuk metadata + tag.
            question_id: ID pertanyaan untuk paired comparison lintas mode.

        Returns:
            `LangfuseSpan` object. Caller bertanggung jawab memanggil
            `end_trace()` untuk menutupnya.
        """
        span = self._client.start_span(
            name="chat-query",
            input=question,
            metadata={"mode": mode, "question_id": question_id},
        )
        span.update_trace(
            session_id=session_id,
            tags=["idx30", mode],
            input=question,
        )
        return span

    def event(
        self,
        trace: Any,
        name: str,
        metadata: dict | None = None,
        input_data: Any = None,
        output_data: Any = None,
    ) -> None:
        """Log satu child event di bawah trace span.

        v3: `span.create_event()` menggantikan `trace.event()` dari v2.

        Args:
            trace: LangfuseSpan dari start_trace().
            name: nama event (e.g. "cache_hit", "generator_failed").
            metadata: metadata tambahan.
            input_data: input event (opsional).
            output_data: output event (opsional).
        """
        trace.create_event(
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
            trace_handle: LangfuseSpan dari start_trace().
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
        """Simpan latency untuk satu stage secara langsung.

        Dipakai untuk stage yang di-skip (nilai 0.0) agar semua 4 field
        selalu hadir di end_trace metadata.

        Args:
            trace_handle: LangfuseSpan dari start_trace().
            stage_name: label stage.
            duration_ms: durasi dalam milidetik; 0.0 untuk stage yang di-skip.
        """
        key = id(trace_handle)
        if key not in self._latencies:
            self._latencies[key] = {}
        self._latencies[key][stage_name] = duration_ms

    def end_trace(self, trace_handle: Any, metadata: dict, output: str = "") -> None:
        """Finalisasi trace: set output, merge latency, update span metadata, end span.

        v3: `span.update_trace(output=...)` untuk trace-level output (tampil di UI),
        lalu `span.update(metadata=...)` + `span.end()`.

        Metadata yang di-update mencakup 11 field wajib §15:
        mode, question_id, cache_status, validator_status, latency_ms_total,
        latency_ms_retrieval, latency_ms_generation, latency_ms_critic,
        hallucination_flags, evidence_count, confidence.

        Args:
            trace_handle: LangfuseSpan dari start_trace().
            metadata: dict dengan field wajib §15 (kecuali latency_ms_* yang
                diisi dari _latencies).
            output: jawaban final (result.answer) — diset sebagai trace-level output
                agar tampil di Langfuse UI dan tersedia untuk RAGAS evaluation.
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
        if output:
            trace_handle.update_trace(output=output)
        trace_handle.update(metadata=full_metadata)
        trace_handle.end()

    def flush(self) -> None:
        """Flush semua pending OTEL spans ke Langfuse backend.

        Penting dipanggil saat FastAPI shutdown agar tidak ada trace yang hilang.
        """
        self._client.flush()
