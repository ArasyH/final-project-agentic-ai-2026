from __future__ import annotations
"""Generator Agent — manual ReAct loop dengan Llama-3.1-8B-Instant (Groq).

Implementasi: manual Thought→Action→Observation loop tanpa LangChain agent
abstraction. Lebih transparent dan reproducible untuk pelaporan metodologi SINTA 2.
Satu tool: retrieve_from_kb (wraps RetrievalService).

Prompt version: REACT_PROMPT_V1 (Bahasa Indonesia).
Max iterasi: REACT_MAX_ITERATIONS dari app/config.py (default 5).
"""
import re
import time
from typing import Any

import groq
from pydantic import BaseModel

from app.config import REACT_MAX_ITERATIONS
from app.schemas import EvidenceItem
from app.services.llm_service import build_generator_llm
from app.services.retrieval_service import RetrievalService
from app.services.telemetry_service import TelemetryService

GENERATOR_TEMPERATURE: float = 0.0
PROMPT_VERSION: str = "react_v1"

FAILSAFE_ANSWER: str = (
    "Maaf, sistem tidak dapat menghasilkan jawaban saat ini. "
    "Silakan ulangi pertanyaan atau hubungi administrator."
)

# Prompt Bahasa Indonesia — versioned, jangan ganti in-place (tambah _v2, _v3)
REACT_PROMPT_V1 = """Anda adalah analis pasar saham Indonesia (IDX30).
Jawab HANYA berdasarkan evidence dari knowledge base. Jangan mengarang angka, harga, atau fakta.

Tool tersedia:
- retrieve_from_kb: mengambil data relevan dari knowledge base IDX30

Format WAJIB — satu langkah per respons:
Pikiran: [analisis singkat tentang informasi apa yang dibutuhkan]
Aksi: retrieve_from_kb
Input Aksi: [query singkat untuk knowledge base]

Jika sudah cukup evidence, gunakan:
Pikiran: Saya sudah memiliki informasi yang cukup untuk menjawab.
Jawaban Final: [jawaban berdasarkan evidence saja; sertakan tanggal/periode data]

Pertanyaan: {question}

{scratchpad}"""


# ── Output schema (internal, tidak masuk schemas.py) ────────────────────────

class GeneratorOutput(BaseModel):
    """Output internal GeneratorAgent — tidak diekspor ke schemas.py.

    Digunakan hanya untuk komunikasi antara GeneratorAgent dan mode runner.

    Attributes:
        answer: teks jawaban final atau fail-safe message.
        evidence: list EvidenceItem yang di-retrieve selama loop.
        iterations_used: jumlah iterasi ReAct yang berjalan.
        retrieval_latency_ms: akumulasi wall-clock ms semua retrieve_from_kb calls
            dalam satu generate() — dipakai pipeline untuk memisahkan
            latency_ms_retrieval dari latency_ms_generation di telemetry.
        error: None jika berhasil; pesan error singkat jika gagal.
    """

    answer: str
    evidence: list[EvidenceItem]
    iterations_used: int
    retrieval_latency_ms: float = 0.0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """True jika generate selesai tanpa error."""
        return self.error is None


# ── Agent class ──────────────────────────────────────────────────────────────

class GeneratorAgent:
    """LLM generator dengan manual ReAct loop (Thought→Action→Observation).

    Hanya satu tool: retrieve_from_kb (wraps RetrievalService).

    Args:
        retrieval_service: instance RetrievalService untuk KB lookup.
        telemetry_service: instance TelemetryService untuk Langfuse span per iterasi.
        llm: optional ChatGroq override untuk dependency injection / testing.
            Jika None, dibuat via build_generator_llm() dengan temperature=0.0.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        telemetry_service: TelemetryService,
        llm: Any = None,
    ) -> None:
        self._retriever = retrieval_service
        self._telemetry = telemetry_service
        self._llm = llm or build_generator_llm(temperature=GENERATOR_TEMPERATURE)
        self.prompt_version: str = PROMPT_VERSION
        # Accumulator untuk retrieval latency — di-reset tiap generate() call.
        # Catatan thread-safety: instance ini tidak thread-safe untuk concurrent
        # generate() calls. Gunakan satu instance per request.
        self._current_retrieval_latency_ms: float = 0.0

    # ── public ───────────────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        session_id: str,
        tickers: list[str] | None = None,
    ) -> GeneratorOutput:
        """Jalankan ReAct loop dan kembalikan jawaban + evidence.

        Loop berhenti jika LLM menghasilkan "Jawaban Final:" atau max iterasi
        tercapai. Setiap thought/action/observation di-trace ke Langfuse.

        Args:
            question: pertanyaan asli pengguna.
            session_id: ID sesi untuk Langfuse trace span.
            tickers: daftar ticker IDX30 yang terdeteksi dari query (opsional).
                Diteruskan ke RetrievalService untuk filter metadata KB.

        Returns:
            GeneratorOutput. Jika error, field `error` non-None dan `answer`
            berisi FAILSAFE_ANSWER. `succeeded` property bisa dipakai mode runner
            untuk memutuskan apakah perlu fallback.
        """
        trace = self._telemetry.start_trace(
            session_id=session_id,
            question=question,
            mode="generator_react",
        )

        scratchpad = ""
        all_evidence: list[EvidenceItem] = []
        self._current_retrieval_latency_ms = 0.0

        for iteration in range(1, REACT_MAX_ITERATIONS + 1):
            prompt = REACT_PROMPT_V1.format(question=question, scratchpad=scratchpad)

            try:
                raw = self._llm.invoke(prompt).content
            except (groq.APIError, ConnectionError) as exc:
                self._telemetry.event(
                    trace,
                    name=f"llm_error_iter{iteration}",
                    metadata={"error": type(exc).__name__, "detail": str(exc)[:300]},
                )
                return GeneratorOutput(
                    answer=FAILSAFE_ANSWER,
                    evidence=all_evidence,
                    iterations_used=iteration,
                    retrieval_latency_ms=self._current_retrieval_latency_ms,
                    error=f"llm_error: {type(exc).__name__}",
                )

            # ── Final answer branch ──────────────────────────────────────
            if "Jawaban Final:" in raw:
                answer = self._extract_final_answer(raw)
                self._telemetry.event(
                    trace,
                    name=f"final_answer_iter{iteration}",
                    input_data=question,
                    output_data=answer,
                    metadata={
                        "iterations_used": iteration,
                        "evidence_count": len(all_evidence),
                        "prompt_version": PROMPT_VERSION,
                    },
                )
                return GeneratorOutput(
                    answer=answer,
                    evidence=all_evidence,
                    iterations_used=iteration,
                    retrieval_latency_ms=self._current_retrieval_latency_ms,
                )

            # ── ReAct step branch ────────────────────────────────────────
            try:
                thought, action_input = self._parse_react_step(raw)
            except ValueError as exc:
                self._telemetry.event(
                    trace,
                    name=f"parse_error_iter{iteration}",
                    metadata={"raw_snippet": raw[:300], "error": str(exc)},
                )
                return GeneratorOutput(
                    answer=FAILSAFE_ANSWER,
                    evidence=all_evidence,
                    iterations_used=iteration,
                    retrieval_latency_ms=self._current_retrieval_latency_ms,
                    error=f"parse_error: {exc}",
                )

            self._telemetry.event(
                trace,
                name=f"thought_iter{iteration}",
                output_data=thought,
            )

            observation, retrieved = self._retrieve(action_input, tickers=tickers)
            all_evidence.extend(retrieved)

            self._telemetry.event(
                trace,
                name=f"observation_iter{iteration}",
                input_data=action_input,
                output_data=observation[:500],
                metadata={"evidence_count": len(retrieved)},
            )

            scratchpad += (
                f"Pikiran: {thought}\n"
                f"Aksi: retrieve_from_kb\n"
                f"Input Aksi: {action_input}\n"
                f"Observasi: {observation}\n\n"
            )

        # Max iterasi tercapai tanpa Jawaban Final
        self._telemetry.event(
            trace,
            name="max_iterations_reached",
            metadata={
                "max_iterations": REACT_MAX_ITERATIONS,
                "evidence_count": len(all_evidence),
            },
        )
        return GeneratorOutput(
            answer=FAILSAFE_ANSWER,
            evidence=all_evidence,
            iterations_used=REACT_MAX_ITERATIONS,
            retrieval_latency_ms=self._current_retrieval_latency_ms,
            error="max_iterations_reached",
        )

    # ── private helpers ──────────────────────────────────────────────────────

    def _retrieve(
        self,
        query: str,
        tickers: list[str] | None = None,
    ) -> tuple[str, list[EvidenceItem]]:
        """Panggil RetrievalService dan format sebagai teks observasi.

        Waktu KB lookup di-akumulasikan ke `_current_retrieval_latency_ms`
        via try/finally agar tercatat bahkan jika retrieve() melempar exception.

        Args:
            query: query string untuk KB lookup.
            tickers: ticker filter untuk metadata retrieval (opsional).

        Returns:
            Tuple (observation_string, list_of_EvidenceItem).
        """
        _start = time.perf_counter()
        try:
            docs = self._retriever.retrieve(query, tickers=tickers)
        finally:
            self._current_retrieval_latency_ms += (time.perf_counter() - _start) * 1000.0
        if not docs:
            return "(tidak ada data relevan di knowledge base)", []

        evidence_items = [
            EvidenceItem(content=doc.page_content, source_id=f"kb_{i}")
            for i, doc in enumerate(docs)
        ]
        observation = "\n---\n".join(
            f"[kb_{i}] {doc.page_content[:400]}" for i, doc in enumerate(docs)
        )
        return observation, evidence_items

    def _parse_react_step(self, raw: str) -> tuple[str, str]:
        """Parse satu langkah ReAct: ekstrak Pikiran dan Input Aksi.

        Karena hanya ada satu tool (retrieve_from_kb), validasi nama aksi
        tidak dilakukan — semua "Aksi:" diasumsikan retrieve_from_kb.

        Args:
            raw: teks mentah dari LLM satu iterasi.

        Returns:
            Tuple (thought, action_input).

        Raises:
            ValueError: jika "Pikiran:" atau "Input Aksi:" tidak ditemukan.
        """
        thought_match = re.search(
            r"Pikiran:\s*(.+?)(?=\nAksi:|\nJawaban Final:|$)", raw, re.DOTALL
        )
        input_match = re.search(
            r"Input Aksi:\s*(.+?)(?=\nObservasi:|$)", raw, re.DOTALL
        )

        if not thought_match:
            raise ValueError(f"'Pikiran:' tidak ditemukan: {raw[:200]!r}")
        if not input_match:
            raise ValueError(f"'Input Aksi:' tidak ditemukan: {raw[:200]!r}")

        return thought_match.group(1).strip(), input_match.group(1).strip()

    def _extract_final_answer(self, raw: str) -> str:
        """Ekstrak teks setelah 'Jawaban Final:'.

        Args:
            raw: teks mentah dari LLM yang mengandung 'Jawaban Final:'.

        Returns:
            String jawaban final, di-strip.
        """
        match = re.search(r"Jawaban Final:\s*(.+)", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        parts = raw.split("Jawaban Final:", 1)
        return parts[1].strip() if len(parts) > 1 else raw.strip()
