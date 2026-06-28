from __future__ import annotations
"""Generator Agent — manual ReAct loop dengan Llama-3.1-8B-Instant (Groq).

Implementasi: manual Thought→Action→Observation loop tanpa LangChain agent
abstraction. Lebih transparent dan reproducible untuk pelaporan metodologi SINTA 2.
Satu tool: retrieve_from_kb (wraps RetrievalService).

Events di-log ke trace_handle yang diterima dari caller (mode runner) — tidak ada
sub-trace tersendiri. 1 request = 1 root trace bersih di Langfuse UI.

Prompt versions:
  REACT_PROMPT_V1 — dipakai eksperimen 50Q (data final, JANGAN dihapus).
  REACT_PROMPT_V2 — nonaktif: few-shot complete-trace menyebabkan model skip
                    retrieve_from_kb (evidence_count=0, halluc rate 100%).
  REACT_PROMPT_V3 — nonaktif: {fundamental_context} menyebabkan model merasa
                    sudah punya domain knowledge → skip retrieve (evidence=0 76%).
                    Klausul "akui keterbatasan" dieksploitasi sebagai escape hatch
                    tanpa pernah memanggil retrieve_from_kb.
  REACT_PROMPT_V4 — nonaktif: menghapus {fundamental_context} membuat model tidak
                    punya alasan apapun untuk retrieve → ev=0 naik ke 98%.
  REACT_PROMPT_V5 — aktif (C2): basis V1 + satu baris ticker hint + satu kalimat
                    timestamp di Jawaban Final. Sesederhana V1 agar Llama-3.1-8B
                    mengikuti format multi-turn.

Max iterasi: REACT_MAX_ITERATIONS dari app/config.py (default 5).
Scratchpad trim: REACT_MAX_SCRATCHPAD_CHARS dari app/config.py (default 3000).
"""
import re
import time
from typing import Any

import groq
from pydantic import BaseModel

from app.config import REACT_MAX_ITERATIONS, REACT_MAX_SCRATCHPAD_CHARS
from app.schemas import EvidenceItem
from app.services.llm_service import build_generator_llm
from app.services.retrieval_service import RetrievalService
from app.services.telemetry_service import TelemetryService

GENERATOR_TEMPERATURE: float = 0.0
PROMPT_VERSION: str = "react_v5"

FAILSAFE_ANSWER: str = (
    "Maaf, sistem tidak dapat menghasilkan jawaban saat ini. "
    "Silakan ulangi pertanyaan atau hubungi administrator."
)

# ── Prompt V1 — JANGAN dihapus; dipakai eksperimen 50Q (data sudah final) ─────
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

# ── Fundamental Knowledge — injeksi konteks domain IDX30 ─────────────────────
# Versi ini berlaku untuk Generator dan tersedia sebagai konstanta terpisah
# supaya mudah diperbarui tanpa menyentuh template prompt.
FUNDAMENTAL_KNOWLEDGE_V1 = (
    "Metrik fundamental IDX30 yang dapat dicari di knowledge base:\n"
    "  PER (Price-to-Earnings Ratio), PBV (Price-to-Book Value),\n"
    "  ROE (Return on Equity), EPS (Earnings per Share),\n"
    "  DER (Debt-to-Equity Ratio), NPM (Net Profit Margin),\n"
    "  Revenue/Pendapatan, Laba Bersih/Net Income,\n"
    "  Market Cap/Kapitalisasi Pasar, Dividend Yield.\n"
    "Sertakan nama metrik secara eksplisit di Input Aksi jika pertanyaan terkait fundamental."
)

# ── Prompt V2 — NONAKTIF (bug: complete-trace example menyebabkan skip retrieve) ──
# Root cause: few-shot menampilkan full trace termasuk Jawaban Final, sehingga
# Llama-3.1-8B shortcut ke Jawaban Final tanpa retrieve_from_kb (evidence_count=0).
# JANGAN dihapus — dipertahankan sebagai catatan regresi.
REACT_PROMPT_V2 = """Anda adalah analis pasar saham Indonesia dengan spesialisasi IDX30.
Jawab HANYA berdasarkan evidence dari knowledge base. Jangan mengarang angka, harga, atau fakta.

{ticker_context}

Referensi metrik fundamental (gunakan untuk menyusun query KB yang tepat):
{fundamental_context}

Tool tersedia:
- retrieve_from_kb: mengambil data relevan dari knowledge base IDX30

Format WAJIB per langkah:
Pikiran: [analisis singkat informasi apa yang dibutuhkan]
Aksi: retrieve_from_kb
Input Aksi: [query spesifik — cantumkan ticker/nama saham + metrik yang dicari]

Jika evidence sudah cukup, gunakan:
Pikiran: Saya sudah memiliki informasi yang cukup untuk menjawab.
Jawaban Final: [jawaban berdasarkan evidence]. Data per [tanggal/periode dari evidence]. [Jika ada keterbatasan data, sebutkan secara eksplisit.]

Contoh alur yang benar:
Pertanyaan: Berapa harga penutupan BBCA?
Pikiran: Perlu data harga penutupan BBCA dari knowledge base.
Aksi: retrieve_from_kb
Input Aksi: harga penutupan BBCA closing price terkini
Observasi: [kb_0] BBCA: harga penutupan Rp 9.200 per saham, 2026-06-15. Volume 45,2 juta lot.
Pikiran: Saya sudah memiliki informasi yang cukup untuk menjawab.
Jawaban Final: Harga penutupan BBCA adalah Rp 9.200 per saham. Data per 15 Juni 2026.

Pertanyaan: {question}

{scratchpad}"""

# ── Prompt V3 — NONAKTIF ──────────────────────────────────────────────────────
# Root cause kegagalan V3:
#   - {fundamental_context} menyediakan nama metrik IDX30 langsung di prompt →
#     model merasa punya domain knowledge cukup → skip retrieve_from_kb (76% ev=0).
#   - Klausul "akui keterbatasan" dieksploitasi: model menulis "tidak tersedia"
#     TANPA pernah memanggil retrieve_from_kb terlebih dahulu.
# JANGAN dihapus — dipertahankan sebagai catatan regresi.
REACT_PROMPT_V3 = """Anda adalah analis pasar saham Indonesia dengan spesialisasi IDX30.
Jawab HANYA berdasarkan evidence dari knowledge base. Jangan mengarang angka, harga, atau fakta.

{ticker_context}

Referensi metrik fundamental (gunakan untuk menyusun query KB yang tepat):
{fundamental_context}

Tool tersedia:
- retrieve_from_kb: mengambil data relevan dari knowledge base IDX30

ATURAN WAJIB:
1. SELALU mulai dengan Pikiran → Aksi → Input Aksi. JANGAN langsung menulis Jawaban Final.
2. Jawaban Final HANYA boleh ditulis SETELAH menerima minimal satu Observasi dari retrieve_from_kb.
3. Jika evidence tidak ditemukan, akui keterbatasan — jangan mengarang fakta.

Format per langkah retrieve:
Pikiran: [analisis singkat informasi apa yang dibutuhkan]
Aksi: retrieve_from_kb
Input Aksi: [query spesifik — cantumkan ticker/nama saham + metrik yang dicari]

Setelah mendapat Observasi dan evidence sudah cukup:
Pikiran: Saya sudah memiliki informasi yang cukup untuk menjawab.
Jawaban Final: [jawaban berdasarkan evidence]. Data per [tanggal/periode dari evidence]. [Jika ada keterbatasan data, sebutkan secara eksplisit.]

Contoh langkah pertama (sistem akan memberi Observasi — jangan tulis Observasi sendiri):
Pertanyaan: Berapa harga penutupan BBCA?
Pikiran: Saya perlu mencari data harga penutupan BBCA di knowledge base.
Aksi: retrieve_from_kb
Input Aksi: harga penutupan BBCA closing price terkini

Pertanyaan: {question}

{scratchpad}"""

# ── Prompt V4 — NONAKTIF ──────────────────────────────────────────────────────
# Root cause kegagalan V4: menghapus fundamental_context dari V3 membuat model
# tidak punya konten apapun → langsung menulis "tidak tersedia" tanpa retrieve.
# ev=0 naik dari 76% (V3) menjadi 98% (V4). JANGAN dihapus — catatan regresi.
# Fix A1 vs V3:
#   - Hapus {fundamental_context}: tanpa domain knowledge bawaan, model WAJIB
#     memanggil retrieve_from_kb untuk mendapat informasi apapun.
#   - Perketat aturan "tidak tersedia": frasa ini HANYA boleh ditulis setelah
#     ada Observasi (bukan sebagai escape hatch sebelum retrieve).
#   - Ticker_context dipertahankan: tidak menyediakan nilai/metrik, hanya
#     memberi arah pencarian — tidak mensubstitusi retrieval.
#   - One-step example dipertahankan (tidak tampilkan Observasi/Jawaban Final
#     di contoh agar model tidak shortcut ke Jawaban Final).
REACT_PROMPT_V4 = """Anda adalah analis pasar saham Indonesia dengan spesialisasi IDX30.
Jawab HANYA berdasarkan evidence yang Anda peroleh dari knowledge base melalui retrieve_from_kb.
Jangan mengarang angka, harga, atau fakta apapun dari pengetahuan umum Anda.

{ticker_context}

Tool tersedia:
- retrieve_from_kb: mengambil data relevan dari knowledge base IDX30

ATURAN WAJIB:
1. SELALU mulai dengan Pikiran → Aksi: retrieve_from_kb → Input Aksi.
   JANGAN langsung menulis Jawaban Final.
2. Jawaban Final HANYA boleh ditulis SETELAH mendapat minimal satu Observasi
   dari retrieve_from_kb.
3. Frasa "tidak tersedia" atau "tidak ada data" HANYA boleh muncul di Jawaban Final
   setelah Observasi menunjukkan bahwa KB memang tidak memiliki data tersebut.
   JANGAN tulis "tidak tersedia" sebelum ada Observasi.

Format per langkah retrieve:
Pikiran: [analisis singkat informasi apa yang dibutuhkan]
Aksi: retrieve_from_kb
Input Aksi: [query spesifik — cantumkan ticker + data yang dicari]

Setelah mendapat Observasi dan evidence sudah cukup:
Pikiran: Saya sudah memiliki informasi yang cukup untuk menjawab.
Jawaban Final: [jawaban berdasarkan evidence dari Observasi]. Data per [tanggal dari evidence].

Contoh langkah pertama (sistem akan memberi Observasi — JANGAN tulis Observasi sendiri):
Pertanyaan: Berapa harga penutupan BBCA?
Pikiran: Saya perlu mencari data harga penutupan BBCA dari knowledge base.
Aksi: retrieve_from_kb
Input Aksi: harga penutupan BBCA closing price

Pertanyaan: {question}

{scratchpad}"""


# ── Prompt V5 — AKTIF (C2) ───────────────────────────────────────────────────
# Basis: REACT_PROMPT_V1 (terbukti efektif: ev=0 hanya 12%).
# Tambahan minimal vs V1:
#   1. Satu baris ticker hint di awal — memberi arah pencarian tanpa menyediakan nilai.
#   2. Satu kalimat timestamp di format Jawaban Final — instruksi eksplisit H3 prevention.
# Tidak ada: ATURAN WAJIB bernomor, fundamental_context, ticker_context section,
# aturan "tidak tersedia" — semua terbukti menyebabkan over-instruction pada Llama-3.1-8B.
REACT_PROMPT_V5 = """Anda adalah analis pasar saham Indonesia (IDX30).
Jawab HANYA berdasarkan evidence dari knowledge base. Jangan mengarang angka, harga, atau fakta.
{ticker_hint}
Tool tersedia:
- retrieve_from_kb: mengambil data relevan dari knowledge base IDX30

Format WAJIB — satu langkah per respons:
Pikiran: [analisis singkat tentang informasi apa yang dibutuhkan]
Aksi: retrieve_from_kb
Input Aksi: [query singkat untuk knowledge base]

Jika sudah cukup evidence, gunakan:
Pikiran: Saya sudah memiliki informasi yang cukup untuk menjawab.
Jawaban Final: [jawaban berdasarkan evidence saja; sertakan tanggal/periode data dari evidence]

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
    Events di-log ke trace_handle dari caller — tidak membuat sub-trace sendiri.

    Args:
        retrieval_service: instance RetrievalService untuk KB lookup.
        telemetry_service: instance TelemetryService untuk Langfuse event per iterasi.
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
        trace_handle: Any = None,
    ) -> GeneratorOutput:
        """Jalankan ReAct loop dan kembalikan jawaban + evidence.

        Loop berhenti jika LLM menghasilkan "Jawaban Final:" atau max iterasi
        tercapai. Setiap thought/action/observation di-log sebagai event ke
        trace_handle (jika disediakan oleh caller) — tidak ada sub-trace baru.
        1 request = 1 root trace bersih di Langfuse UI.

        Scratchpad di-trim ke REACT_MAX_SCRATCHPAD_CHARS (2 blok terakhir dipertahankan)
        untuk mencegah context overflow pada Llama-3.1-8B di iterasi akhir.
        Force-final diinjeksi ke scratchpad pada iterasi terakhir agar model
        tidak berakhir di error max_iterations_reached.

        Args:
            question: pertanyaan asli pengguna.
            session_id: ID sesi (dipertahankan di signature untuk dokumentasi).
            tickers: daftar ticker IDX30 yang terdeteksi dari query (opsional).
                Diteruskan ke RetrievalService untuk filter metadata KB.
            trace_handle: Langfuse span dari caller mode runner. Jika None,
                events di-skip (misal: saat dipakai standalone / unit test).

        Returns:
            GeneratorOutput. Jika error, field `error` non-None dan `answer`
            berisi FAILSAFE_ANSWER. `succeeded` property bisa dipakai mode runner
            untuk memutuskan apakah perlu fallback.
        """
        scratchpad = ""
        all_evidence: list[EvidenceItem] = []
        self._current_retrieval_latency_ms = 0.0

        # Ticker hint: satu baris ringkas, tidak menyediakan nilai/metrik.
        ticker_hint = (
            f"Saham relevan: {', '.join(tickers)}.\n"
            if tickers
            else ""
        )

        for iteration in range(1, REACT_MAX_ITERATIONS + 1):
            # ── Scratchpad trimming (cegah context overflow) ─────────────────
            effective_scratchpad = scratchpad
            if len(scratchpad) > REACT_MAX_SCRATCHPAD_CHARS:
                blocks = scratchpad.strip().split("\n\n")
                effective_scratchpad = (
                    "[...iterasi sebelumnya disingkat...]\n\n"
                    + "\n\n".join(blocks[-2:])
                )

            # ── Force-final di iterasi terakhir ──────────────────────────────
            if iteration == REACT_MAX_ITERATIONS:
                effective_scratchpad += (
                    "\n[SISTEM: Ini iterasi terakhir yang tersedia. "
                    "Gunakan evidence yang sudah dikumpulkan dan berikan "
                    "Jawaban Final sekarang. Jangan lakukan retrieve lagi.]\n"
                )

            prompt = REACT_PROMPT_V5.format(
                ticker_hint=ticker_hint,
                question=question,
                scratchpad=effective_scratchpad,
            )

            try:
                raw = self._llm.invoke(prompt).content
            except (groq.APIError, ConnectionError) as exc:
                if trace_handle is not None:
                    self._telemetry.event(
                        trace_handle,
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
                if trace_handle is not None:
                    self._telemetry.event(
                        trace_handle,
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
                if trace_handle is not None:
                    self._telemetry.event(
                        trace_handle,
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

            if trace_handle is not None:
                self._telemetry.event(
                    trace_handle,
                    name=f"thought_iter{iteration}",
                    output_data=thought,
                )

            observation, retrieved = self._retrieve(action_input, tickers=tickers)
            all_evidence.extend(retrieved)

            if trace_handle is not None:
                self._telemetry.event(
                    trace_handle,
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

        # Max iterasi tercapai tanpa Jawaban Final (force-final tidak berhasil)
        if trace_handle is not None:
            self._telemetry.event(
                trace_handle,
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
