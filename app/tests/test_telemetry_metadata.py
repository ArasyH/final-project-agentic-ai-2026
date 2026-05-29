"""Verifikasi 11 field metadata wajib §15 di semua 4 mode + question_id propagation.

Strategi mock:
- Langfuse client di-mock via patch("app.services.telemetry_service.Langfuse")
  sehingga TelemetryService real (measure_latency, _record_latency, end_trace)
  berjalan — hanya Langfuse network call yang di-bypass.
- LLM, RetrievalService, GeneratorAgent, CriticAgent, GuardrailsService, CacheService
  di-mock di namespace module masing-masing.
"""
import time as _time
from unittest.mock import MagicMock, patch

from app.agents.generator_agent import GeneratorAgent, GeneratorOutput
from app.modes.mode_1_llm_only import run_mode_1
from app.modes.mode_2_rag_only import run_mode_2
from app.modes.mode_3_rag_jc import run_mode_3
from app.modes.mode_4_rag_jc_cache import run_mode_4
from app.schemas import InternalResponse

_REQUIRED_KEYS = {
    "mode",
    "question_id",
    "cache_status",
    "validator_status",
    "hallucination_flags",
    "evidence_count",
    "confidence",
    "latency_ms_total",
    "latency_ms_retrieval",
    "latency_ms_generation",
    "latency_ms_critic",
}

_TS = "2026-05-19T09:00:00+00:00"


def _make_trace_mock():
    """Fake trace object yang capture update() calls."""
    trace = MagicMock()
    trace.id = "fake-trace-id"
    return trace


def _fake_llm(content: str = "jawaban test") -> MagicMock:
    m = MagicMock()
    m.invoke.return_value.content = content
    return m


def _fake_doc(content: str = "BBCA close: 9125.") -> MagicMock:
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"title": "test", "timestamp": _TS}
    return doc


def _passed_response(**kwargs) -> InternalResponse:
    defaults = dict(
        answer="Harga BBCA 9125.",
        evidence=[], sources=[], tickers=["BBCA"],
        timestamp=_TS, confidence=0.85,
        validator_status="passed", cache_status="bypassed",
        mode="mode_3_rag_jc", hallucination_flags=[],
    )
    defaults.update(kwargs)
    return InternalResponse(**defaults)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_end_trace_metadata(mock_trace: MagicMock) -> dict:
    """Ambil metadata yang dikirim ke trace.update() oleh end_trace()."""
    assert mock_trace.update.called, "end_trace() tidak dipanggil (trace.update tidak tercall)"
    return mock_trace.update.call_args[1]["metadata"]


# ── Test 1: mode_1 — retrieval=0, critic=0 ───────────────────────────────────

def test_mode_1_metadata_complete():
    mock_trace = _make_trace_mock()
    with (
        patch("app.services.telemetry_service.Langfuse") as MockLF,
        patch("app.modes.mode_1_llm_only.build_llm", return_value=_fake_llm()),
    ):
        MockLF.return_value.start_span.return_value = mock_trace

        run_mode_1("Berapa harga BBCA?", "s1", "Q001")

    meta = _extract_end_trace_metadata(mock_trace)
    assert _REQUIRED_KEYS.issubset(meta.keys()), f"Missing keys: {_REQUIRED_KEYS - meta.keys()}"
    assert meta["latency_ms_retrieval"] == 0.0
    assert meta["latency_ms_critic"] == 0.0
    assert meta["latency_ms_generation"] > 0 or meta["latency_ms_generation"] == 0.0
    assert meta["question_id"] == "Q001"
    assert meta["mode"] == "mode_1_llm_only"
    assert meta["evidence_count"] == 0


# ── Test 2: mode_2 — critic=0 ────────────────────────────────────────────────

def test_mode_2_metadata_complete():
    mock_trace = _make_trace_mock()

    def _slow_retrieve(q, **_kw):
        _time.sleep(0.015)  # 15ms — memastikan retrieval latency terukur
        return [_fake_doc()]

    with (
        patch("app.services.telemetry_service.Langfuse") as MockLF,
        patch("app.modes.mode_2_rag_only.build_llm", return_value=_fake_llm()),
        patch("app.modes.mode_2_rag_only.RetrievalService") as MockRetriever,
    ):
        MockLF.return_value.start_span.return_value = mock_trace
        MockRetriever.return_value.retrieve.side_effect = _slow_retrieve

        run_mode_2("Berapa harga BBCA?", "s1", "Q001")

    meta = _extract_end_trace_metadata(mock_trace)
    assert _REQUIRED_KEYS.issubset(meta.keys()), f"Missing keys: {_REQUIRED_KEYS - meta.keys()}"
    assert meta["latency_ms_critic"] == 0.0
    # Retrieval harus reflect real KB lookup time (>= 10ms dari sleep)
    assert meta["latency_ms_retrieval"] >= 10.0, (
        f"latency_ms_retrieval={meta['latency_ms_retrieval']} terlalu kecil — "
        "mode_2 harus measure real retrieval time"
    )
    assert meta["question_id"] == "Q001"
    assert meta["mode"] == "mode_2_rag_only"
    assert meta["evidence_count"] == 1


# ── Test 3: mode_3 — semua stage hadir ───────────────────────────────────────

def test_mode_3_metadata_complete():
    mock_trace = _make_trace_mock()
    with (
        patch("app.services.telemetry_service.Langfuse") as MockLF,
        patch("app.modes.mode_3_rag_jc._run_rag_jc_pipeline") as mock_pipeline,
        patch("app.modes.mode_3_rag_jc.RetrievalService"),
        patch("app.modes.mode_3_rag_jc.GuardrailsService"),
        patch("app.modes.mode_3_rag_jc.CriticAgent"),
        patch("app.modes.mode_3_rag_jc.GeneratorAgent"),
    ):
        MockLF.return_value.start_span.return_value = mock_trace
        mock_pipeline.return_value = _passed_response(mode="mode_3_rag_jc", cache_status="bypassed")

        run_mode_3("Berapa harga BBCA?", "s1", "Q001")

    meta = _extract_end_trace_metadata(mock_trace)
    assert _REQUIRED_KEYS.issubset(meta.keys()), f"Missing keys: {_REQUIRED_KEYS - meta.keys()}"
    assert meta["question_id"] == "Q001"
    assert meta["mode"] == "mode_3_rag_jc"
    assert meta["validator_status"] == "passed"


# ── Test 4: mode_4 cache hit — retrieval/generation/critic = 0 ───────────────

def test_mode_4_cache_hit_metadata_complete():
    mock_trace = _make_trace_mock()
    hit_dict = {
        "hit": True, "status": "hit", "score": 0.92,
        "answer": "Harga BBCA 9125.",
        "intent": "price_lookup",
        "evidence_summary": [], "source_metadata": [],
        "timestamp": _TS,
    }
    with (
        patch("app.services.telemetry_service.Langfuse") as MockLF,
        patch("app.modes.mode_4_rag_jc_cache.CacheService") as MockCache,
        patch("app.modes.mode_4_rag_jc_cache.RetrievalService"),
        patch("app.modes.mode_4_rag_jc_cache.GuardrailsService"),
        patch("app.modes.mode_4_rag_jc_cache.CriticAgent"),
        patch("app.modes.mode_4_rag_jc_cache.GeneratorAgent"),
    ):
        MockLF.return_value.start_span.return_value = mock_trace
        MockCache.return_value.lookup.return_value = hit_dict

        run_mode_4("Berapa harga BBCA?", "s1", "Q001")

    meta = _extract_end_trace_metadata(mock_trace)
    assert _REQUIRED_KEYS.issubset(meta.keys()), f"Missing keys: {_REQUIRED_KEYS - meta.keys()}"
    assert meta["latency_ms_retrieval"] == 0.0
    assert meta["latency_ms_generation"] == 0.0
    assert meta["latency_ms_critic"] == 0.0
    assert meta["cache_status"] == "hit"
    assert meta["question_id"] == "Q001"


# ── Test 5: mode_4 cache miss — semua latency hadir ──────────────────────────

def test_mode_4_cache_miss_metadata_complete():
    mock_trace = _make_trace_mock()
    miss_dict = {"hit": False, "status": "miss", "score": 0.1}
    with (
        patch("app.services.telemetry_service.Langfuse") as MockLF,
        patch("app.modes.mode_4_rag_jc_cache.CacheService") as MockCache,
        patch("app.modes.mode_4_rag_jc_cache._run_rag_jc_pipeline") as mock_pipeline,
        patch("app.modes.mode_4_rag_jc_cache.RetrievalService"),
        patch("app.modes.mode_4_rag_jc_cache.GuardrailsService"),
        patch("app.modes.mode_4_rag_jc_cache.CriticAgent"),
        patch("app.modes.mode_4_rag_jc_cache.GeneratorAgent"),
    ):
        MockLF.return_value.start_span.return_value = mock_trace
        MockCache.return_value.lookup.return_value = miss_dict
        mock_pipeline.return_value = _passed_response(
            mode="mode_4_rag_jc_cache", cache_status="miss"
        )

        run_mode_4("Berapa harga BBCA?", "s1", "Q001")

    meta = _extract_end_trace_metadata(mock_trace)
    assert _REQUIRED_KEYS.issubset(meta.keys()), f"Missing keys: {_REQUIRED_KEYS - meta.keys()}"
    assert meta["cache_status"] == "miss"
    assert meta["question_id"] == "Q001"


# ── Test 6: chat_api auto-generate question_id ────────────────────────────────

def test_chat_api_autogenerate_question_id():
    """question_id di-generate otomatis jika klien tidak kirim."""
    captured = {}

    def fake_run(question, session_id, question_id, mode):
        captured["question_id"] = question_id
        return _passed_response(mode=mode, cache_status="bypassed")

    with patch("app.chat_api.orchestrator") as mock_orch:
        mock_orch.run.side_effect = fake_run
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        client.post("/chat", json={"question": "test"})

    qid = captured.get("question_id", "")
    assert qid.startswith("Q-"), f"Expected Q-xxx prefix, got: {qid!r}"
    assert len(qid) == 10  # "Q-" + 8 hex chars


# ── Test 7: chat_api gunakan question_id dari klien ──────────────────────────

def test_chat_api_uses_client_question_id():
    """question_id dari klien harus diteruskan ke orchestrator tanpa diubah."""
    captured = {}

    def fake_run(question, session_id, question_id, mode):
        captured["question_id"] = question_id
        return _passed_response(mode=mode, cache_status="bypassed")

    with patch("app.chat_api.orchestrator") as mock_orch:
        mock_orch.run.side_effect = fake_run
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        client.post("/chat", json={"question": "test", "question_id": "Q042"})

    assert captured.get("question_id") == "Q042"


# ── Test 8: mode_4 cache miss = 1 trace (bukan 2) ────────────────────────────

def test_mode_4_cache_miss_single_trace():
    """Verifikasi trace unifikasi: 1 request = 1 Langfuse trace."""
    with (
        patch("app.modes.mode_4_rag_jc_cache.TelemetryService") as MockTelemetry,
        patch("app.modes.mode_4_rag_jc_cache.CacheService") as MockCache,
        patch("app.modes.mode_4_rag_jc_cache.RetrievalService"),
        patch("app.modes.mode_4_rag_jc_cache.GuardrailsService"),
        patch("app.modes.mode_4_rag_jc_cache.CriticAgent"),
        patch("app.modes.mode_4_rag_jc_cache.GeneratorAgent"),
        patch("app.modes.mode_4_rag_jc_cache._run_rag_jc_pipeline") as mock_pipeline,
    ):
        mock_tel_instance = MockTelemetry.return_value
        mock_trace = _make_trace_mock()
        mock_tel_instance.start_trace.return_value = mock_trace
        mock_tel_instance.measure_latency.return_value.__enter__ = MagicMock(return_value=None)
        mock_tel_instance.measure_latency.return_value.__exit__ = MagicMock(return_value=False)

        MockCache.return_value.lookup.return_value = {"hit": False, "status": "miss", "score": 0.1}
        mock_pipeline.return_value = _passed_response(
            mode="mode_4_rag_jc_cache", cache_status="miss"
        )

        run_mode_4("Berapa harga BBCA?", "s1", "Q001")

        # TelemetryService hanya diinstansiasi 1x di run_mode_4 (bukan 2x)
        assert MockTelemetry.call_count == 1
        # start_trace hanya dipanggil 1x
        assert mock_tel_instance.start_trace.call_count == 1
        # pipeline menerima parent_trace dari run_mode_4 (bukan membuat trace baru)
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["parent_trace"] is mock_trace
        assert call_kwargs["telemetry"] is mock_tel_instance


# ── Test 9: GeneratorOutput.retrieval_latency_ms accumulator ─────────────────

def test_generator_retrieval_latency_accumulator():
    """retrieval_latency_ms mengakumulasi semua retrieve_from_kb calls dalam 1 generate()."""

    def _slow_retrieve(query, tickers=None):
        _time.sleep(0.015)  # 15ms per call
        doc = MagicMock()
        doc.page_content = "BBCA close: 9125."
        return [doc]

    fake_retriever = MagicMock()
    fake_retriever.retrieve.side_effect = _slow_retrieve

    fake_telemetry = MagicMock()
    fake_trace = MagicMock()
    fake_telemetry.start_trace.return_value = fake_trace

    # LLM: iterasi 1 → ReAct step; iterasi 2 → Jawaban Final
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [
        MagicMock(content="Pikiran: perlu data BBCA\nAksi: retrieve_from_kb\nInput Aksi: harga BBCA"),
        MagicMock(content="Pikiran: sudah cukup\nJawaban Final: Harga BBCA adalah 9125."),
    ]

    agent = GeneratorAgent(
        retrieval_service=fake_retriever,
        telemetry_service=fake_telemetry,
        llm=fake_llm,
    )

    output = agent.generate(question="Berapa harga BBCA?", session_id="s1")

    assert output.succeeded
    # retrieve dipanggil 1x (1 ReAct iteration), jadi >= 10ms
    assert output.retrieval_latency_ms >= 10.0, (
        f"retrieval_latency_ms={output.retrieval_latency_ms:.2f} terlalu kecil"
    )
    assert fake_retriever.retrieve.call_count == 1


# ── Test 10: mode_3 pipeline gunakan retrieval_latency_ms dari Generator ──────

def test_mode_3_pipeline_retrieval_from_generator():
    """latency_ms_retrieval di trace harus mencerminkan gen_output.retrieval_latency_ms."""
    mock_trace = _make_trace_mock()

    mock_gen_output = GeneratorOutput(
        answer="Harga BBCA 9125.",
        evidence=[],
        iterations_used=2,
        retrieval_latency_ms=20.0,  # 20ms simulasi KB retrieval
    )

    mock_guardrail = MagicMock()
    mock_guardrail.overall_status = "passed"
    mock_guardrail.hallucination_flags = []

    mock_verdict = MagicMock()
    mock_verdict.overall_verdict = "pass"
    mock_verdict.H2_fabricated_metric.flag = False
    mock_verdict.H4_incorrect_inference.flag = False

    with (
        patch("app.services.telemetry_service.Langfuse") as MockLF,
        patch("app.modes.mode_3_rag_jc.GeneratorAgent") as MockGen,
        patch("app.modes.mode_3_rag_jc.GuardrailsService") as MockGuards,
        patch("app.modes.mode_3_rag_jc.CriticAgent") as MockCritic,
        patch("app.modes.mode_3_rag_jc.RetrievalService"),
    ):
        MockLF.return_value.start_span.return_value = mock_trace
        MockGen.return_value.generate.return_value = mock_gen_output
        MockGuards.return_value.check.return_value = mock_guardrail
        MockCritic.return_value.validate.return_value = mock_verdict

        run_mode_3("Berapa harga BBCA?", "s1", "Q001")

    meta = _extract_end_trace_metadata(mock_trace)
    assert meta["latency_ms_retrieval"] == 20.0, (
        f"latency_ms_retrieval={meta['latency_ms_retrieval']} harus 20.0 "
        "(dari gen_output.retrieval_latency_ms)"
    )
    # generation pure <= total (tidak bisa lebih besar dari total)
    assert meta["latency_ms_generation"] >= 0.0
    assert meta["latency_ms_generation"] <= meta["latency_ms_total"] + 1.0


# ── Test 11: mode_4 cache miss — trace unifikasi + retrieval dari Generator ───

def test_mode_4_miss_logs_retrieval_from_generator():
    """mode_4 cache miss: latency_ms_retrieval dari pipeline, bukan normalize_query."""
    mock_trace = _make_trace_mock()
    miss_dict = {"hit": False, "status": "miss", "score": 0.1}

    mock_gen_output = GeneratorOutput(
        answer="Harga BBCA 9125.",
        evidence=[],
        iterations_used=1,
        retrieval_latency_ms=18.0,
    )

    mock_guardrail = MagicMock()
    mock_guardrail.overall_status = "passed"
    mock_guardrail.hallucination_flags = []

    mock_verdict = MagicMock()
    mock_verdict.overall_verdict = "pass"
    mock_verdict.H2_fabricated_metric.flag = False
    mock_verdict.H4_incorrect_inference.flag = False

    with (
        patch("app.services.telemetry_service.Langfuse") as MockLF,
        patch("app.modes.mode_4_rag_jc_cache.CacheService") as MockCache,
        patch("app.modes.mode_4_rag_jc_cache.GeneratorAgent") as MockGen,
        patch("app.modes.mode_4_rag_jc_cache.GuardrailsService") as MockGuards,
        patch("app.modes.mode_4_rag_jc_cache.CriticAgent") as MockCritic,
        patch("app.modes.mode_4_rag_jc_cache.RetrievalService"),
    ):
        MockLF.return_value.start_span.return_value = mock_trace
        MockCache.return_value.lookup.return_value = miss_dict
        MockCache.return_value.store.return_value = None
        MockGen.return_value.generate.return_value = mock_gen_output
        MockGuards.return_value.check.return_value = mock_guardrail
        MockCritic.return_value.validate.return_value = mock_verdict

        run_mode_4("Berapa harga BBCA?", "s1", "Q001")

    meta = _extract_end_trace_metadata(mock_trace)
    assert meta["latency_ms_retrieval"] == 18.0, (
        f"latency_ms_retrieval={meta['latency_ms_retrieval']} harus 18.0"
    )
    assert meta["cache_status"] == "miss"
    assert meta["latency_ms_generation"] >= 0.0
