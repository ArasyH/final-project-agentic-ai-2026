from unittest.mock import MagicMock

from app.agents.critic_agent import CriticAgent, CriticVerdict


def _fake_llm(content: str) -> MagicMock:
    fake = MagicMock()
    fake.invoke.return_value.content = content
    return fake


def test_critic_pass_verdict():
    fake_llm = _fake_llm('''
    {
      "H1_unsupported_numeric": {"flag": false, "rationale": "ok"},
      "H2_fabricated_metric": {"flag": false, "rationale": "ok"},
      "H3_stale_timestamp": {"flag": false, "rationale": "ok"},
      "H4_incorrect_inference": {"flag": false, "rationale": "ok"},
      "overall_verdict": "pass"
    }
    ''')
    critic = CriticAgent(llm=fake_llm)
    verdict = critic.validate(
        question="Berapa harga BBCA?",
        answer="9125 rupiah.",
        evidence=[{"content": "Harga BBCA: 9125."}],
    )
    assert isinstance(verdict, CriticVerdict)
    assert verdict.overall_verdict == "pass"
    assert verdict.H2_fabricated_metric.flag is False


def test_critic_overrides_verdict_when_any_flag_true():
    """Server-side derive overall_verdict: jangan trust LLM."""
    fake_llm = _fake_llm('''
    {
      "H1_unsupported_numeric": {"flag": false, "rationale": "ok"},
      "H2_fabricated_metric": {"flag": true, "rationale": "PER tidak di evidence"},
      "H3_stale_timestamp": {"flag": false, "rationale": "ok"},
      "H4_incorrect_inference": {"flag": false, "rationale": "ok"},
      "overall_verdict": "pass"
    }
    ''')
    critic = CriticAgent(llm=fake_llm)
    verdict = critic.validate(question="x", answer="y", evidence=[])
    assert verdict.overall_verdict == "fail"


def test_critic_failsafe_on_invalid_json():
    fake_llm = _fake_llm("definitely not json")
    critic = CriticAgent(llm=fake_llm)
    verdict = critic.validate(question="x", answer="y", evidence=[])
    assert verdict.overall_verdict == "fail"
    assert verdict.H1_unsupported_numeric.flag is True


def test_critic_handles_markdown_fence():
    fake_llm = _fake_llm('''```json
{
  "H1_unsupported_numeric": {"flag": false, "rationale": "ok"},
  "H2_fabricated_metric": {"flag": false, "rationale": "ok"},
  "H3_stale_timestamp": {"flag": false, "rationale": "ok"},
  "H4_incorrect_inference": {"flag": false, "rationale": "ok"},
  "overall_verdict": "pass"
}
```''')
    critic = CriticAgent(llm=fake_llm)
    verdict = critic.validate(question="x", answer="y", evidence=[])
    assert verdict.overall_verdict == "pass"


def test_critic_failsafe_on_llm_exception():
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = RuntimeError("groq down")
    critic = CriticAgent(llm=fake_llm)
    verdict = critic.validate(question="x", answer="y", evidence=[])
    assert verdict.overall_verdict == "fail"
```

---

### Justifikasi (4 bullet)

- **Pemisahan deterministik vs LLM-based** memungkinkan H1/H3 di-validasi tanpa biaya inference + sub-millisecond latency, sementara H2/H4 yang butuh reasoning ditangani LLM 70B → mendukung klaim *defensible methodology* untuk SINTA 2.
- **`overall_verdict` di-derive server-side** dari flag individual (bukan trust LLM output) → mencegah inkonsistensi internal pada output Critic, yang penting untuk pelaporan tabel deskriptif Bab IV.
- **Fail-safe verdict** kalau LLM error/JSON invalid → mode runner tidak crash, telemetry tetap punya entry untuk pasangan (mode, question_id) — wajib untuk paired comparison 50 pertanyaan (§16 Keputusan #2).
- **Backward-compat `build_llm()`** dipertahankan supaya mode 1/2/3 lama tidak break sebelum task [b] selesai.

---

### Risiko & Known Limitations

1. **H1 substring match** kasus edge: angka "9125" akan dianggap supported jika evidence mengandung "91250" (substring). Pendekatan konservatif dipilih (false-negative > false-positive untuk H1) — kalau prefer ketat, ganti substring → word-boundary regex di [b].
2. **`_normalize_number` kehilangan info desimal** (9.125 thousand-separator vs decimal). Untuk demo IDX30 (harga saham bilangan bulat), risiko rendah. Akan ditinjau ulang setelah evaluasi 50 pertanyaan.
3. **Telemetry hooks belum ada di guardrails/critic** → Langfuse trace ditambah di task [b] saat services dipanggil dari mode runner (mode runner adalah trace owner, services adalah span participant). Atomic principle.
4. **Critic prompt belum di-validate dengan Llama-3.3-70B real call** → smoke test pakai mock. Validation real call dilakukan di task [b] saat mode 3 baru pertama kali dijalankan.
5. **Docstring style**: saya pakai Google style (NumPy juga acceptable). Konfirmasi style preference Anda — kalau prefer NumPy, saya konversi di task [b] sekalian.

---

### Checklist Definition of Done

- [x] Type hints lengkap di public functions
- [x] Docstring di setiap class & public function (Google style)
- [x] `temperature=0.0` di critic LLM
- [x] Hyperparameter dari `app/config.py` (`GROQ_CRITIC_MODEL`, `MAX_EVIDENCE_AGE_HOURS`)
- [ ] **Langfuse trace** — sengaja di-defer ke task [b] (lihat Risiko #3)
- [x] Tidak ada perubahan di luar `app/`
- [x] Tidak ada dependency baru di `requirements.txt`
- [x] `pydantic` schema kompatibel — task [a] tidak menyentuh `ChatRequest`/`ChatResponse`
- [x] Smoke test snippet disertakan (11 test cases)
- [x] Tidak ada `print()` debug
- [x] Tidak ada TODO/FIXME tanpa reference
- [x] Mapping ke kategori halusinasi: H1 (numeric_traceability), H3 (timestamp_freshness), H2+H4 (Critic) — eksplisit di docstring tiap modul

---

### Smoke Test Minimal

```bash
# Dari root repo:
cd final-project-agentic-ai-2026
pytest app/tests/test_guardrails_service.py -v
pytest app/tests/test_critic_agent.py -v

# Quick sanity check tanpa pytest (deterministik, tidak butuh Groq API):
python -c "
from app.services.guardrails_service import GuardrailsService
svc = GuardrailsService()
result = svc.check(
    answer='Harga BBCA adalah 9125 per tanggal 8 Mei 2026.',
    evidence=[{'content': 'BBCA close: 9125.', 'metadata': {'timestamp': '2026-05-08T16:00:00+00:00'}}],
)
print('Status:', result.overall_status)
print('Flags:', result.hallucination_flags)
"
# Expected output:
# Status: passed
# Flags: []
