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
    """overall_verdict di-derive server-side: jangan trust nilai dari LLM."""
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
