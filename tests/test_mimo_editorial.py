from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from youtube_content_agent.editorial import (
    MimoEditorialProvider,
    _audit_passed,
    _detect_hard_coherence_risks,
    _parse_editorial_json,
    _safe_timeline_diagnostic,
    _topic_count_instruction,
)
from youtube_content_agent.errors import ConfigurationError, ExternalToolError
from youtube_content_agent.models import (
    EditorialResponse,
    StoryCoherenceAudit,
    Transcript,
    TranscriptSegment,
    VideoMetadata,
)

VALID_RESPONSE = """{
  "topics": [{
    "topic": "普通工作日里的领导力",
    "source_start": 100,
    "source_end": 140,
    "slides": [
      {"timestamp": 102, "source_quote": "Verified source", "zh_text": "第一张"},
      {"timestamp": 108, "source_quote": "Verified source", "zh_text": "第二张"},
      {"timestamp": 114, "source_quote": "Verified source", "zh_text": "第三张"},
      {"timestamp": 120, "source_quote": "Verified source", "zh_text": "第四张"},
      {"timestamp": 126, "source_quote": "Verified source", "zh_text": "第五张"},
      {"timestamp": 134, "source_quote": "Verified source", "zh_text": "第六张"}
    ],
    "caption": {
      "title": "领导力藏在普通工作日",
      "hook": "为什么危机时刻并不能定义领导力？",
      "body": "真正拉开差距的，是一个人在普通工作日里能否持续在场，并且不断做出高质量决定。"
    },
    "quality_score": 0.9
  }]
}"""

VALID_AUDIT = """{
  "topics": [{"topic_index": 1, "coherent": true, "score": 0.9, "issues": []}]
}"""

INVALID_AUDIT = """{
  "topics": [{
    "topic_index": 1,
    "coherent": false,
    "score": 0.6,
    "issues": [{
      "slide_index": 2,
      "category": "missing_bridge",
      "explanation": "缺少承上启下的判断条件",
      "repair_instruction": "补充明确的逻辑桥梁"
    }]
  }]
}"""


def test_mimo_requires_api_key() -> None:
    with pytest.raises(ConfigurationError, match="MIMO_API_KEY"):
        MimoEditorialProvider(None, "mimo-v2.5", "https://api.xiaomimimo.com/v1")


def test_parse_mimo_json_accepts_markdown_fence() -> None:
    parsed = _parse_editorial_json(f"```json\n{VALID_RESPONSE}\n```", "MiMo")
    assert parsed.topics[0].source_start == 100


def test_parse_mimo_json_accepts_zero_topics() -> None:
    assert _parse_editorial_json('{"topics": []}', "MiMo").topics == []


def test_parse_mimo_json_rejects_more_than_six_topics() -> None:
    topic = json.loads(VALID_RESPONSE)["topics"][0]
    payload = json.dumps({"topics": [topic] * 7})
    with pytest.raises(ExternalToolError):
        _parse_editorial_json(payload, "MiMo")


def test_topic_count_instruction_allows_zero_without_padding() -> None:
    instruction = _topic_count_instruction(6)
    assert "between 0 and 6" in instruction
    assert "without padding" in instruction


def test_hard_coherence_risk_detects_stacked_contrast_connectors() -> None:
    response = EditorialResponse.model_validate_json(VALID_RESPONSE)
    risky = response.model_copy(
        update={
            "topics": [
                response.topics[0].model_copy(
                    update={
                        "slides": [
                            response.topics[0].slides[0].model_copy(
                                update={"zh_text": "但并非如此 另一方面还有例外"}
                            ),
                            *response.topics[0].slides[1:],
                        ]
                    }
                )
            ]
        }
    )
    assert "stacks 但 and 另一方面" in _detect_hard_coherence_risks(risky)[0]


def test_coherence_audit_can_pass_with_non_blocking_suggestions() -> None:
    audit = StoryCoherenceAudit.model_validate_json(
        INVALID_AUDIT.replace('"coherent": false', '"coherent": true').replace(
            '"score": 0.6', '"score": 0.88'
        )
    )
    assert _audit_passed(audit, 1) is True


def test_timeline_diagnostic_only_reports_numeric_ranges() -> None:
    assert _safe_timeline_diagnostic(VALID_RESPONSE) == (
        "source=100.00-140.00,duration=40.00,slides=102.00-134.00"
    )


def test_mimo_calls_chat_completions(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}
    calls: list[dict[str, object]] = []
    responses = iter((VALID_RESPONSE, VALID_RESPONSE, VALID_AUDIT))

    class FakeCompletions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(kwargs)
            captured.setdefault("initial_messages", kwargs["messages"])
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("youtube_content_agent.editorial.OpenAI", lambda **_: fake_client)
    provider = MimoEditorialProvider(
        "secret-not-logged", "mimo-v2.5", "https://api.xiaomimimo.com/v1"
    )
    metadata = VideoMetadata(
        video_id="video",
        youtube_url="https://www.youtube.com/watch?v=video",
        title="Interview",
        channel="Channel",
        published_at=datetime(2026, 8, 17, tzinfo=UTC),
        duration=200,
    )
    transcript = _transcript()
    result = provider.create_topics(metadata, transcript, 6)
    assert result.topics[0].topic == "普通工作日里的领导力"
    assert captured["model"] == "mimo-v2.5"
    assert captured["temperature"] == 0.3
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    messages = captured["initial_messages"]
    assert isinstance(messages, list)
    assert "between 0 and 6" in messages[1]["content"]
    assert len(calls) == 3
    audit_messages = calls[2]["messages"]
    assert isinstance(audit_messages, list)
    assert "independent Chinese story-continuity reviewer" in audit_messages[1]["content"]


def test_mimo_repairs_invalid_long_segment_once(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    invalid_response = VALID_RESPONSE.replace('"source_end": 140', '"source_end": 400')
    responses = iter((invalid_response, VALID_RESPONSE, VALID_RESPONSE, VALID_AUDIT))
    call_count = 0

    class FakeCompletions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            del kwargs
            call_count += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("youtube_content_agent.editorial.OpenAI", lambda **_: fake_client)
    provider = MimoEditorialProvider(
        "secret-not-logged", "mimo-v2.5", "https://api.xiaomimimo.com/v1"
    )
    metadata = VideoMetadata(
        video_id="video",
        youtube_url="https://www.youtube.com/watch?v=video",
        title="Interview",
        channel="Channel",
        duration=500,
    )
    transcript = _transcript()
    result = provider.create_topics(metadata, transcript, 1)
    assert result.topics[0].source_end == 140
    assert call_count == 4


def test_mimo_skips_review_calls_when_no_topic_is_selected(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    call_count = 0

    class FakeCompletions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            del kwargs
            call_count += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"topics": []}'))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("youtube_content_agent.editorial.OpenAI", lambda **_: fake_client)
    provider = MimoEditorialProvider(
        "secret-not-logged", "mimo-v2.5", "https://api.xiaomimimo.com/v1"
    )
    result = provider.create_topics(_metadata(), _transcript(), 6)
    assert result.topics == []
    assert call_count == 1


def test_mimo_rejects_coherence_edit_that_changes_source_scope(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    changed_scope = VALID_RESPONSE.replace('"source_start": 100', '"source_start": 101')
    responses = iter((VALID_RESPONSE, VALID_RESPONSE, INVALID_AUDIT, changed_scope))

    class FakeCompletions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("youtube_content_agent.editorial.OpenAI", lambda **_: fake_client)
    provider = MimoEditorialProvider(
        "secret-not-logged", "mimo-v2.5", "https://api.xiaomimimo.com/v1"
    )
    with pytest.raises(ExternalToolError, match="story coherence 改变了 Topic 1 的来源区间"):
        provider.create_topics(_metadata(), _transcript(), 1)


def test_mimo_repairs_ungrounded_source_quote_before_reviews(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    invalid_quote = VALID_RESPONSE.replace('"Verified source"', '"Invented quote"', 1)
    responses = iter((invalid_quote, VALID_RESPONSE, VALID_RESPONSE, VALID_AUDIT))
    calls: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("youtube_content_agent.editorial.OpenAI", lambda **_: fake_client)
    provider = MimoEditorialProvider(
        "secret-not-logged", "mimo-v2.5", "https://api.xiaomimimo.com/v1"
    )
    result = provider.create_topics(_metadata(), _transcript(), 1)
    assert result.topics[0].slides[0].source_quote == "Verified source"
    assert len(calls) == 4
    repair_messages = calls[1]["messages"]
    assert isinstance(repair_messages, list)
    assert "cannot be grounded" in repair_messages[1]["content"]


def test_mimo_rewrites_failed_audit_and_reviews_again(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    revised = VALID_RESPONSE.replace('"第一张"', '"先建立共同前提"').replace(
        '"第二张"', '"再展开核心观点"'
    )
    responses = iter(
        (VALID_RESPONSE, VALID_RESPONSE, INVALID_AUDIT, revised, revised, VALID_AUDIT)
    )
    calls: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("youtube_content_agent.editorial.OpenAI", lambda **_: fake_client)
    provider = MimoEditorialProvider(
        "secret-not-logged", "mimo-v2.5", "https://api.xiaomimimo.com/v1"
    )
    result = provider.create_topics(_metadata(), _transcript(), 1)
    assert result.topics[0].slides[0].zh_text == "先建立共同前提"
    assert len(calls) == 6
    rewrite_messages = calls[3]["messages"]
    assert isinstance(rewrite_messages, list)
    assert "Fix every supplied audit issue" in rewrite_messages[1]["content"]


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        video_id="video",
        youtube_url="https://www.youtube.com/watch?v=video",
        title="Interview",
        channel="Channel",
        duration=200,
    )


def _transcript() -> Transcript:
    return Transcript(
        video_id="video",
        language="en",
        source="fixture",
        segments=[
            TranscriptSegment(start=start, end=min(140, start + 6), text="Verified source")
            for start in (100, 108, 114, 120, 126, 134)
        ],
    )
