from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from youtube_content_agent.editorial import (
    MimoEditorialProvider,
    _parse_editorial_json,
    _safe_timeline_diagnostic,
)
from youtube_content_agent.errors import ConfigurationError, ExternalToolError
from youtube_content_agent.models import Transcript, TranscriptSegment, VideoMetadata

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


def test_mimo_requires_api_key() -> None:
    with pytest.raises(ConfigurationError, match="MIMO_API_KEY"):
        MimoEditorialProvider(None, "mimo-v2.5", "https://api.xiaomimimo.com/v1")


def test_parse_mimo_json_accepts_markdown_fence() -> None:
    parsed = _parse_editorial_json(f"```json\n{VALID_RESPONSE}\n```", "MiMo")
    assert parsed.topics[0].source_start == 100


def test_parse_mimo_json_rejects_invalid_schema() -> None:
    with pytest.raises(ExternalToolError, match="数据模型校验"):
        _parse_editorial_json('{"topics": []}', "MiMo")


def test_parse_mimo_json_reports_safe_validation_location() -> None:
    with pytest.raises(
        ExternalToolError, match=r"topics:too_short:List should have at least 1 item"
    ):
        _parse_editorial_json('{"topics": []}', "MiMo")


def test_timeline_diagnostic_only_reports_numeric_ranges() -> None:
    assert _safe_timeline_diagnostic(VALID_RESPONSE) == (
        "source=100.00-140.00,duration=40.00,slides=102.00-134.00"
    )


def test_mimo_calls_chat_completions(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=VALID_RESPONSE))]
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
    transcript = Transcript(
        video_id="video",
        language="en",
        source="fixture",
        segments=[TranscriptSegment(start=100, end=140, text="Verified source")],
    )
    result = provider.create_topics(metadata, transcript, 1)
    assert result.topics[0].topic == "普通工作日里的领导力"
    assert captured["model"] == "mimo-v2.5"
    assert captured["temperature"] == 0.3
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_mimo_repairs_invalid_long_segment_once(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    invalid_response = VALID_RESPONSE.replace('"source_end": 140', '"source_end": 400')
    responses = iter((invalid_response, VALID_RESPONSE, VALID_RESPONSE))
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
    transcript = Transcript(
        video_id="video",
        language="en",
        source="fixture",
        segments=[TranscriptSegment(start=100, end=400, text="Verified source")],
    )
    result = provider.create_topics(metadata, transcript, 1)
    assert result.topics[0].source_end == 140
    assert call_count == 3
