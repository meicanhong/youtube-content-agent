import pytest
from pydantic import ValidationError

from youtube_content_agent.models import CaptionDraft, SlideProposal, TopicProposal


def make_topic() -> TopicProposal:
    return TopicProposal(
        topic="测试选题",
        source_start=10,
        source_end=50,
        slides=[
            SlideProposal(
                timestamp=value, source_quote=f"source {index}", zh_text=f"第{index}张字幕"
            )
            for index, value in enumerate((12, 18, 24, 30, 36, 44), start=1)
        ],
        caption=CaptionDraft(
            title="测试标题",
            hook="测试开头",
            body="这是一段足够长的中文正文，用来验证结构化模型的边界。",
        ),
        quality_score=0.8,
    )


def test_topic_requires_ascending_timestamps() -> None:
    topic = make_topic().model_dump()
    topic["slides"][2]["timestamp"] = 11
    with pytest.raises(ValidationError, match="ascending"):
        TopicProposal.model_validate(topic)


def test_topic_rejects_timestamp_outside_source() -> None:
    topic = make_topic().model_dump()
    topic["slides"][0]["timestamp"] = 2
    with pytest.raises(ValidationError, match="inside"):
        TopicProposal.model_validate(topic)
