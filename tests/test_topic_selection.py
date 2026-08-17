from __future__ import annotations

from youtube_content_agent.models import (
    CaptionDraft,
    EditorialResponse,
    SlideProposal,
    TopicProposal,
)
from youtube_content_agent.topic_selection import TopicSelectionPolicy


def _topic(
    name: str,
    title: str,
    start: float,
    end: float,
    quality_score: float = 0.9,
) -> TopicProposal:
    step = (end - start) / 7
    return TopicProposal(
        topic=name,
        source_start=start,
        source_end=end,
        slides=[
            SlideProposal(
                timestamp=start + step * index,
                source_quote=f"source quote number {index}",
                zh_text=f"第{index}句话",
            )
            for index in range(1, 7)
        ],
        caption=CaptionDraft(
            title=title,
            hook="一个值得展开的问题",
            body="这是一段满足模型长度要求的正文，用来验证多段金句筛选规则能够稳定工作。",
        ),
        quality_score=quality_score,
    )


def test_select_keeps_up_to_six_distinct_topics() -> None:
    response = EditorialResponse(
        topics=[_topic(f"观点{i}", f"标题{i}", i * 100, i * 100 + 40) for i in range(6)]
    )
    selected = TopicSelectionPolicy().select(response, 6)
    assert len(selected.topics) == 6


def test_select_can_return_zero_topics() -> None:
    response = EditorialResponse(topics=[_topic("弱观点", "弱标题", 0, 40, quality_score=0.5)])
    assert TopicSelectionPolicy().select(response, 6).topics == []
    assert TopicSelectionPolicy().select(response, 0).topics == []


def test_select_removes_duplicate_and_overlapping_topics() -> None:
    first = _topic("选择创业", "选择创业", 100, 160)
    duplicate_title = _topic("另一种表达", "选择创业", 220, 280)
    overlapping = _topic("重叠观点", "重叠标题", 130, 180)
    distinct = _topic("建立技能", "建立技能", 300, 360)
    selected = TopicSelectionPolicy().select(
        EditorialResponse(topics=[first, duplicate_title, overlapping, distinct]), 6
    )
    assert [topic.topic for topic in selected.topics] == ["选择创业", "建立技能"]
