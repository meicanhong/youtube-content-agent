import logging

from youtube_content_agent.grounding import GroundingService
from youtube_content_agent.models import (
    CaptionDraft,
    SlideProposal,
    TopicProposal,
    Transcript,
    TranscriptSegment,
)


def test_grounding_derives_original_text_from_transcript() -> None:
    transcript = Transcript(
        video_id="abc",
        language="en",
        source="fixture",
        segments=[
            TranscriptSegment(start=10 + index * 5, end=15 + index * 5, text=f"source {index}")
            for index in range(9)
        ],
    )
    proposal = TopicProposal(
        topic="一个观点",
        source_start=10,
        source_end=50,
        slides=[
            SlideProposal(
                timestamp=value, source_quote=f"source {source_index}", zh_text=f"中文内容 {index}"
            )
            for index, (value, source_index) in enumerate(
                ((12, 0), (18, 1), (24, 2), (30, 4), (36, 5), (44, 6)), start=1
            )
        ],
        caption=CaptionDraft(
            title="一个标题",
            hook="一个问题",
            body="这是一段足够长的中文正文，用来验证来源约束是否生效。",
        ),
        quality_score=0.9,
    )
    source, content = GroundingService().ground(proposal, transcript)
    assert source.original_text.startswith("source 0")
    assert content.slides[0].original_text == "source 0"
    assert all(slide.original_text.startswith("source") for slide in content.slides)


def test_quote_can_span_many_short_transcript_segments() -> None:
    segments = [
        TranscriptSegment(start=index, end=index + 1, text=f"word{index}") for index in range(12)
    ]
    timestamp, original = GroundingService._resolve_quote(
        "word0 word1 word2 word3 word4 word5 word6 word7 word8", 0, segments
    )
    assert timestamp == 0
    assert original == "word0 word1 word2 word3 word4 word5 word6 word7 word8"


def test_quote_can_span_more_than_twenty_four_transcript_segments() -> None:
    segments = [
        TranscriptSegment(start=index, end=index + 1, text=f"word{index:02d}")
        for index in range(32)
    ]
    quote = " ".join(segment.text for segment in segments)

    timestamp, original = GroundingService._resolve_quote(quote, 0, segments)

    assert timestamp == 0
    assert original == quote


def test_grounding_tolerates_subtitle_boundary_float_noise() -> None:
    transcript = Transcript(
        video_id="boundary",
        language="en",
        source="fixture",
        segments=[
            TranscriptSegment(start=15, end=19.999999999, text="first half"),
            TranscriptSegment(start=18, end=22, text="second half"),
            TranscriptSegment(start=22, end=42, text="supporting source"),
        ],
    )
    proposal = TopicProposal(
        topic="边界观点",
        source_start=20,
        source_end=40,
        slides=[
            SlideProposal(
                timestamp=20 + index,
                source_quote="first half second half",
                zh_text=f"边界内容 {index}",
            )
            for index in range(6)
        ],
        caption=CaptionDraft(
            title="边界标题",
            hook="边界问题",
            body="这是一段足够长的中文正文，用来验证浮点时间边界能够被容忍。",
        ),
        quality_score=0.9,
    )

    source, content = GroundingService().ground(proposal, transcript)

    assert source.transcript_segments[0].text == "first half"
    assert content.slides[0].timestamp == 15


def test_grounding_falls_back_to_real_transcript_at_proposed_timestamp(caplog) -> None:  # type: ignore[no-untyped-def]
    segments = [
        TranscriptSegment(start=100, end=104, text="This is not actually a dunkey"),
        TranscriptSegment(start=104, end=108, text="but pretend it is a dunkey"),
        TranscriptSegment(start=108, end=112, text="the animal is hungry and thirsty"),
    ]

    with caplog.at_level(logging.WARNING):
        timestamp, original = GroundingService._resolve_quote(
            "This is not actually a donkey, but pretend it is a donkey",
            104,
            segments,
        )

    assert timestamp == 104
    assert "dunkey" in original
    assert "using transcript timestamp fallback" in caplog.text
