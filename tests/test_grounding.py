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
