from __future__ import annotations

import re

from .errors import GroundingError
from .models import (
    ContentData,
    GroundedSlide,
    SourceSegment,
    TopicProposal,
    Transcript,
    TranscriptSegment,
)


class GroundingService:
    """Derive every source quote from transcript data instead of trusting generated text."""

    def ground(
        self, proposal: TopicProposal, transcript: Transcript
    ) -> tuple[SourceSegment, ContentData]:
        source_segments = [
            segment
            for segment in transcript.segments
            if segment.end >= proposal.source_start and segment.start <= proposal.source_end
        ]
        if not source_segments:
            raise GroundingError(
                f"选题 {proposal.topic!r} 的 source segment 在 Transcript 中没有对应内容"
            )
        coverage_start = min(segment.start for segment in source_segments)
        coverage_end = max(segment.end for segment in source_segments)
        if coverage_start > proposal.source_start + 5 or coverage_end < proposal.source_end - 5:
            raise GroundingError(f"选题 {proposal.topic!r} 的 Transcript 时间覆盖不完整")

        slides = []
        for index, slide in enumerate(proposal.slides, start=1):
            timestamp, original_text = self._resolve_quote(
                slide.source_quote, slide.timestamp, source_segments
            )
            slides.append(
                GroundedSlide(
                    index=index,
                    timestamp=timestamp,
                    original_text=original_text,
                    zh_text=slide.zh_text,
                )
            )
        grounded_timestamps = [slide.timestamp for slide in slides]
        if grounded_timestamps != sorted(grounded_timestamps):
            raise GroundingError(f"选题 {proposal.topic!r} 回源后的 Slide 时间戳不是升序")
        source = SourceSegment(
            start=proposal.source_start,
            end=proposal.source_end,
            original_text=" ".join(segment.text for segment in source_segments),
            transcript_segments=source_segments,
        )
        content = ContentData(
            topic=proposal.topic,
            hook=proposal.caption.hook,
            slides=slides,
            caption=proposal.caption,
            quality_score=proposal.quality_score,
        )
        return source, content

    @classmethod
    def _resolve_quote(
        cls, quote: str, proposed_timestamp: float, segments: list[TranscriptSegment]
    ) -> tuple[float, str]:
        normalized_quote = cls._normalize(quote)
        matches: list[tuple[float, float, str]] = []
        for start_index, segment in enumerate(segments):
            for end_index in range(start_index, min(start_index + 24, len(segments))):
                original_text = " ".join(
                    item.text for item in segments[start_index : end_index + 1]
                )
                if normalized_quote in cls._normalize(original_text):
                    distance = abs(segment.start - proposed_timestamp)
                    matches.append((distance, segment.start, original_text))
                    break
        if not matches:
            raise GroundingError(f"Slide source_quote 无法在 Source Segment 中逐字找到：{quote!r}")
        distance, timestamp, original_text = min(matches, key=lambda item: (item[0], len(item[2])))
        if distance > 12:
            raise GroundingError(
                f"Slide source_quote 与建议时间戳相差 {distance:.2f} 秒，超过 12 秒限制"
            )
        return timestamp, original_text

    @staticmethod
    def _normalize(value: str) -> str:
        return "".join(re.findall(r"[\w]+", value.casefold()))
