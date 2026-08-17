from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

SOURCE_QUOTE_MAX_LENGTH = 2000


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoMetadata(StrictModel):
    video_id: str
    youtube_url: HttpUrl
    title: str
    channel: str
    published_at: datetime | None = None
    duration: float = Field(gt=0)
    view_count: int | None = Field(default=None, ge=0)
    thumbnail_url: HttpUrl | None = None


class TranscriptSegment(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def end_after_start(self) -> TranscriptSegment:
        if self.end <= self.start:
            raise ValueError("Transcript segment end must be after start")
        return self


class Transcript(StrictModel):
    video_id: str
    language: str
    source: str
    segments: list[TranscriptSegment] = Field(min_length=1)


class SlideProposal(StrictModel):
    timestamp: float = Field(ge=0)
    source_quote: str = Field(min_length=3, max_length=SOURCE_QUOTE_MAX_LENGTH)
    zh_text: str = Field(min_length=2, max_length=80)


class CaptionDraft(StrictModel):
    title: str = Field(min_length=2, max_length=80)
    hook: str = Field(min_length=2, max_length=160)
    body: str = Field(min_length=20)


class TopicProposal(StrictModel):
    topic: str = Field(min_length=2, max_length=80)
    source_start: float = Field(ge=0)
    source_end: float = Field(gt=0)
    slides: list[SlideProposal] = Field(min_length=6, max_length=14)
    caption: CaptionDraft
    quality_score: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_timeline(self) -> TopicProposal:
        if self.source_end <= self.source_start:
            raise ValueError("source_end must be after source_start")
        if not 20 <= self.source_end - self.source_start <= 240:
            raise ValueError("source segment must be between 20 and 240 seconds")
        timestamps = [slide.timestamp for slide in self.slides]
        if timestamps != sorted(timestamps):
            raise ValueError("slide timestamps must be ascending")
        if any(not self.source_start <= value <= self.source_end for value in timestamps):
            raise ValueError("every slide timestamp must be inside the source segment")
        return self


class EditorialResponse(StrictModel):
    topics: list[TopicProposal] = Field(max_length=6)


class StoryCoherenceIssue(StrictModel):
    slide_index: int = Field(ge=1, le=14)
    category: Literal[
        "missing_bridge",
        "dangling_connector",
        "ambiguous_reference",
        "abrupt_transition",
        "overloaded_slide",
        "translation_clarity",
    ]
    explanation: str = Field(min_length=3, max_length=1000)
    repair_instruction: str = Field(min_length=3, max_length=1000)


class TopicCoherenceAudit(StrictModel):
    topic_index: int = Field(ge=1, le=6)
    coherent: bool
    score: float = Field(ge=0, le=1)
    issues: list[StoryCoherenceIssue] = Field(max_length=20)


class StoryCoherenceAudit(StrictModel):
    topics: list[TopicCoherenceAudit] = Field(max_length=6)


class SourceSegment(StrictModel):
    start: float
    end: float
    original_text: str
    transcript_segments: list[TranscriptSegment]


class GroundedSlide(StrictModel):
    index: int = Field(ge=1)
    timestamp: float
    original_text: str
    zh_text: str
    image: str | None = None


class ContentData(StrictModel):
    topic: str
    hook: str
    slides: list[GroundedSlide]
    caption: CaptionDraft
    quality_score: float


class SourceData(StrictModel):
    video_id: str
    youtube_url: HttpUrl
    title: str
    channel: str
    published_at: datetime | None
    source_segment: SourceSegment


class PackageMetadata(StrictModel):
    schema_version: str = "1.0"
    generated_at: datetime
    transcript_source: str
    editorial_provider: str
    image_count: int = Field(ge=0)
    storyboard_image: str | None = None
    storyboard_images: list[str] = Field(default_factory=list)
    complete: bool


class PackageResult(StrictModel):
    directory: Path
    source: SourceData
    content: ContentData
    metadata: PackageMetadata


class RunManifest(StrictModel):
    schema_version: str = "1.0"
    generated_at: datetime
    video: VideoMetadata
    transcript_file: str
    packages: list[str]
    editorial_provider: str
