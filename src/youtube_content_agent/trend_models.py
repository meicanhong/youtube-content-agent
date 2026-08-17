from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrendStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PodcastSeed(TrendStrictModel):
    chart_rank: int = Field(ge=1, le=100)
    podcast_name: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    playlist_url: str = Field(pattern=r"^https://www\.youtube\.com/playlist\?list=")


class TrendVideo(TrendStrictModel):
    video_id: str = Field(min_length=3)
    youtube_url: str = Field(pattern=r"^https://www\.youtube\.com/watch\?v=")
    title: str = Field(min_length=1)
    description: str = ""
    channel: str = Field(min_length=1)
    published_at: datetime
    duration_seconds: int = Field(gt=0)
    view_count: int = Field(ge=0)
    has_captions: bool
    is_live: bool = False
    seed_rank: int = Field(ge=1, le=100)
    seed_name: str = Field(min_length=1)


class RuleAssessment(TrendStrictModel):
    eligible: bool
    reasons: list[str]
    views_per_day: float = Field(ge=0)
    rule_score: float = Field(ge=0, le=100)


class AiAssessment(TrendStrictModel):
    video_id: str
    china_interest: float = Field(ge=0, le=100)
    insight_density: float = Field(ge=0, le=100)
    narrative_arc: float = Field(ge=0, le=100)
    visual_suitability: float = Field(ge=0, le=100)
    guest_recognition: float = Field(ge=0, le=100)
    risk_score: float = Field(ge=0, le=100)
    reason: str = Field(min_length=2, max_length=240)

    @property
    def content_score(self) -> float:
        positive = (
            self.china_interest * 0.30
            + self.insight_density * 0.25
            + self.narrative_arc * 0.20
            + self.visual_suitability * 0.15
            + self.guest_recognition * 0.10
        )
        return max(0.0, min(100.0, positive - self.risk_score * 0.25))


class TrendAiResponse(TrendStrictModel):
    assessments: list[AiAssessment] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def unique_video_ids(self) -> TrendAiResponse:
        ids = [item.video_id for item in self.assessments]
        if len(ids) != len(set(ids)):
            raise ValueError("AI assessments contain duplicate video IDs")
        return self


class RankedTrendVideo(TrendStrictModel):
    rank: int = Field(ge=1)
    video: TrendVideo
    rule: RuleAssessment
    ai: AiAssessment
    final_score: float = Field(ge=0, le=100)


class GenerationResult(TrendStrictModel):
    video_id: str
    youtube_url: str
    status: str
    output_dir: str | None = None
    error: str | None = None


class TrendReport(TrendStrictModel):
    schema_version: str = "1.0"
    generated_at: datetime
    month: str = Field(pattern=r"^\d{4}-\d{2}$")
    chart_source: str
    seed_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    eligible_count: int = Field(ge=0)
    ai_provider: str
    excluded_reason_counts: dict[str, int]
    seeds: list[PodcastSeed]
    ranked: list[RankedTrendVideo]
    generation_results: list[GenerationResult] = Field(default_factory=list)
