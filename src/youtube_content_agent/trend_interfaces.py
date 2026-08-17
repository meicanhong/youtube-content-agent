from datetime import datetime
from typing import Protocol

from .trend_models import AiAssessment, PodcastSeed, TrendVideo


class PodcastSeedGateway(Protocol):
    @property
    def source_name(self) -> str: ...

    def fetch_seeds(self, limit: int) -> list[PodcastSeed]: ...


class TrendVideoGateway(Protocol):
    @property
    def source_name(self) -> str: ...

    def fetch_month_videos(
        self,
        seeds: list[PodcastSeed],
        month_start: datetime,
        month_end: datetime,
        episodes_per_show: int,
    ) -> list[TrendVideo]: ...


class TrendAiRanker(Protocol):
    @property
    def name(self) -> str: ...

    def assess(self, videos: list[TrendVideo]) -> list[AiAssessment]: ...
