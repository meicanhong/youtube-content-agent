from __future__ import annotations

import calendar
import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime

from .errors import ConfigurationError, ExternalToolError
from .trend_interfaces import PodcastSeedGateway, TrendAiRanker, TrendVideoGateway
from .trend_models import (
    AiAssessment,
    RankedTrendVideo,
    RuleAssessment,
    TrendReport,
    TrendVideo,
)


class TrendRulePolicy:
    EXCLUDED_TITLE_RE = re.compile(
        r"\b(shorts?|clips?|highlights?|trailer|teaser|preview|best moments?|compilation)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        min_duration_seconds: int = 20 * 60,
        max_duration_seconds: int = 4 * 60 * 60,
        min_views: int = 10_000,
        require_captions: bool = True,
    ) -> None:
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds
        self.min_views = min_views
        self.require_captions = require_captions

    def assess(self, videos: list[TrendVideo], as_of: datetime) -> dict[str, RuleAssessment]:
        raw: dict[str, tuple[list[str], float]] = {}
        for video in videos:
            reasons = self._rejection_reasons(video)
            age_days = max(1.0, (as_of - video.published_at).total_seconds() / 86400 + 1)
            raw[video.video_id] = (reasons, video.view_count / age_days)

        eligible = [video for video in videos if not raw[video.video_id][0]]
        view_logs = [math.log1p(video.view_count) for video in eligible]
        velocity_logs = [math.log1p(raw[video.video_id][1]) for video in eligible]
        view_bounds = self._bounds(view_logs)
        velocity_bounds = self._bounds(velocity_logs)

        assessments: dict[str, RuleAssessment] = {}
        for video in videos:
            reasons, views_per_day = raw[video.video_id]
            if reasons:
                score = 0.0
            else:
                view_score = self._normalize(math.log1p(video.view_count), view_bounds)
                velocity_score = self._normalize(math.log1p(views_per_day), velocity_bounds)
                chart_score = (101 - video.seed_rank) / 100
                age_days = max(0.0, (as_of - video.published_at).total_seconds() / 86400)
                recency_score = max(0.0, 1 - age_days / 31)
                score = 100 * (
                    view_score * 0.30
                    + velocity_score * 0.35
                    + chart_score * 0.20
                    + recency_score * 0.15
                )
            assessments[video.video_id] = RuleAssessment(
                eligible=not reasons,
                reasons=reasons,
                views_per_day=round(views_per_day, 2),
                rule_score=round(score, 2),
            )
        return assessments

    def _rejection_reasons(self, video: TrendVideo) -> list[str]:
        reasons: list[str] = []
        if video.duration_seconds < self.min_duration_seconds:
            reasons.append("duration_too_short")
        if video.duration_seconds > self.max_duration_seconds:
            reasons.append("duration_too_long")
        if video.view_count < self.min_views:
            reasons.append("views_below_minimum")
        if self.require_captions and not video.has_captions:
            reasons.append("captions_unavailable")
        if video.is_live:
            reasons.append("live_or_archived_stream")
        if self.EXCLUDED_TITLE_RE.search(video.title):
            reasons.append("clip_or_promo_title")
        return reasons

    @staticmethod
    def _bounds(values: list[float]) -> tuple[float, float]:
        return (min(values), max(values)) if values else (0.0, 0.0)

    @staticmethod
    def _normalize(value: float, bounds: tuple[float, float]) -> float:
        lower, upper = bounds
        if upper <= lower:
            return 1.0 if upper > 0 else 0.0
        return (value - lower) / (upper - lower)


class TrendService:
    def __init__(
        self,
        seeds: PodcastSeedGateway,
        videos: TrendVideoGateway,
        ai: TrendAiRanker,
        rules: TrendRulePolicy | None = None,
    ) -> None:
        self.seeds = seeds
        self.videos = videos
        self.ai = ai
        self.rules = rules or TrendRulePolicy()

    def run(
        self,
        month: str,
        *,
        max_seeds: int = 100,
        episodes_per_show: int = 25,
        preselect: int = 30,
        top_n: int = 10,
    ) -> TrendReport:
        month_start, month_end = self._month_bounds(month)
        if not 1 <= top_n <= preselect <= 50:
            raise ConfigurationError("需要满足 1 <= top_n <= preselect <= 50")
        as_of = datetime.now(UTC)
        seed_list = self.seeds.fetch_seeds(max_seeds)
        candidates = self.videos.fetch_month_videos(
            seed_list, month_start, month_end, episodes_per_show
        )
        if not candidates:
            raise ExternalToolError(f"{month} 没有读取到任何榜单候选视频")
        assessments = self.rules.assess(candidates, as_of)
        eligible = [video for video in candidates if assessments[video.video_id].eligible]
        if not eligible:
            raise ExternalToolError("规则过滤后没有符合条件的完整字幕访谈")
        ai_candidates = self._preselect(eligible, assessments, preselect)
        ai_assessments = self.ai.assess(ai_candidates)
        ranked = self._rank(ai_candidates, assessments, ai_assessments, top_n)
        excluded_counts = Counter(
            reason for assessment in assessments.values() for reason in assessment.reasons
        )
        return TrendReport(
            generated_at=as_of,
            month=month,
            chart_source=self.seeds.source_name,
            video_source=self.videos.source_name,
            seed_count=len(seed_list),
            candidate_count=len(candidates),
            eligible_count=len(eligible),
            ai_provider=self.ai.name,
            excluded_reason_counts=dict(sorted(excluded_counts.items())),
            seeds=seed_list,
            ranked=ranked,
        )

    @staticmethod
    def _preselect(
        videos: list[TrendVideo],
        assessments: dict[str, RuleAssessment],
        limit: int,
    ) -> list[TrendVideo]:
        ordered = sorted(
            videos,
            key=lambda video: assessments[video.video_id].rule_score,
            reverse=True,
        )
        selected: list[TrendVideo] = []
        per_show: defaultdict[str, int] = defaultdict(int)
        for video in ordered:
            if per_show[video.seed_name] >= 3:
                continue
            selected.append(video)
            per_show[video.seed_name] += 1
            if len(selected) == limit:
                break
        return selected

    @staticmethod
    def _rank(
        videos: list[TrendVideo],
        rules: dict[str, RuleAssessment],
        ai_assessments: list[AiAssessment],
        top_n: int,
    ) -> list[RankedTrendVideo]:
        ai_by_id = {assessment.video_id: assessment for assessment in ai_assessments}
        scored = sorted(
            videos,
            key=lambda video: (
                rules[video.video_id].rule_score * 0.55
                + ai_by_id[video.video_id].content_score * 0.45
            ),
            reverse=True,
        )
        selected: list[TrendVideo] = []
        per_show: defaultdict[str, int] = defaultdict(int)
        for video in scored:
            if per_show[video.seed_name] >= 2:
                continue
            selected.append(video)
            per_show[video.seed_name] += 1
            if len(selected) == top_n:
                break
        if len(selected) < top_n:
            selected_ids = {video.video_id for video in selected}
            selected.extend(video for video in scored if video.video_id not in selected_ids)
            selected = selected[:top_n]
        return [
            RankedTrendVideo(
                rank=index,
                video=video,
                rule=rules[video.video_id],
                ai=ai_by_id[video.video_id],
                final_score=round(
                    rules[video.video_id].rule_score * 0.55
                    + ai_by_id[video.video_id].content_score * 0.45,
                    2,
                ),
            )
            for index, video in enumerate(selected, start=1)
        ]

    @staticmethod
    def _month_bounds(month: str) -> tuple[datetime, datetime]:
        try:
            year, month_number = (int(part) for part in month.split("-"))
            calendar.monthrange(year, month_number)
        except (ValueError, TypeError) as exc:
            raise ConfigurationError("month 必须是 YYYY-MM，例如 2026-08") from exc
        start = datetime(year, month_number, 1, tzinfo=UTC)
        if month_number == 12:
            end = datetime(year + 1, 1, 1, tzinfo=UTC)
        else:
            end = datetime(year, month_number + 1, 1, tzinfo=UTC)
        return start, end
