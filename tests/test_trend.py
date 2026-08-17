from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from youtube_content_agent.errors import ExternalToolError
from youtube_content_agent.trend import TrendRulePolicy, TrendService
from youtube_content_agent.trend_ai import FixtureTrendRanker
from youtube_content_agent.trend_cli import app
from youtube_content_agent.trend_sources import (
    FixturePodcastSeedGateway,
    FixtureTrendVideoGateway,
    PodcastChartGateway,
    YouTubeDataGateway,
    YtDlpTrendVideoGateway,
)
from youtube_content_agent.trend_storage import write_trend_report

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_chart_json_ld_parser_reads_ranked_entries() -> None:
    document = """
    <script type="application/ld+json">
    {"@type":"CollectionPage","mainEntity":{"itemListElement":[
      {"position":2,"name":"Second","url":"https://podchartdb.com/podcasts/2"},
      {"position":1,"name":"First","url":"https://podchartdb.com/podcasts/1"}
    ]}}
    </script>
    """

    assert PodcastChartGateway._parse_chart_entries(document) == [
        (1, "First", "https://podchartdb.com/podcasts/1"),
        (2, "Second", "https://podchartdb.com/podcasts/2"),
    ]


def test_youtube_duration_parser() -> None:
    assert YouTubeDataGateway._parse_duration("PT2H3M4S") == 7384
    assert YouTubeDataGateway._parse_duration("P1DT5M") == 86700
    with pytest.raises(ExternalToolError):
        YouTubeDataGateway._parse_duration("")


def test_rule_policy_rejects_clips_and_missing_captions() -> None:
    gateway = FixtureTrendVideoGateway(FIXTURES / "trend_videos.json")
    videos = gateway.fetch_month_videos(
        [],
        datetime(2026, 8, 1, tzinfo=UTC),
        datetime(2026, 9, 1, tzinfo=UTC),
        25,
    )

    assessments = TrendRulePolicy().assess(videos, datetime(2026, 8, 17, tzinfo=UTC))

    assert assessments["clip001"].eligible is False
    assert "clip_or_promo_title" in assessments["clip001"].reasons
    assert assessments["nocap001"].reasons == ["captions_unavailable"]
    assert assessments["doac001"].eligible is True


def test_yt_dlp_fallback_maps_public_metadata() -> None:
    seed = FixturePodcastSeedGateway(FIXTURES / "trend_seeds.json").fetch_seeds(1)[0]
    raw = {
        "id": "fallback001",
        "webpage_url": "https://www.youtube.com/watch?v=fallback001",
        "title": "A Full Interview",
        "description": "Long-form conversation",
        "channel": "Example Channel",
        "timestamp": datetime(2026, 8, 10, tzinfo=UTC).timestamp(),
        "duration": 3600,
        "view_count": 250000,
        "live_status": "not_live",
        "subtitles": {},
        "automatic_captions": {"en": [{"ext": "vtt"}]},
    }

    video = YtDlpTrendVideoGateway._map_entry(
        raw,
        seed,
        datetime(2026, 8, 1, tzinfo=UTC),
        datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert video is not None
    assert video.video_id == "fallback001"
    assert video.has_captions is True
    assert video.view_count == 250000


def test_trend_service_and_report_are_auditable(tmp_path: Path) -> None:
    service = TrendService(
        FixturePodcastSeedGateway(FIXTURES / "trend_seeds.json"),
        FixtureTrendVideoGateway(FIXTURES / "trend_videos.json"),
        FixtureTrendRanker(FIXTURES / "trend_ai.json"),
    )

    report = service.run("2026-08", max_seeds=3, preselect=3, top_n=3)
    write_trend_report(tmp_path, report)

    assert report.seed_count == 3
    assert report.candidate_count == 5
    assert report.eligible_count == 3
    assert report.video_source == "fixture:trend_videos.json"
    assert report.ranked[0].video.video_id == "doac001"
    assert report.excluded_reason_counts == {
        "captions_unavailable": 1,
        "clip_or_promo_title": 1,
        "duration_too_short": 1,
    }
    assert (tmp_path / "trend-report.json").exists()
    assert "doac001" in (tmp_path / "top10.md").read_text(encoding="utf-8")


def test_trend_cli_runs_without_network_or_paid_ai(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--month",
            "2026-08",
            "--output-dir",
            str(tmp_path),
            "--top-n",
            "3",
            "--preselect",
            "3",
            "--max-seeds",
            "3",
            "--seed-fixture",
            str(FIXTURES / "trend_seeds.json"),
            "--video-fixture",
            str(FIXTURES / "trend_videos.json"),
            "--ai-fixture",
            str(FIXTURES / "trend_ai.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "选出 3 条视频" in result.output
    assert (tmp_path / "2026-08" / "trend-report.json").exists()
