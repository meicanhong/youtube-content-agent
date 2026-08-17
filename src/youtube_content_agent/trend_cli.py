from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from .config import Settings
from .editorial import MimoEditorialProvider, OpenAIEditorialProvider
from .errors import AgentError, ConfigurationError
from .interfaces import EditorialProvider
from .logging_config import configure_logging
from .pipeline import ContentPipeline
from .storage import slugify
from .trend import TrendService
from .trend_ai import FixtureTrendRanker, MimoTrendRanker
from .trend_interfaces import PodcastSeedGateway, TrendAiRanker, TrendVideoGateway
from .trend_models import GenerationResult, RankedTrendVideo
from .trend_sources import (
    FixturePodcastSeedGateway,
    FixtureTrendVideoGateway,
    PodcastChartGateway,
    YouTubeDataGateway,
    YtDlpTrendVideoGateway,
)
from .trend_storage import write_trend_report
from .visual import SlideRenderer
from .youtube import YtDlpGateway

app = typer.Typer(no_args_is_help=False, help="AI + Rule YouTube 播客访谈发现 Agent")


@app.command()
def run(
    month: Annotated[str | None, typer.Option(help="Target month in YYYY-MM")] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("outputs/trends"),
    work_dir: Annotated[Path, typer.Option(help="Cache and intermediate files")] = Path(
        "work/trend-agent"
    ),
    top_n: Annotated[int, typer.Option(min=1, max=20)] = 10,
    max_seeds: Annotated[int, typer.Option(min=1, max=100)] = 100,
    preselect: Annotated[int, typer.Option(min=1, max=50)] = 30,
    episodes_per_show: Annotated[int, typer.Option(min=1, max=50)] = 25,
    generate_top: Annotated[int, typer.Option(min=0, max=10)] = 0,
    max_topics: Annotated[int, typer.Option(min=1, max=5)] = 1,
    seed_fixture: Annotated[Path | None, typer.Option()] = None,
    video_fixture: Annotated[Path | None, typer.Option()] = None,
    ai_fixture: Annotated[Path | None, typer.Option()] = None,
    no_youtube_api: Annotated[
        bool, typer.Option("--no-youtube-api", help="Use slower yt-dlp metadata fallback")
    ] = False,
    cookies_from_browser: Annotated[
        str | None, typer.Option(help="Browser session for yt-dlp, e.g. chrome")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Discover and rank this month's publishable podcast/interview videos."""
    configure_logging(verbose)
    selected_month = month or datetime.now(UTC).strftime("%Y-%m")
    run_output = output_dir / selected_month
    settings = Settings()
    try:
        if generate_top > top_n:
            raise ConfigurationError("generate_top 不能大于 top_n")
        seed_gateway, video_gateway = _build_sources(
            settings,
            work_dir,
            seed_fixture,
            video_fixture,
            no_youtube_api,
            cookies_from_browser,
        )
        ai_ranker: TrendAiRanker
        if ai_fixture:
            ai_ranker = FixtureTrendRanker(ai_fixture)
        else:
            ai_ranker = MimoTrendRanker(
                settings.mimo_api_key, settings.mimo_model, settings.mimo_base_url
            )
        report = TrendService(seed_gateway, video_gateway, ai_ranker).run(
            selected_month,
            max_seeds=max_seeds,
            episodes_per_show=episodes_per_show,
            preselect=preselect,
            top_n=top_n,
        )
        write_trend_report(run_output, report)
        if generate_top:
            generation_results = _generate_ranked(
                report.ranked[:generate_top], settings, run_output, work_dir, max_topics
            )
            report = report.model_copy(update={"generation_results": generation_results})
            write_trend_report(run_output, report)
    except AgentError as exc:
        typer.echo(f"失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"完成：从 {report.seed_count} 个种子节目选出 {len(report.ranked)} 条视频")
    typer.echo(str((run_output / "top10.md").resolve()))
    for item in report.ranked:
        typer.echo(f"{item.rank:02d}  {item.final_score:5.2f}  {item.video.youtube_url}")


def _build_sources(
    settings: Settings,
    work_dir: Path,
    seed_fixture: Path | None,
    video_fixture: Path | None,
    no_youtube_api: bool,
    cookies_from_browser: str | None,
) -> tuple[PodcastSeedGateway, TrendVideoGateway]:
    if (seed_fixture is None) != (video_fixture is None):
        raise ConfigurationError("--seed-fixture 和 --video-fixture 必须同时提供")
    if seed_fixture and video_fixture:
        return FixturePodcastSeedGateway(seed_fixture), FixtureTrendVideoGateway(video_fixture)
    if no_youtube_api or not settings.youtube_api_key:
        return (
            PodcastChartGateway(cache_path=work_dir / "podcast-seeds.json"),
            YtDlpTrendVideoGateway(
                settings.yt_dlp_bin,
                cookies_from_browser or settings.yt_dlp_cookies_from_browser,
                work_dir / "yt-dlp-trend-cache",
            ),
        )
    return (
        PodcastChartGateway(cache_path=work_dir / "podcast-seeds.json"),
        YouTubeDataGateway(settings.youtube_api_key),
    )


def _generate_ranked(
    ranked: list[RankedTrendVideo],
    settings: Settings,
    output_dir: Path,
    work_dir: Path,
    max_topics: int,
) -> list[GenerationResult]:
    editorial: EditorialProvider
    if settings.editorial_provider == "mimo":
        editorial = MimoEditorialProvider(
            settings.mimo_api_key, settings.mimo_model, settings.mimo_base_url
        )
    else:
        editorial = OpenAIEditorialProvider(settings.openai_api_key, settings.openai_model)
    pipeline = ContentPipeline(
        youtube=YtDlpGateway(settings),
        editorial=editorial,
        renderer=SlideRenderer(settings.ffmpeg_bin),
    )
    results: list[GenerationResult] = []
    for item in ranked:
        video = item.video
        destination = (
            output_dir / "content" / (f"{item.rank:02d}-{slugify(video.title, video.video_id)}")
        )
        try:
            pipeline.run(video.youtube_url, destination, max_topics, work_dir / "content")
        except AgentError as exc:
            results.append(
                GenerationResult(
                    video_id=video.video_id,
                    youtube_url=video.youtube_url,
                    status="failed",
                    error=str(exc),
                )
            )
            continue
        results.append(
            GenerationResult(
                video_id=video.video_id,
                youtube_url=video.youtube_url,
                status="complete",
                output_dir=str(destination.resolve()),
            )
        )
    return results


if __name__ == "__main__":
    app()
