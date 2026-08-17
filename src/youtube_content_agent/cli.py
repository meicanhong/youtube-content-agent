from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .config import Settings
from .editorial import FixtureEditorialProvider, MimoEditorialProvider, OpenAIEditorialProvider
from .errors import AgentError
from .interfaces import EditorialProvider
from .logging_config import configure_logging
from .pipeline import ContentPipeline
from .visual import SlideRenderer
from .youtube import YtDlpGateway

app = typer.Typer(no_args_is_help=True, help="YouTube 访谈中文图文 Content Package Agent")


@app.command()
def run(
    url: Annotated[str, typer.Argument(help="YouTube video URL")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o", help="Output directory")] = Path(
        "outputs/run"
    ),
    editorial_fixture: Annotated[
        Path | None,
        typer.Option(help="Explicit offline Editorial JSON; never used implicitly"),
    ] = None,
    work_dir: Annotated[
        Path,
        typer.Option(help="Intermediate cache root (transcript and source media)"),
    ] = Path("work/youtube-content-agent"),
    max_topics: Annotated[int, typer.Option(min=1, max=5)] = 3,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Generate traceable Chinese carousel packages from one YouTube URL."""
    configure_logging(verbose)
    settings = Settings()
    try:
        editorial: EditorialProvider
        if editorial_fixture:
            editorial = FixtureEditorialProvider(editorial_fixture)
        elif settings.editorial_provider == "mimo":
            editorial = MimoEditorialProvider(
                settings.mimo_api_key,
                settings.mimo_model,
                settings.mimo_base_url,
            )
        else:
            editorial = OpenAIEditorialProvider(settings.openai_api_key, settings.openai_model)
        pipeline = ContentPipeline(
            youtube=YtDlpGateway(settings),
            editorial=editorial,
            renderer=SlideRenderer(settings.ffmpeg_bin),
        )
        results = pipeline.run(url, output_dir, max_topics, work_dir)
    except AgentError as exc:
        typer.echo(f"失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"完成：生成 {len(results)} 个 Content Package")
    for result in results:
        typer.echo(str(result.directory.resolve()))


if __name__ == "__main__":
    app()
