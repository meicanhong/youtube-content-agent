from pathlib import Path
from typing import Annotated

import typer

from .config import Settings
from .errors import AgentError
from .logging_config import configure_logging
from .rerender import PackageRerenderer
from .visual import SlideRenderer

app = typer.Typer(no_args_is_help=True, help="仅用已有内容和视频缓存重新生成图片")


@app.command()
def run(
    target: Annotated[Path, typer.Argument(help="Content Package 或整次输出目录")],
    work_dir: Annotated[
        Path,
        typer.Option(help="已有视频片段缓存根目录"),
    ] = Path("work/youtube-content-agent"),
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Rerender saved packages without creating or calling an Editorial provider."""
    configure_logging(verbose)
    settings = Settings()
    try:
        outputs = PackageRerenderer(
            SlideRenderer(settings.ffmpeg_bin),
            work_dir,
        ).rerender(target)
    except AgentError as exc:
        typer.echo(f"失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"完成：未调用 LLM，重新生成 {len(outputs)} 张故事长图")
    for output in outputs:
        typer.echo(str(output.resolve()))


if __name__ == "__main__":
    app()
