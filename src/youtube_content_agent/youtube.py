from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from .config import Settings
from .errors import ExternalToolError, TranscriptUnavailableError
from .models import Transcript, VideoMetadata
from .vtt import parse_json3, parse_vtt

logger = logging.getLogger(__name__)


class YtDlpGateway:
    """Boundary adapter for public YouTube metadata, subtitles and media sections."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _base_command(self) -> list[str]:
        command = [self.settings.yt_dlp_bin, "--no-playlist", "--no-warnings"]
        if self.settings.yt_dlp_cookies_from_browser:
            command.extend(["--cookies-from-browser", self.settings.yt_dlp_cookies_from_browser])
        return command

    def _run(self, command: list[str], operation: str) -> subprocess.CompletedProcess[str]:
        logger.info(
            "external tool started",
            extra={"event": "external_call", "operation": operation, "provider": "yt-dlp"},
        )
        for attempt in range(1, 3):
            try:
                result = subprocess.run(command, text=True, capture_output=True, check=False)
            except OSError as exc:
                raise ExternalToolError(f"无法启动 yt-dlp：{exc}") from exc
            if result.returncode == 0:
                return result
            detail = self._safe_error(result.stderr)
            retryable = any(marker in detail for marker in ("429", "HTTP Error 5"))
            if not retryable or attempt == 2:
                raise ExternalToolError(f"yt-dlp {operation} 失败：{detail}")
            logger.warning(
                "temporary YouTube failure; retrying",
                extra={
                    "event": "external_retry",
                    "operation": operation,
                    "provider": "yt-dlp",
                },
            )
            time.sleep(attempt)
        raise AssertionError("unreachable")

    @staticmethod
    def _safe_error(stderr: str) -> str:
        last_line = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
        return re.sub(r"https?://\S+", "<url>", last_line)

    def fetch_metadata(self, url: str) -> VideoMetadata:
        result = self._run(
            [*self._base_command(), "--dump-single-json", "--skip-download", url],
            "metadata",
        )
        raw = json.loads(result.stdout)
        timestamp = raw.get("timestamp")
        thumbnail = raw.get("thumbnail")
        return VideoMetadata(
            video_id=raw["id"],
            youtube_url=raw.get("webpage_url") or url,
            title=raw["title"],
            channel=raw.get("channel") or raw.get("uploader") or "Unknown",
            published_at=datetime.fromtimestamp(timestamp, UTC) if timestamp else None,
            duration=float(raw["duration"]),
            view_count=raw.get("view_count"),
            thumbnail_url=thumbnail if thumbnail and thumbnail.startswith("http") else None,
        )

    def fetch_transcript(self, metadata: VideoMetadata, work_dir: Path) -> Transcript:
        work_dir.mkdir(parents=True, exist_ok=True)
        cache_path = work_dir / f"{metadata.video_id}.transcript.json"
        if cache_path.exists():
            try:
                return Transcript.model_validate_json(cache_path.read_text(encoding="utf-8"))
            except ValueError:
                logger.warning(
                    "invalid transcript cache ignored",
                    extra={
                        "event": "cache_invalid",
                        "operation": "transcript",
                        "resource_id": metadata.video_id,
                    },
                )
        with TemporaryDirectory(dir=work_dir) as temp_name:
            template = str(Path(temp_name) / "subtitle.%(ext)s")
            command = [
                *self._base_command(),
                "--skip-download",
                "--write-subs",
                "--write-auto-subs",
                "--sub-langs",
                "en",
                "--sub-format",
                "json3/vtt",
                "-o",
                template,
                str(metadata.youtube_url),
            ]
            self._run(command, "transcript")
            candidates = sorted(
                [*Path(temp_name).glob("*.json3"), *Path(temp_name).glob("*.vtt")],
                key=self._subtitle_priority,
            )
            for candidate in candidates:
                segments = (
                    parse_json3(candidate) if candidate.suffix == ".json3" else parse_vtt(candidate)
                )
                if segments:
                    language = self._language_from_name(candidate.name)
                    transcript = Transcript(
                        video_id=metadata.video_id,
                        language=language,
                        source="youtube_manual_or_auto_captions",
                        segments=segments,
                    )
                    cache_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
                    return transcript
        raise TranscriptUnavailableError(
            "该视频没有可用英文字幕。当前 MVP 不做隐式 ASR 降级，请选择有字幕的视频。"
        )

    @staticmethod
    def _subtitle_priority(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        return (0 if ".en." in name or name.endswith(".en.vtt") else 1, name)

    @staticmethod
    def _language_from_name(name: str) -> str:
        parts = name.split(".")
        return parts[-2] if len(parts) >= 3 else "en"

    def download_segment(
        self, metadata: VideoMetadata, start: float, end: float, destination: Path
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            return self._download_remote_segment(metadata, start, end, destination)
        except ExternalToolError:
            logger.warning(
                "remote section download failed; using cached full-video fallback",
                extra={
                    "event": "fallback_used",
                    "operation": "download_segment",
                    "provider": "yt-dlp+ffmpeg",
                    "resource_id": metadata.video_id,
                },
            )
            return self._download_full_then_cut(metadata, start, end, destination)

    def _download_remote_segment(
        self, metadata: VideoMetadata, start: float, end: float, destination: Path
    ) -> Path:
        with TemporaryDirectory(dir=destination.parent) as temp_name:
            output = Path(temp_name) / "segment.%(ext)s"
            command = [
                *self._base_command(),
                "-f",
                "bv*[height<=720]+ba/b[height<=720]/b",
                "--merge-output-format",
                "mp4",
                "--download-sections",
                f"*{max(0, start):.3f}-{min(metadata.duration, end):.3f}",
                "--force-keyframes-at-cuts",
                "-o",
                str(output),
                str(metadata.youtube_url),
            ]
            self._run(command, "download_segment")
            files = [path for path in Path(temp_name).iterdir() if path.is_file()]
            if not files:
                raise ExternalToolError("yt-dlp 未生成视频片段")
            media = max(files, key=lambda path: path.stat().st_size)
            shutil.move(str(media), destination)
        return destination

    def _download_full_then_cut(
        self, metadata: VideoMetadata, start: float, end: float, destination: Path
    ) -> Path:
        cached_sources = sorted(destination.parent.glob(f"{metadata.video_id}.source.*"))
        if cached_sources:
            source = cached_sources[0]
        else:
            template = destination.parent / f"{metadata.video_id}.source.%(ext)s"
            self._run(
                [
                    *self._base_command(),
                    "-f",
                    "b[height<=480]/b",
                    "--merge-output-format",
                    "mp4",
                    "-o",
                    str(template),
                    str(metadata.youtube_url),
                ],
                "download_full_video_fallback",
            )
            cached_sources = sorted(destination.parent.glob(f"{metadata.video_id}.source.*"))
            if not cached_sources:
                raise ExternalToolError("完整视频 fallback 未生成媒体文件")
            source = cached_sources[0]

        duration = min(metadata.duration, end) - max(0, start)
        command = [
            self.settings.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0, start):.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-y",
            str(destination),
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0 or not destination.exists():
            raise ExternalToolError(f"本地 FFmpeg 裁切失败：{self._safe_error(result.stderr)}")
        return destination
