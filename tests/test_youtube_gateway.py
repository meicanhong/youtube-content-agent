from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from youtube_content_agent.config import Settings
from youtube_content_agent.models import Transcript, TranscriptSegment, VideoMetadata
from youtube_content_agent.youtube import YtDlpGateway


def test_fetch_transcript_reuses_valid_cache(tmp_path: Path) -> None:
    gateway = YtDlpGateway(Settings())
    metadata = VideoMetadata(
        video_id="cached",
        youtube_url="https://www.youtube.com/watch?v=cached",
        title="Cached",
        channel="Channel",
        duration=60,
    )
    cached = Transcript(
        video_id="cached",
        language="en",
        source="fixture-cache",
        segments=[TranscriptSegment(start=1, end=2, text="cached text")],
    )
    cache_path = tmp_path / "cached.transcript.json"
    cache_path.write_text(cached.model_dump_json(), encoding="utf-8")
    loaded = gateway.fetch_transcript(metadata, tmp_path)
    assert loaded == cached


def test_safe_error_removes_urls() -> None:
    error = YtDlpGateway._safe_error(
        "warning\nERROR: failed https://example.com/media?token=secret"
    )
    assert error == "ERROR: failed <url>"
    assert "secret" not in error


def test_run_retries_429_once(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = 0

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess([], 1, "", "HTTP Error 429")
        return subprocess.CompletedProcess([], 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("youtube_content_agent.youtube.time.sleep", lambda _: None)
    result = YtDlpGateway(Settings())._run(["yt-dlp"], "test")
    assert result.stdout == "ok"
    assert calls == 2


def test_fetch_metadata_does_not_require_downloadable_formats(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: list[str] = []
    payload = {
        "id": "video-id",
        "webpage_url": "https://www.youtube.com/watch?v=video-id",
        "title": "Interview",
        "channel": "Channel",
        "duration": 3600,
    }

    def fake_run(command, operation):  # type: ignore[no-untyped-def]
        captured.extend(command)
        assert operation == "metadata"
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    gateway = YtDlpGateway(Settings())
    monkeypatch.setattr(gateway, "_run", fake_run)

    metadata = gateway.fetch_metadata("https://www.youtube.com/watch?v=video-id")

    assert metadata.video_id == "video-id"
    assert "--ignore-no-formats-error" in captured


def test_fetch_transcript_does_not_require_downloadable_formats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[str] = []
    metadata = VideoMetadata(
        video_id="captioned",
        youtube_url="https://www.youtube.com/watch?v=captioned",
        title="Captioned",
        channel="Channel",
        duration=60,
    )

    def fake_run(command, operation):  # type: ignore[no-untyped-def]
        captured.extend(command)
        assert operation == "transcript"
        template = command[command.index("-o") + 1]
        subtitle = Path(template.replace("%(ext)s", "en.vtt"))
        subtitle.write_text(
            "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nCaption text\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    gateway = YtDlpGateway(Settings())
    monkeypatch.setattr(gateway, "_run", fake_run)

    transcript = gateway.fetch_transcript(metadata, tmp_path)

    assert transcript.segments[0].text == "Caption text"
    assert "--ignore-no-formats-error" in captured
