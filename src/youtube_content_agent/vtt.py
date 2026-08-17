from __future__ import annotations

import html
import json
import re
from pathlib import Path

from .models import TranscriptSegment

TIMING_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{2}:\d{2}[.,]\d{3})\s+-->\s+"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{2}:\d{2}[.,]\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")


def parse_timestamp(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def clean_caption(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    return " ".join(text.replace("\u200b", "").split())


def parse_vtt(path: Path) -> list[TranscriptSegment]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    segments: list[TranscriptSegment] = []
    index = 0
    previous_text = ""
    while index < len(lines):
        match = TIMING_RE.search(lines[index])
        if not match:
            index += 1
            continue
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            if not TIMING_RE.search(lines[index]):
                text_lines.append(lines[index])
            index += 1
        text = clean_caption(" ".join(text_lines))
        if text and text != previous_text:
            segments.append(
                TranscriptSegment(
                    start=parse_timestamp(match.group("start")),
                    end=parse_timestamp(match.group("end")),
                    text=text,
                )
            )
            previous_text = text
    return segments


def parse_json3(path: Path) -> list[TranscriptSegment]:
    """Parse YouTube json3 captions without the rolling-text duplication found in VTT."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments: list[TranscriptSegment] = []
    for event in payload.get("events", []):
        raw_parts = event.get("segs") or []
        text = clean_caption("".join(str(part.get("utf8", "")) for part in raw_parts))
        if not text or text == "\n":
            continue
        start = float(event.get("tStartMs", 0)) / 1000
        duration = float(event.get("dDurationMs", 0)) / 1000
        if duration <= 0:
            continue
        segments.append(TranscriptSegment(start=start, end=start + duration, text=text))
    return segments
