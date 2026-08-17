import json
from pathlib import Path

from youtube_content_agent.vtt import parse_json3, parse_vtt


def test_parse_json3(tmp_path: Path) -> None:
    path = tmp_path / "captions.en.json3"
    path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "tStartMs": 1000,
                        "dDurationMs": 2500,
                        "segs": [{"utf8": "Hello "}, {"utf8": "world"}],
                    },
                    {"tStartMs": 3500, "dDurationMs": 1000, "segs": [{"utf8": "\n"}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    segments = parse_json3(path)
    assert len(segments) == 1
    assert segments[0].start == 1
    assert segments[0].end == 3.5
    assert segments[0].text == "Hello world"


def test_parse_vtt_removes_tags_and_duplicate_cues(tmp_path: Path) -> None:
    path = tmp_path / "captions.vtt"
    path.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n<c>Hello world</c>\n\n"
        "00:00:03.000 --> 00:00:04.000\n<c>Hello world</c>\n",
        encoding="utf-8",
    )
    segments = parse_vtt(path)
    assert len(segments) == 1
    assert segments[0].text == "Hello world"
