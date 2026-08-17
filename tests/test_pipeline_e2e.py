from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from youtube_content_agent.grounding import GroundingService
from youtube_content_agent.models import (
    CaptionDraft,
    EditorialResponse,
    SlideProposal,
    TopicProposal,
    Transcript,
    TranscriptSegment,
    VideoMetadata,
)
from youtube_content_agent.pipeline import ContentPipeline
from youtube_content_agent.visual import SlideRenderer


class FakeYouTubeGateway:
    def fetch_metadata(self, url: str) -> VideoMetadata:
        return VideoMetadata(
            video_id="fixture-video",
            youtube_url=url,
            title="Fixture Interview",
            channel="Fixture Channel",
            published_at=datetime(2026, 8, 1, tzinfo=UTC),
            duration=200,
            view_count=1000,
        )

    def fetch_transcript(self, metadata: VideoMetadata, work_dir: Path) -> Transcript:
        del work_dir
        return Transcript(
            video_id=metadata.video_id,
            language="en",
            source="fixture",
            segments=[
                TranscriptSegment(
                    start=100 + index * 5,
                    end=105 + index * 5,
                    text=f"This is verified source sentence {index}.",
                )
                for index in range(14)
            ],
        )

    def download_segment(
        self, metadata: VideoMetadata, start: float, end: float, destination: Path
    ) -> Path:
        del metadata, start
        destination.parent.mkdir(parents=True, exist_ok=True)
        duration = end - 97
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=640x360:rate=1",
                "-t",
                str(duration),
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(destination),
            ],
            check=True,
        )
        return destination


class FakeEditorialProvider:
    @property
    def name(self) -> str:
        return "fixture:test"

    def create_topics(
        self, metadata: VideoMetadata, transcript: Transcript, max_topics: int
    ) -> EditorialResponse:
        del metadata, transcript, max_topics
        return EditorialResponse(
            topics=[
                TopicProposal(
                    topic="测试观点",
                    source_start=100,
                    source_end=160,
                    slides=[
                        SlideProposal(
                            timestamp=value,
                            source_quote=f"verified source sentence {index - 1}",
                            zh_text=f"连续叙事的第 {index} 步",
                        )
                        for index, value in enumerate((102, 107, 112, 117, 122, 127), start=1)
                    ],
                    caption=CaptionDraft(
                        title="真正值钱的是可追溯",
                        hook="为什么漂亮金句还不够？",
                        body="因为每一张图都必须能回到原视频、原字幕和准确时间点。只有这样，编辑才能快速检查，而不是重新看完整视频。",
                    ),
                    quality_score=0.91,
                )
            ]
        )


class MultiEditorialProvider:
    @property
    def name(self) -> str:
        return "fixture:multi"

    def create_topics(
        self, metadata: VideoMetadata, transcript: Transcript, max_topics: int
    ) -> EditorialResponse:
        del metadata, transcript
        topics = [
            TopicProposal(
                topic=f"测试观点{topic_index}",
                source_start=start,
                source_end=start + 30,
                slides=[
                    SlideProposal(
                        timestamp=start + 2 + slide_index * 5,
                        source_quote=f"verified source sentence {source_index + slide_index}",
                        zh_text=f"观点{topic_index}的第{slide_index + 1}步",
                    )
                    for slide_index in range(6)
                ],
                caption=CaptionDraft(
                    title=f"独立金句主题{topic_index}",
                    hook=f"这是第{topic_index}个独立观点",
                    body="这是一段彼此独立且能够完整回溯到原始字幕的内容，足以单独生成一张长图。",
                ),
                quality_score=0.9,
            )
            for topic_index, (start, source_index) in enumerate(((100, 0), (135, 7)), start=1)
        ]
        return EditorialResponse(topics=topics[:max_topics])


class FailingEditorialProvider:
    @property
    def name(self) -> str:
        return "fixture:must-not-run"

    def create_topics(
        self, metadata: VideoMetadata, transcript: Transcript, max_topics: int
    ) -> EditorialResponse:
        del metadata, transcript, max_topics
        raise AssertionError("max_topics=0 must not call the editorial provider")


def test_pipeline_writes_complete_package(tmp_path: Path) -> None:
    pipeline = ContentPipeline(
        youtube=FakeYouTubeGateway(),
        editorial=FakeEditorialProvider(),
        renderer=SlideRenderer("ffmpeg"),
        grounding=GroundingService(),
    )
    results = pipeline.run(
        "https://www.youtube.com/watch?v=fixture",
        tmp_path / "output",
        work_root=tmp_path / "work",
    )
    assert len(results) == 1
    package = results[0].directory
    assert (package / "source.json").exists()
    assert (package / "content.json").exists()
    assert (package / "caption.md").exists()
    assert (package / "metadata.json").exists()
    assert (package / "storyboard.jpg").exists()
    assert len(list((package / "images").glob("*.jpg"))) == 6
    assert results[0].metadata.complete is True
    assert results[0].metadata.storyboard_image == "storyboard.jpg"
    with Image.open(package / "storyboard.jpg") as storyboard:
        assert storyboard.size == (1080, 1440)
    assert all(
        slide.original_text.startswith("This is verified") for slide in results[0].content.slides
    )
    assert results[0].directory.name == "01-真正值钱的是可追溯"


def test_pipeline_writes_one_storyboard_per_distinct_topic(tmp_path: Path) -> None:
    pipeline = ContentPipeline(
        youtube=FakeYouTubeGateway(),
        editorial=MultiEditorialProvider(),
        renderer=SlideRenderer("ffmpeg"),
    )
    results = pipeline.run(
        "https://www.youtube.com/watch?v=fixture",
        tmp_path / "multi-output",
        max_topics=6,
        work_root=tmp_path / "multi-work",
    )
    assert len(results) == 2
    assert all((result.directory / "storyboard.jpg").exists() for result in results)
    assert [result.directory.name[:2] for result in results] == ["01", "02"]


def test_pipeline_allows_zero_output_without_editorial_call(tmp_path: Path) -> None:
    pipeline = ContentPipeline(
        youtube=FakeYouTubeGateway(),
        editorial=FailingEditorialProvider(),
        renderer=SlideRenderer("ffmpeg"),
    )
    results = pipeline.run(
        "https://www.youtube.com/watch?v=fixture",
        tmp_path / "zero-output",
        max_topics=0,
        work_root=tmp_path / "zero-work",
    )
    assert results == []
    assert (tmp_path / "zero-output" / "manifest.json").exists()
