from datetime import UTC, datetime
from pathlib import Path

from youtube_content_agent.models import (
    CaptionDraft,
    ContentData,
    GroundedSlide,
    PackageMetadata,
    SourceData,
    SourceSegment,
    TranscriptSegment,
)
from youtube_content_agent.rerender import PackageRerenderer
from youtube_content_agent.storage import write_json


class FakeRenderer:
    def render(
        self,
        media_path: Path,
        media_origin: float,
        content: ContentData,
        images_dir: Path,
        storyboard_path: Path,
    ) -> tuple[ContentData, list[Path]]:
        assert media_path.exists()
        assert media_origin == 97
        images_dir.mkdir(parents=True, exist_ok=True)
        slides = []
        for slide in content.slides:
            image = images_dir / f"{slide.index:02d}.jpg"
            image.write_bytes(b"new-image")
            slides.append(slide.model_copy(update={"image": f"images/{image.name}"}))
        storyboards = [
            storyboard_path.with_name(f"{storyboard_path.stem}-{index:02d}.jpg")
            for index in (1, 2)
        ]
        for storyboard in storyboards:
            storyboard.write_bytes(b"new-storyboard")
        return content.model_copy(update={"slides": slides}), storyboards


def test_rerender_replaces_visuals_from_saved_package_without_editorial(tmp_path: Path) -> None:
    package = tmp_path / "output" / "01-topic"
    package.mkdir(parents=True)
    source = SourceData(
        video_id="video-id",
        youtube_url="https://www.youtube.com/watch?v=video-id",
        title="Interview",
        channel="Channel",
        published_at=None,
        source_segment=SourceSegment(
            start=100,
            end=140,
            original_text="verified source",
            transcript_segments=[
                TranscriptSegment(start=100, end=105, text="verified source")
            ],
        ),
    )
    content = ContentData(
        topic="完整主题",
        hook="完整开头",
        slides=[
            GroundedSlide(
                index=index,
                timestamp=100 + index,
                original_text="verified source",
                zh_text=f"第{index}步",
                image=f"images/{index:02d}.jpg",
            )
            for index in range(1, 7)
        ],
        caption=CaptionDraft(
            title="完整标题",
            hook="完整开头",
            body="这是一段足够长的正文，用来验证不调用模型也可以安全重渲染全部视觉文件。",
        ),
        quality_score=0.9,
    )
    metadata = PackageMetadata(
        generated_at=datetime(2026, 8, 17, tzinfo=UTC),
        transcript_source="fixture",
        editorial_provider="openai:gpt-5.6-luna",
        image_count=6,
        storyboard_image="storyboard.jpg",
        storyboard_images=["storyboard.jpg"],
        complete=True,
    )
    write_json(package / "source.json", source)
    write_json(package / "content.json", content)
    write_json(package / "metadata.json", metadata)
    (package / "storyboard.jpg").write_bytes(b"old-storyboard")
    (package / "images").mkdir()
    (package / "images" / "01.jpg").write_bytes(b"old-image")
    work_root = tmp_path / "work"
    media = work_root / "video-id" / "01-topic.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"cached-video")

    outputs = PackageRerenderer(FakeRenderer(), work_root).rerender(package)

    assert [path.name for path in outputs] == ["storyboard-01.jpg", "storyboard-02.jpg"]
    assert not (package / "storyboard.jpg").exists()
    assert all(path.read_bytes() == b"new-storyboard" for path in outputs)
    assert (package / "images" / "01.jpg").read_bytes() == b"new-image"
    saved_metadata = PackageMetadata.model_validate_json(
        (package / "metadata.json").read_text(encoding="utf-8")
    )
    assert saved_metadata.editorial_provider == "openai:gpt-5.6-luna"
    assert saved_metadata.storyboard_images == ["storyboard-01.jpg", "storyboard-02.jpg"]
    assert saved_metadata.complete is True
