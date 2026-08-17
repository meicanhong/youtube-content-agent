from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from .grounding import GroundingService
from .interfaces import EditorialProvider, YouTubeGateway
from .models import (
    PackageMetadata,
    PackageResult,
    RunManifest,
    SourceData,
)
from .storage import slugify, write_json, write_package
from .visual import SlideRenderer

logger = logging.getLogger(__name__)


class ContentPipeline:
    """Readable orchestration: fetch -> understand -> ground -> render -> persist."""

    def __init__(
        self,
        youtube: YouTubeGateway,
        editorial: EditorialProvider,
        renderer: SlideRenderer,
        grounding: GroundingService | None = None,
    ) -> None:
        self.youtube = youtube
        self.editorial = editorial
        self.renderer = renderer
        self.grounding = grounding or GroundingService()

    def run(
        self,
        url: str,
        output_dir: Path,
        max_topics: int = 3,
        work_root: Path = Path("work/youtube-content-agent"),
    ) -> list[PackageResult]:
        started_at = datetime.now(UTC)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = self.youtube.fetch_metadata(url)
        work_dir = work_root / metadata.video_id
        work_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "video metadata resolved",
            extra={"event": "metadata_ready", "resource_id": metadata.video_id},
        )
        transcript = self.youtube.fetch_transcript(metadata, work_dir)
        write_json(output_dir / "transcript.json", transcript)
        proposals = self.editorial.create_topics(metadata, transcript, max_topics)

        results: list[PackageResult] = []
        for index, proposal in enumerate(proposals.topics, start=1):
            source, content = self.grounding.ground(proposal, transcript)
            package_name = f"{index:02d}-{slugify(proposal.caption.title, f'topic-{index}')}"
            package_dir = output_dir / package_name
            media_path = work_dir / f"{package_name}.mp4"
            padding = 3.0
            media_origin = max(0.0, source.start - padding)
            self.youtube.download_segment(
                metadata,
                media_origin,
                min(metadata.duration, source.end + padding),
                media_path,
            )
            content = self.renderer.render(
                media_path,
                media_origin,
                content,
                package_dir / "images",
                package_dir / "storyboard.jpg",
            )
            source_data = SourceData(
                video_id=metadata.video_id,
                youtube_url=metadata.youtube_url,
                title=metadata.title,
                channel=metadata.channel,
                published_at=metadata.published_at,
                source_segment=source,
            )
            package_metadata = PackageMetadata(
                generated_at=datetime.now(UTC),
                transcript_source=transcript.source,
                editorial_provider=self.editorial.name,
                image_count=len(content.slides),
                storyboard_image="storyboard.jpg",
                complete=(
                    all(slide.image for slide in content.slides)
                    and (package_dir / "storyboard.jpg").exists()
                ),
            )
            write_package(package_dir, source_data, content, package_metadata)
            results.append(
                PackageResult(
                    directory=package_dir,
                    source=source_data,
                    content=content,
                    metadata=package_metadata,
                )
            )

        manifest = RunManifest(
            generated_at=started_at,
            video=metadata,
            transcript_file="transcript.json",
            packages=[result.directory.name for result in results],
            editorial_provider=self.editorial.name,
        )
        write_json(output_dir / "manifest.json", manifest)
        logger.info(
            "content pipeline completed",
            extra={
                "event": "pipeline_complete",
                "resource_id": metadata.video_id,
                "status": "success",
            },
        )
        return results
