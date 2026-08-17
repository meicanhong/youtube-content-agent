from pathlib import Path
from typing import Protocol

from .models import EditorialResponse, Transcript, VideoMetadata


class YouTubeGateway(Protocol):
    def fetch_metadata(self, url: str) -> VideoMetadata: ...

    def fetch_transcript(self, metadata: VideoMetadata, work_dir: Path) -> Transcript: ...

    def download_segment(
        self, metadata: VideoMetadata, start: float, end: float, destination: Path
    ) -> Path: ...


class EditorialProvider(Protocol):
    @property
    def name(self) -> str: ...

    def create_topics(
        self, metadata: VideoMetadata, transcript: Transcript, max_topics: int
    ) -> EditorialResponse: ...
