from __future__ import annotations

import html
import json
import logging
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from pydantic import TypeAdapter, ValidationError

from .errors import ConfigurationError, ExternalToolError
from .trend_models import PodcastSeed, TrendVideo

logger = logging.getLogger(__name__)

JSON_LD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?"
    r"(?:(?P<seconds>\d+)S)?$"
)


class PodcastChartGateway:
    """Resolve the latest YouTube Podcast Top 100 snapshot into playlist seeds."""

    def __init__(
        self,
        chart_url: str = "https://podchartdb.com/",
        cache_path: Path | None = None,
        cache_ttl: timedelta = timedelta(hours=20),
    ) -> None:
        self.chart_url = chart_url
        self.cache_path = cache_path
        self.cache_ttl = cache_ttl

    @property
    def source_name(self) -> str:
        return "podchartdb:youtube-weekly-watch-time"

    def fetch_seeds(self, limit: int) -> list[PodcastSeed]:
        cached = self._read_cache(limit)
        if cached is not None:
            return cached
        chart_html = self._fetch_text(self.chart_url, "podcast_chart")
        entries = self._parse_chart_entries(chart_html)[:limit]
        seeds_by_rank: dict[int, PodcastSeed] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(entries))) as executor:
            futures = {executor.submit(self._resolve_seed, entry): entry for entry in entries}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    seed = future.result()
                except ExternalToolError:
                    logger.warning(
                        "podcast seed resolution failed",
                        extra={
                            "event": "external_partial_failure",
                            "operation": "resolve_podcast_seed",
                            "resource_id": str(entry[0]),
                            "provider": "podchartdb",
                        },
                    )
                    continue
                seeds_by_rank[seed.chart_rank] = seed
        seeds = [seeds_by_rank[rank] for rank in sorted(seeds_by_rank)]
        if not seeds:
            raise ExternalToolError("播客榜单未解析出任何 YouTube Playlist")
        self._write_cache(seeds)
        return seeds

    def _resolve_seed(self, entry: tuple[int, str, str]) -> PodcastSeed:
        rank, name, detail_url = entry
        detail_html = self._fetch_text(detail_url, "podcast_detail")
        series = self._find_json_ld_type(detail_html, "PodcastSeries")
        same_as = series.get("sameAs", [])
        playlist_url = next(
            (
                value
                for value in same_as
                if isinstance(value, str) and "youtube.com/playlist?list=" in value
            ),
            None,
        )
        publisher = series.get("publisher", {})
        publisher_name = publisher.get("name") if isinstance(publisher, dict) else None
        if not playlist_url:
            raise ExternalToolError(f"榜单第 {rank} 名缺少 YouTube Playlist")
        return PodcastSeed(
            chart_rank=rank,
            podcast_name=name,
            publisher=str(publisher_name or name),
            playlist_url=str(playlist_url),
        )

    def _read_cache(self, limit: int) -> list[PodcastSeed] | None:
        if self.cache_path is None or not self.cache_path.exists():
            return None
        age = datetime.now(UTC).timestamp() - self.cache_path.stat().st_mtime
        if age > self.cache_ttl.total_seconds():
            return None
        try:
            seeds = TypeAdapter(list[PodcastSeed]).validate_json(
                self.cache_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError):
            return None
        return seeds[:limit] if len(seeds) >= limit else None

    def _write_cache(self, seeds: list[PodcastSeed]) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [seed.model_dump(mode="json") for seed in seeds]
        self.cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _parse_chart_entries(chart_html: str) -> list[tuple[int, str, str]]:
        collection = PodcastChartGateway._find_json_ld_type(chart_html, "CollectionPage")
        items = collection.get("mainEntity", {}).get("itemListElement", [])
        entries: list[tuple[int, str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                entries.append((int(item["position"]), str(item["name"]), str(item["url"])))
            except (KeyError, TypeError, ValueError):
                continue
        if not entries:
            raise ExternalToolError("无法从播客榜单 JSON-LD 读取 Top 100")
        return sorted(entries, key=lambda entry: entry[0])

    @staticmethod
    def _find_json_ld_type(document: str, schema_type: str) -> dict[str, Any]:
        for raw in JSON_LD_RE.findall(document):
            try:
                payload = json.loads(html.unescape(raw.strip()))
            except json.JSONDecodeError:
                continue
            documents = payload if isinstance(payload, list) else [payload]
            for item in documents:
                if isinstance(item, dict) and item.get("@type") == schema_type:
                    return item
        raise ExternalToolError(f"页面缺少 {schema_type} JSON-LD")

    @staticmethod
    def _fetch_text(url: str, operation: str) -> str:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 TrendAgent/1.0"})
        for attempt in range(1, 3):
            try:
                with urlopen(request, timeout=20) as response:
                    return cast(bytes, response.read()).decode("utf-8")
            except (HTTPError, URLError, TimeoutError) as exc:
                if attempt == 2:
                    raise ExternalToolError(
                        f"{operation} 网络请求失败：{type(exc).__name__}"
                    ) from exc
                time.sleep(attempt)
        raise AssertionError("unreachable")


class YouTubeDataGateway:
    """Fetch public episode metadata from podcast playlists through YouTube Data API v3."""

    API_BASE = "https://www.googleapis.com/youtube/v3"

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise ConfigurationError(
                "Trend Agent 需要 YOUTUBE_API_KEY，用于读取 Playlist 和本月单集播放数据。"
            )
        self.api_key = api_key

    @property
    def source_name(self) -> str:
        return "youtube-data-api-v3"

    def fetch_month_videos(
        self,
        seeds: list[PodcastSeed],
        month_start: datetime,
        month_end: datetime,
        episodes_per_show: int,
    ) -> list[TrendVideo]:
        video_seeds: dict[str, PodcastSeed] = {}
        for seed in seeds:
            playlist_id = self._playlist_id(seed.playlist_url)
            payload = self._get_json(
                "playlistItems",
                {
                    "part": "contentDetails",
                    "playlistId": playlist_id,
                    "maxResults": min(50, episodes_per_show),
                },
            )
            for item in payload.get("items", []):
                details = item.get("contentDetails", {})
                video_id = details.get("videoId")
                published_at = self._parse_datetime(details.get("videoPublishedAt"))
                if (
                    video_id
                    and published_at is not None
                    and month_start <= published_at < month_end
                    and video_id not in video_seeds
                ):
                    video_seeds[video_id] = seed

        videos: list[TrendVideo] = []
        video_ids = list(video_seeds)
        for start in range(0, len(video_ids), 50):
            batch = video_ids[start : start + 50]
            payload = self._get_json(
                "videos",
                {
                    "part": "snippet,contentDetails,statistics,status,liveStreamingDetails",
                    "id": ",".join(batch),
                    "maxResults": 50,
                },
            )
            for item in payload.get("items", []):
                video = self._map_video(item, video_seeds, month_start, month_end)
                if video is not None:
                    videos.append(video)
        return sorted(videos, key=lambda video: (-video.view_count, video.seed_rank))

    def _map_video(
        self,
        item: dict[str, Any],
        video_seeds: dict[str, PodcastSeed],
        month_start: datetime,
        month_end: datetime,
    ) -> TrendVideo | None:
        video_id = str(item.get("id", ""))
        seed = video_seeds.get(video_id)
        snippet = item.get("snippet", {})
        content = item.get("contentDetails", {})
        status = item.get("status", {})
        published_at = self._parse_datetime(snippet.get("publishedAt"))
        if (
            seed is None
            or published_at is None
            or not month_start <= published_at < month_end
            or status.get("privacyStatus") != "public"
        ):
            return None
        try:
            duration_seconds = self._parse_duration(str(content.get("duration", "")))
        except ExternalToolError:
            return None
        if duration_seconds <= 0:
            return None
        return TrendVideo(
            video_id=video_id,
            youtube_url=f"https://www.youtube.com/watch?v={video_id}",
            title=str(snippet.get("title") or "Untitled"),
            description=str(snippet.get("description") or "")[:4000],
            channel=str(snippet.get("channelTitle") or seed.publisher),
            published_at=published_at,
            duration_seconds=duration_seconds,
            view_count=int(item.get("statistics", {}).get("viewCount", 0)),
            has_captions=str(content.get("caption", "false")).lower() == "true",
            is_live=bool(item.get("liveStreamingDetails")),
            seed_rank=seed.chart_rank,
            seed_name=seed.podcast_name,
        )

    def _get_json(self, resource: str, params: dict[str, str | int]) -> dict[str, Any]:
        query = urlencode({**params, "key": self.api_key})
        request = Request(f"{self.API_BASE}/{resource}?{query}")
        for attempt in range(1, 3):
            try:
                with urlopen(request, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ExternalToolError("YouTube Data API 返回了非对象响应")
                return payload
            except HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == 2:
                    raise ExternalToolError(f"YouTube Data API 请求失败：HTTP {exc.code}") from exc
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt == 2:
                    raise ExternalToolError(
                        f"YouTube Data API 请求失败：{type(exc).__name__}"
                    ) from exc
            time.sleep(attempt)
        raise AssertionError("unreachable")

    @staticmethod
    def _playlist_id(url: str) -> str:
        values = parse_qs(urlparse(url).query).get("list", [])
        if not values or not values[0]:
            raise ExternalToolError("播客种子缺少有效 Playlist ID")
        return values[0]

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            return None

    @staticmethod
    def _parse_duration(value: str) -> int:
        match = ISO_DURATION_RE.fullmatch(value)
        if match is None:
            raise ExternalToolError(f"无法解析 YouTube 视频时长：{value or 'empty'}")
        values = {name: int(raw or 0) for name, raw in match.groupdict().items()}
        return (
            values["days"] * 86400
            + values["hours"] * 3600
            + values["minutes"] * 60
            + values["seconds"]
        )


class FixturePodcastSeedGateway:
    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def source_name(self) -> str:
        return f"fixture:{self.path.name}"

    def fetch_seeds(self, limit: int) -> list[PodcastSeed]:
        try:
            seeds = TypeAdapter(list[PodcastSeed]).validate_json(
                self.path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise ConfigurationError(f"Trend seed fixture 无法读取：{self.path}") from exc
        return seeds[:limit]


class FixtureTrendVideoGateway:
    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def source_name(self) -> str:
        return f"fixture:{self.path.name}"

    def fetch_month_videos(
        self,
        seeds: list[PodcastSeed],
        month_start: datetime,
        month_end: datetime,
        episodes_per_show: int,
    ) -> list[TrendVideo]:
        del seeds, episodes_per_show
        try:
            videos = TypeAdapter(list[TrendVideo]).validate_json(
                self.path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise ConfigurationError(f"Trend video fixture 无法读取：{self.path}") from exc
        return [video for video in videos if month_start <= video.published_at < month_end]


class YtDlpTrendVideoGateway:
    """No-API-key fallback using authenticated yt-dlp metadata extraction."""

    def __init__(
        self,
        yt_dlp_bin: str,
        cookies_from_browser: str | None,
        cache_dir: Path,
        max_workers: int = 3,
    ) -> None:
        self.yt_dlp_bin = yt_dlp_bin
        self.cookies_from_browser = cookies_from_browser
        self.cache_dir = cache_dir
        self.max_workers = max_workers

    @property
    def source_name(self) -> str:
        auth = self.cookies_from_browser or "anonymous"
        return f"yt-dlp:{auth}"

    def fetch_month_videos(
        self,
        seeds: list[PodcastSeed],
        month_start: datetime,
        month_end: datetime,
        episodes_per_show: int,
    ) -> list[TrendVideo]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        collected: list[TrendVideo] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(seeds))) as executor:
            futures = {
                executor.submit(
                    self._fetch_seed,
                    seed,
                    month_start,
                    month_end,
                    episodes_per_show,
                ): seed
                for seed in seeds
            }
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    collected.extend(future.result())
                except ExternalToolError:
                    logger.warning(
                        "yt-dlp trend seed failed",
                        extra={
                            "event": "external_partial_failure",
                            "operation": "yt_dlp_trend_seed",
                            "resource_id": str(seed.chart_rank),
                            "provider": "yt-dlp",
                        },
                    )
        unique: dict[str, TrendVideo] = {}
        for video in sorted(collected, key=lambda item: item.seed_rank):
            unique.setdefault(video.video_id, video)
        if not unique:
            raise ExternalToolError(
                "yt-dlp 未读取到候选；请设置 --cookies-from-browser chrome 后重试"
            )
        return sorted(unique.values(), key=lambda video: (-video.view_count, video.seed_rank))

    def _fetch_seed(
        self,
        seed: PodcastSeed,
        month_start: datetime,
        month_end: datetime,
        episodes_per_show: int,
    ) -> list[TrendVideo]:
        cache_path = self.cache_dir / (
            f"{month_start.strftime('%Y-%m')}-{seed.chart_rank:03d}.json"
        )
        cached = self._read_seed_cache(cache_path)
        if cached is not None:
            return cached
        command = [
            self.yt_dlp_bin,
            "--ignore-no-formats-error",
            "--ignore-errors",
            "--skip-download",
            "--dump-json",
            "--no-warnings",
            "--playlist-end",
            str(episodes_per_show),
        ]
        if self.cookies_from_browser:
            command.extend(["--cookies-from-browser", self.cookies_from_browser])
        command.append(seed.playlist_url)
        try:
            result = subprocess.run(command, text=True, capture_output=True, check=False)
        except OSError as exc:
            raise ExternalToolError(f"无法启动 yt-dlp：{exc}") from exc
        videos: list[TrendVideo] = []
        for line in result.stdout.splitlines():
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            video = self._map_entry(raw, seed, month_start, month_end)
            if video is not None:
                videos.append(video)
        if result.returncode != 0 and not videos:
            detail = self._safe_error(result.stderr)
            raise ExternalToolError(f"yt-dlp 榜单元数据失败：{detail}")
        payload = [video.model_dump(mode="json") for video in videos]
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return videos

    @staticmethod
    def _map_entry(
        raw: dict[str, Any],
        seed: PodcastSeed,
        month_start: datetime,
        month_end: datetime,
    ) -> TrendVideo | None:
        video_id = raw.get("id")
        timestamp = raw.get("timestamp")
        duration = raw.get("duration")
        if not video_id or timestamp is None or duration is None:
            return None
        published_at = datetime.fromtimestamp(float(timestamp), UTC)
        if not month_start <= published_at < month_end or float(duration) <= 0:
            return None
        subtitles = raw.get("subtitles") or {}
        automatic = raw.get("automatic_captions") or {}
        caption_languages = {*subtitles.keys(), *automatic.keys()}
        has_english = any(str(language).lower().startswith("en") for language in caption_languages)
        live_status = str(raw.get("live_status") or "not_live")
        return TrendVideo(
            video_id=str(video_id),
            youtube_url=str(
                raw.get("webpage_url")
                or raw.get("url")
                or f"https://www.youtube.com/watch?v={video_id}"
            ),
            title=str(raw.get("title") or "Untitled"),
            description=str(raw.get("description") or "")[:4000],
            channel=str(raw.get("channel") or raw.get("uploader") or seed.publisher),
            published_at=published_at,
            duration_seconds=round(float(duration)),
            view_count=max(0, int(raw.get("view_count") or 0)),
            has_captions=has_english,
            is_live=live_status in {"is_live", "is_upcoming"},
            seed_rank=seed.chart_rank,
            seed_name=seed.podcast_name,
        )

    @staticmethod
    def _read_seed_cache(path: Path) -> list[TrendVideo] | None:
        if not path.exists():
            return None
        age = datetime.now(UTC).timestamp() - path.stat().st_mtime
        if age > timedelta(hours=6).total_seconds():
            return None
        try:
            return TypeAdapter(list[TrendVideo]).validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            return None

    @staticmethod
    def _safe_error(stderr: str) -> str:
        last_line = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
        return re.sub(r"https?://\S+", "<url>", last_line)[:300]
