from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import ConfigurationError
from .models import ContentData, PackageMetadata, RunManifest, SourceData
from .storage import write_json

logger = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)


class StoryboardRenderer(Protocol):
    def render(
        self,
        media_path: Path,
        media_origin: float,
        content: ContentData,
        images_dir: Path,
        storyboard_path: Path,
    ) -> tuple[ContentData, list[Path]]: ...


class PackageRerenderer:
    """Regenerate visual files from saved package data without invoking an LLM."""

    def __init__(self, renderer: StoryboardRenderer, work_root: Path) -> None:
        self.renderer = renderer
        self.work_root = work_root

    def rerender(self, target: Path) -> list[Path]:
        packages = self._resolve_packages(target)
        outputs: list[Path] = []
        for package_dir in packages:
            outputs.extend(self._rerender_package(package_dir))
        return outputs

    def _rerender_package(self, package_dir: Path) -> list[Path]:
        source = self._load_model(package_dir / "source.json", SourceData)
        content = self._load_model(package_dir / "content.json", ContentData)
        metadata = self._load_model(package_dir / "metadata.json", PackageMetadata)
        media_path = self.work_root / source.video_id / f"{package_dir.name}.mp4"
        if not media_path.exists():
            raise ConfigurationError(
                f"缺少重渲染所需的视频缓存：{media_path}；请先运行一次完整生成流程"
            )

        temp_images = package_dir / ".rerender-images"
        temp_storyboard = package_dir / ".rerender-storyboard.jpg"
        self._cleanup_temp(temp_images, package_dir)
        try:
            rendered_content, temporary_storyboards = self.renderer.render(
                media_path,
                max(0.0, source.source_segment.start - 3.0),
                content,
                temp_images,
                temp_storyboard,
            )
            final_storyboards = self._install_rendered_files(
                package_dir,
                temp_images,
                temporary_storyboards,
            )
        except Exception:
            self._cleanup_temp(temp_images, package_dir)
            raise

        updated_metadata = metadata.model_copy(
            update={
                "generated_at": datetime.now(UTC),
                "image_count": len(rendered_content.slides),
                "storyboard_image": final_storyboards[0].name,
                "storyboard_images": [path.name for path in final_storyboards],
                "complete": all(slide.image for slide in rendered_content.slides)
                and all(path.exists() for path in final_storyboards),
            }
        )
        write_json(package_dir / "content.json", rendered_content)
        write_json(package_dir / "metadata.json", updated_metadata)
        logger.info(
            "content package rerendered without editorial provider",
            extra={
                "event": "package_rerender_complete",
                "operation": "visual_rerender",
                "resource_id": source.video_id,
                "storyboard_count": len(final_storyboards),
            },
        )
        return final_storyboards

    @staticmethod
    def _install_rendered_files(
        package_dir: Path,
        temp_images: Path,
        temporary_storyboards: list[Path],
    ) -> list[Path]:
        final_images = package_dir / "images"
        final_images.mkdir(parents=True, exist_ok=True)
        for image in temp_images.glob("*.jpg"):
            image.replace(final_images / image.name)
        temp_images.rmdir()

        if len(temporary_storyboards) == 1:
            final_storyboards = [package_dir / "storyboard.jpg"]
        else:
            final_storyboards = [
                package_dir / f"storyboard-{index:02d}.jpg"
                for index in range(1, len(temporary_storyboards) + 1)
            ]
        for old_storyboard in package_dir.glob("storyboard*.jpg"):
            old_storyboard.unlink()
        for temporary, final in zip(
            temporary_storyboards, final_storyboards, strict=True
        ):
            temporary.replace(final)
        return final_storyboards

    @staticmethod
    def _cleanup_temp(temp_images: Path, package_dir: Path) -> None:
        if temp_images.exists():
            for path in temp_images.iterdir():
                if path.is_file():
                    path.unlink()
            temp_images.rmdir()
        for path in package_dir.glob(".rerender-storyboard*.jpg"):
            path.unlink()

    @staticmethod
    def _resolve_packages(target: Path) -> list[Path]:
        if (target / "content.json").exists() and (target / "source.json").exists():
            return [target]
        manifest_path = target / "manifest.json"
        manifest = PackageRerenderer._load_model(manifest_path, RunManifest)
        packages = [target / name for name in manifest.packages]
        missing = [path for path in packages if not path.is_dir()]
        if missing:
            raise ConfigurationError(f"Manifest 中的 Package 不存在：{missing[0]}")
        return packages

    @staticmethod
    def _load_model(path: Path, model_type: type[ModelT]) -> ModelT:
        try:
            return model_type.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigurationError(f"缺少重渲染输入文件：{path}") from exc
        except ValidationError as exc:
            raise ConfigurationError(f"重渲染输入文件格式错误：{path}") from exc
