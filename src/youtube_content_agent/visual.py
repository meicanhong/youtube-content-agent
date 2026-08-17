from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from .errors import ExternalToolError
from .models import ContentData

logger = logging.getLogger(__name__)


class SlideRenderer:
    WIDTH = 1080
    HEIGHT = 1350
    STORYBOARD_WIDTH = 1080
    STORYBOARD_HEIGHT = 1440
    STORYBOARD_HERO_HEIGHT = 480
    STORYBOARD_FONT_SIZE = 46
    STORYBOARD_MIN_FONT_SIZE = 30
    STORYBOARD_TEXT_WIDTH = 1000
    STORYBOARD_PANEL_VERTICAL_PADDING = 18
    STORYBOARD_MAX_FRAMES_PER_PAGE = 10
    STORYBOARD_TERMINAL_PUNCTUATION = "，。；：！？、,.;:!?…—-·“”‘’\"'（）()【】[]《》〈〉"
    STORYBOARD_PUNCTUATION_PAIRS = {
        "”": "“",
        "’": "‘",
        '"': '"',
        "'": "'",
        "）": "（",
        ")": "(",
        "】": "【",
        "]": "[",
        "》": "《",
        "〉": "〈",
    }
    CANDIDATE_OFFSETS = (-0.35, 0.0, 0.35)

    def __init__(self, ffmpeg_bin: str) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.font_path = self._find_font()

    def render(
        self,
        media_path: Path,
        media_origin: float,
        content: ContentData,
        images_dir: Path,
        storyboard_path: Path,
    ) -> tuple[ContentData, list[Path]]:
        images_dir.mkdir(parents=True, exist_ok=True)
        updated_slides = []
        storyboard_frames: list[tuple[Path, str]] = []
        try:
            for slide in content.slides:
                image_path = images_dir / f"{slide.index:02d}.jpg"
                local_timestamp = max(0, slide.timestamp - media_origin)
                candidate = self._best_frame(media_path, local_timestamp, images_dir)
                frame = images_dir / f".frame-{slide.index:02d}.jpg"
                candidate.replace(frame)
                display_text = self._strip_storyboard_punctuation(slide.zh_text)
                if not display_text:
                    raise ExternalToolError(f"第 {slide.index} 个分镜清除标点后没有可显示文字")
                self._compose(frame, display_text, image_path)
                storyboard_frames.append((frame, display_text))
                updated_slides.append(
                    slide.model_copy(update={"image": f"images/{image_path.name}"})
                )
            storyboard_paths = self._compose_storyboards(storyboard_frames, storyboard_path)
        finally:
            for frame, _ in storyboard_frames:
                frame.unlink(missing_ok=True)
        return content.model_copy(update={"slides": updated_slides}), storyboard_paths

    def _compose_storyboards(
        self,
        frames: list[tuple[Path, str]],
        output_path: Path,
    ) -> list[Path]:
        font = ImageFont.truetype(str(self.font_path), self.STORYBOARD_FONT_SIZE)
        expanded = self._split_storyboard_captions(frames, font)
        pages = self._paginate_storyboard_frames(expanded)
        if len(pages) == 1:
            output_paths = [output_path]
        else:
            output_paths = [
                output_path.with_name(f"{output_path.stem}-{index:02d}{output_path.suffix}")
                for index in range(1, len(pages) + 1)
            ]
        for page, page_path in zip(pages, output_paths, strict=True):
            self._compose_storyboard_page(page, font, page_path)
        return output_paths

    @classmethod
    def _paginate_storyboard_frames(
        cls,
        frames: list[tuple[Path, str]],
    ) -> list[list[tuple[Path, str]]]:
        if len(frames) < 2:
            raise ExternalToolError("连续故事长图至少需要 2 个分镜")
        pages = [
            frames[index : index + cls.STORYBOARD_MAX_FRAMES_PER_PAGE]
            for index in range(0, len(frames), cls.STORYBOARD_MAX_FRAMES_PER_PAGE)
        ]
        if len(pages) > 1 and len(pages[-1]) == 1:
            pages[-1].insert(0, pages[-2].pop())
        return pages

    def _best_frame(self, media_path: Path, timestamp: float, work_dir: Path) -> Path:
        candidates: list[tuple[float, Path]] = []
        for index, offset in enumerate(self.CANDIDATE_OFFSETS):
            candidate = work_dir / f".candidate-{index}.jpg"
            command = [
                self.ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{max(0, timestamp + offset):.3f}",
                "-i",
                str(media_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                "-y",
                str(candidate),
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            if result.returncode == 0 and candidate.exists():
                candidates.append((self._frame_score(candidate), candidate))
        if not candidates:
            raise ExternalToolError(f"FFmpeg 无法在 {timestamp:.2f}s 附近截帧")
        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0][1]
        for _, candidate in candidates[1:]:
            candidate.unlink(missing_ok=True)
        return best

    @staticmethod
    def _frame_score(path: Path) -> float:
        with Image.open(path) as image:
            gray = image.convert("L").resize((256, 144))
            mean = ImageStat.Stat(gray).mean[0]
            contrast = ImageStat.Stat(gray).stddev[0]
            sharpness = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0]
            exposure_penalty = abs(mean - 120) * 0.25
            return contrast + sharpness * 1.5 - exposure_penalty

    def _compose(self, frame_path: Path, text: str, output_path: Path) -> None:
        with Image.open(frame_path).convert("RGB") as frame:
            canvas = self._cover(frame, self.WIDTH, self.HEIGHT)
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        box_top = 900
        draw.rounded_rectangle((55, box_top, 1025, 1285), radius=28, fill=(0, 0, 0, 175))
        font, lines = self._fit_text(text, 900)
        line_height = font.size + 22
        total_height = len(lines) * line_height
        y = box_top + (385 - total_height) / 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            x = (self.WIDTH - (bbox[2] - bbox[0])) / 2
            draw.text((x, y), line, font=font, fill="white", stroke_width=2, stroke_fill=(0, 0, 0))
            y += line_height
        composed = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        composed.save(output_path, quality=92, optimize=True)

    def _compose_storyboard(self, frames: list[tuple[Path, str]], output_path: Path) -> None:
        if len(frames) < 2:
            raise ExternalToolError("连续故事长图至少需要 2 个分镜")
        frames, font = self._layout_storyboard_frames(frames)
        self._compose_storyboard_page(frames, font, output_path)

    def _compose_storyboard_page(
        self,
        frames: list[tuple[Path, str]],
        font: ImageFont.FreeTypeFont,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        compact_total = self.STORYBOARD_HEIGHT - self.STORYBOARD_HERO_HEIGHT
        compact_height, remainder = divmod(compact_total, len(frames) - 1)
        heights = [self.STORYBOARD_HERO_HEIGHT] + [compact_height] * (len(frames) - 1)
        for index in range(1, remainder + 1):
            heights[index] += 1

        canvas = Image.new("RGB", (self.STORYBOARD_WIDTH, self.STORYBOARD_HEIGHT), "black")
        y = 0
        for index, ((frame_path, text), panel_height) in enumerate(
            zip(frames, heights, strict=True)
        ):
            with Image.open(frame_path).convert("RGB") as frame:
                panel = self._cover(frame, self.STORYBOARD_WIDTH, panel_height)
            panel = self._caption_storyboard_panel(panel, text, font, hero=index == 0)
            canvas.paste(panel, (0, y))
            y += panel_height
        canvas.save(output_path, quality=94, optimize=True)

    def _layout_storyboard_frames(
        self, frames: list[tuple[Path, str]]
    ) -> tuple[list[tuple[Path, str]], ImageFont.FreeTypeFont]:
        for size in range(
            self.STORYBOARD_FONT_SIZE,
            self.STORYBOARD_MIN_FONT_SIZE - 1,
            -2,
        ):
            font = ImageFont.truetype(str(self.font_path), size)
            expanded = self._split_storyboard_captions(frames, font)
            compact_height = (self.STORYBOARD_HEIGHT - self.STORYBOARD_HERO_HEIGHT) // (
                len(expanded) - 1
            )
            if compact_height >= size + self.STORYBOARD_PANEL_VERTICAL_PADDING:
                return expanded, font
        raise ExternalToolError("故事长图字幕过多，固定 3:4 画布无法保持统一可读字号")

    def _split_storyboard_captions(
        self,
        frames: list[tuple[Path, str]],
        font: ImageFont.FreeTypeFont,
    ) -> list[tuple[Path, str]]:
        expanded: list[tuple[Path, str]] = []
        for frame_path, text in frames:
            for part in self._split_single_line(text, font, self.STORYBOARD_TEXT_WIDTH):
                display_text = self._strip_storyboard_punctuation(part)
                if display_text:
                    expanded.append((frame_path, display_text))
        return expanded

    @classmethod
    def _strip_storyboard_punctuation(cls, text: str) -> str:
        cleaned = cls._strip_terminal_punctuation(text).lstrip()
        return cleaned.lstrip(cls.STORYBOARD_TERMINAL_PUNCTUATION).lstrip()

    @classmethod
    def _strip_terminal_punctuation(cls, text: str) -> str:
        original = text.rstrip()
        cleaned = original.rstrip(cls.STORYBOARD_TERMINAL_PUNCTUATION).rstrip()
        removed_suffix = original[len(cleaned) :]
        for closing, opening in cls.STORYBOARD_PUNCTUATION_PAIRS.items():
            if closing not in removed_suffix:
                continue
            if opening == closing:
                if cleaned.count(opening) % 2 == 0:
                    continue
            elif cleaned.count(opening) <= cleaned.count(closing):
                continue
            index = cleaned.rfind(opening)
            if index >= 0:
                cleaned = cleaned[:index] + cleaned[index + len(opening) :]
        return cleaned.rstrip()

    @staticmethod
    def _split_single_line(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        punctuation = "，。；！？、,.;!?"
        remaining = text.strip()
        parts: list[str] = []
        while remaining:
            if draw.textlength(remaining, font=font) <= max_width:
                parts.append(remaining)
                break
            fit_end = 1
            for index in range(1, len(remaining) + 1):
                if draw.textlength(remaining[:index], font=font) > max_width:
                    break
                fit_end = index
            preferred_end = max(
                (
                    index
                    for index, char in enumerate(remaining[:fit_end], start=1)
                    if char in punctuation and index >= fit_end * 0.55
                ),
                default=fit_end,
            )
            parts.append(remaining[:preferred_end].strip())
            remaining = remaining[preferred_end:].strip()
        return SlideRenderer._rebalance_short_tail(parts, font, max_width)

    @staticmethod
    def _rebalance_short_tail(
        parts: list[str], font: ImageFont.FreeTypeFont, max_width: int
    ) -> list[str]:
        if len(parts) < 2:
            return parts
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        if draw.textlength(parts[-1], font=font) >= max_width * 0.4:
            return parts
        combined = parts[-2] + parts[-1]
        punctuation = "，。；！？、,.;!?"
        candidates: list[tuple[float, str, str]] = []
        for index in range(1, len(combined)):
            left = combined[:index].strip()
            right = combined[index:].strip()
            left_width = draw.textlength(left, font=font)
            right_width = draw.textlength(right, font=font)
            if left_width > max_width or right_width > max_width:
                continue
            punctuation_bonus = 0.7 if combined[index - 1] in punctuation else 1.0
            candidates.append((abs(left_width - right_width) * punctuation_bonus, left, right))
        if not candidates:
            return parts
        _, left, right = min(candidates, key=lambda candidate: candidate[0])
        return [*parts[:-2], left, right]

    def _caption_storyboard_panel(
        self,
        panel: Image.Image,
        text: str,
        font: ImageFont.FreeTypeFont,
        *,
        hero: bool,
    ) -> Image.Image:
        overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if hero:
            caption_height = int(font.size) + 50
            box = (0, panel.height - caption_height, panel.width, panel.height)
            draw.rectangle(box, fill=(0, 0, 0, 165))
        else:
            box = (0, 0, panel.width, panel.height)
            draw.rectangle(box, fill=(0, 0, 0, 112))
        if draw.textlength(text, font=font) > self.STORYBOARD_TEXT_WIDTH:
            raise ExternalToolError("故事长图字幕未正确拆分为单行")
        self._draw_centered_lines(draw, [text], font, box)
        return Image.alpha_composite(panel.convert("RGBA"), overlay).convert("RGB")

    @staticmethod
    def _draw_centered_lines(
        draw: ImageDraw.ImageDraw,
        lines: list[str],
        font: ImageFont.FreeTypeFont,
        box: tuple[int, int, int, int],
    ) -> None:
        line_height = font.size + 14
        total_height = len(lines) * line_height
        y = box[1] + (box[3] - box[1] - total_height) / 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=2)
            x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) / 2
            draw.text(
                (x, y),
                line,
                font=font,
                fill="white",
                stroke_width=3,
                stroke_fill=(0, 0, 0),
            )
            y += line_height

    @staticmethod
    def _cover(image: Image.Image, width: int, height: int) -> Image.Image:
        scale = max(width / image.width, height / image.height)
        resized = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
        left = (resized.width - width) // 2
        top = (resized.height - height) // 2
        return resized.crop((left, top, left + width, top + height))

    @staticmethod
    def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        lines: list[str] = []
        current = ""
        for char in text:
            candidate = current + char
            if current and draw.textlength(candidate, font=font) > max_width:
                lines.append(current)
                current = char
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    def _fit_text(
        self,
        text: str,
        max_width: int,
        *,
        max_size: int = 66,
        min_size: int = 42,
        max_lines: int = 3,
    ) -> tuple[ImageFont.FreeTypeFont, list[str]]:
        fallback: tuple[ImageFont.FreeTypeFont, list[str]] | None = None
        for size in range(max_size, min_size - 1, -2):
            font = ImageFont.truetype(str(self.font_path), size)
            lines = self._wrap_text(text, font, max_width)
            fallback = (font, lines)
            last_line_is_orphan = len(lines) > 1 and len(lines[-1].strip()) == 1
            if len(lines) <= max_lines and not last_line_is_orphan:
                return font, lines
        if fallback is None:
            raise ExternalToolError("无法加载中文字幕字体")
        return fallback

    @staticmethod
    def _find_font() -> Path:
        candidates = [
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/System/Library/Fonts/STHeiti Medium.ttc"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise ExternalToolError("找不到支持中文的系统字体")
