from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from youtube_content_agent.visual import SlideRenderer


def test_storyboard_caption_is_split_into_fixed_size_single_lines() -> None:
    renderer = SlideRenderer("ffmpeg")
    font = ImageFont.truetype(str(renderer.font_path), renderer.STORYBOARD_FONT_SIZE)
    text = "这是一段很长的字幕，需要按照标点和实际宽度拆开，每一个字幕条都只能显示一行。"

    parts = renderer._split_single_line(text, font, 420)

    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    assert len(parts) > 1
    assert "".join(parts) == text
    assert all(draw.textlength(part, font=font) <= 420 for part in parts)
    assert draw.textlength(parts[-1], font=font) >= 420 * 0.4


def test_storyboard_caption_removes_terminal_punctuation_only() -> None:
    assert SlideRenderer._strip_terminal_punctuation("保留句中的逗号，但移除结尾。") == (
        "保留句中的逗号，但移除结尾"
    )
    assert SlideRenderer._strip_terminal_punctuation("他说：“准备好了！”") == "他说：准备好了"
    assert SlideRenderer._strip_terminal_punctuation("都包含一个“低谷”") == "都包含一个低谷"
    assert SlideRenderer._strip_terminal_punctuation("保留“句中强调”，继续表达") == (
        "保留“句中强调”，继续表达"
    )


def test_storyboard_caption_never_starts_with_punctuation() -> None:
    assert SlideRenderer._strip_storyboard_punctuation("，但是事情还没有结束。") == (
        "但是事情还没有结束"
    )
    assert SlideRenderer._strip_storyboard_punctuation("“先把这一段讲完整”") == (
        "先把这一段讲完整"
    )


def test_storyboard_uses_fixed_three_by_four_strip_layout(tmp_path: Path) -> None:
    renderer = SlideRenderer("ffmpeg")
    frame_paths: list[Path] = []
    for index in range(9):
        frame_path = tmp_path / f"frame-{index}.jpg"
        Image.new("RGB", (640, 360), (100 + index * 8, 20, 30)).save(frame_path)
        frame_paths.append(frame_path)
    frames = [(path, f"这是第{index + 1}句参考字幕") for index, path in enumerate(frame_paths)]
    output = tmp_path / "storyboard.jpg"

    renderer._compose_storyboard(frames, output)

    with Image.open(output) as storyboard:
        assert storyboard.size == (1080, 1440)
        assert all(storyboard.getpixel((0, y)) != (255, 255, 255) for y in range(1440))
    expanded, font = renderer._layout_storyboard_frames(frames)
    assert len(expanded) == 9
    assert font.size == 46


def test_storyboard_continues_across_multiple_pages(tmp_path: Path) -> None:
    renderer = SlideRenderer("ffmpeg")
    frames: list[tuple[Path, str]] = []
    for index in range(21):
        frame_path = tmp_path / f"page-frame-{index}.jpg"
        Image.new("RGB", (640, 360), (40 + index * 4, 50, 60)).save(frame_path)
        frames.append((frame_path, f"这是连续内容的第{index + 1}步"))

    outputs = renderer._compose_storyboards(frames, tmp_path / "storyboard.jpg")

    assert [path.name for path in outputs] == [
        "storyboard-01.jpg",
        "storyboard-02.jpg",
        "storyboard-03.jpg",
    ]
    assert [len(page) for page in renderer._paginate_storyboard_frames(frames)] == [10, 9, 2]
    for output in outputs:
        with Image.open(output) as storyboard:
            assert storyboard.size == (1080, 1440)
