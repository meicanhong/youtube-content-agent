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
