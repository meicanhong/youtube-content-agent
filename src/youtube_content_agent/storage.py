from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import ContentData, PackageMetadata, SourceData


def slugify(value: str, fallback: str = "content") -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    readable = "".join(char if char.isalnum() else "-" for char in normalized)
    slug = re.sub(r"-+", "-", readable).strip("-")
    return slug[:64] or fallback


def write_json(path: Path, value: BaseModel | dict[str, Any] | list[Any]) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_package(
    directory: Path, source: SourceData, content: ContentData, metadata: PackageMetadata
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "source.json", source)
    write_json(directory / "content.json", content)
    write_json(directory / "metadata.json", metadata)
    caption = (
        f"# {content.caption.title}\n\n{content.caption.hook}\n\n{content.caption.body.strip()}\n"
    )
    (directory / "caption.md").write_text(caption, encoding="utf-8")
