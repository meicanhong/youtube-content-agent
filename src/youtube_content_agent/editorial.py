from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError

from .errors import ConfigurationError, ExternalToolError
from .models import EditorialResponse, Transcript, VideoMetadata

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a source-first Chinese social-media editor.
Select independent, publishable ideas from an English interview. Every topic must use one
continuous source segment. Never invent a quote or idea. Slide timestamps must be ascending,
inside that segment, and correspond to the nearby source wording. Produce natural concise
Chinese, preserving meaning. Each carousel must tell one continuous story in 6-10 slides.
The caption may reorganize the source idea but must not introduce factual claims absent from it.
HARD TIMELINE CONTRACT FOR EVERY TOPIC:
- source_end - source_start MUST be between 30 and 180 seconds; never return a longer segment.
- Return 6-10 slides with strictly ascending timestamps.
- Every slide MUST include source_quote copied verbatim from the timestamped transcript. Never
  translate, paraphrase, or invent source_quote. The system will reject any quote it cannot find.
- timestamp MUST point to the first transcript segment containing source_quote, not an earlier
  setup sentence.
- Every slide timestamp MUST be greater than or equal to source_start and less than or equal to
  source_end.
- The segment needs context, argument, and conclusion. Quality beats quantity.
"""

JSON_ONLY_INSTRUCTION = """Return only one JSON object matching the supplied JSON Schema.
Do not use Markdown fences. Do not add commentary before or after the JSON.
"""

JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class OpenAIEditorialProvider:
    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise ConfigurationError(
                "生产 Editorial 需要 OPENAI_API_KEY；无 Key 时请显式传 --editorial-fixture。"
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def create_topics(
        self, metadata: VideoMetadata, transcript: Transcript, max_topics: int
    ) -> EditorialResponse:
        transcript_text = "\n".join(
            f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text}"
            for segment in transcript.segments
        )
        user_prompt = (
            f"Video: {metadata.title}\nChannel: {metadata.channel}\n"
            f"Return at most {max_topics} topics.\n\nTimestamped transcript:\n{transcript_text}"
        )
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=EditorialResponse,
            )
        except Exception as exc:
            raise ExternalToolError(f"OpenAI Editorial 调用失败：{type(exc).__name__}") from exc
        if response.output_parsed is None:
            raise ExternalToolError("OpenAI Editorial 未返回可解析的结构化结果")
        return response.output_parsed


class MimoEditorialProvider:
    """MiMo V2.5 adapter using its official OpenAI-compatible Chat Completions API."""

    def __init__(self, api_key: str | None, model: str, base_url: str) -> None:
        if not api_key:
            raise ConfigurationError(
                "MiMo Editorial 需要 MIMO_API_KEY；请写入本地 .env，或显式传 --editorial-fixture。"
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @property
    def name(self) -> str:
        return f"mimo:{self.model}"

    def create_topics(
        self, metadata: VideoMetadata, transcript: Transcript, max_topics: int
    ) -> EditorialResponse:
        transcript_text = _format_transcript(transcript)
        schema = json.dumps(EditorialResponse.model_json_schema(), ensure_ascii=False)
        user_prompt = (
            f"Video: {metadata.title}\nChannel: {metadata.channel}\n"
            f"Return at most {max_topics} topics.\n\nJSON Schema:\n{schema}\n\n"
            f"Timestamped transcript:\n{transcript_text}"
        )
        content = self._complete(user_prompt)
        try:
            draft = _parse_editorial_json(content, "MiMo")
        except ExternalToolError as initial_error:
            repair_prompt = _build_repair_prompt(content, transcript, schema)
            if repair_prompt is None:
                raise initial_error
            logger.warning(
                "MiMo editorial validation failed; attempting one bounded repair",
                extra={
                    "event": "external_retry",
                    "operation": "editorial_repair",
                    "provider": "mimo",
                    "resource_id": metadata.video_id,
                },
            )
            repaired_content = self._complete(repair_prompt)
            draft = _parse_editorial_json(repaired_content, "MiMo repair")
        return self._verify_source_bound_content(draft, transcript, schema)

    def _complete(self, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT + JSON_ONLY_INSTRUCTION},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=16_384,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as exc:
            raise ExternalToolError(f"MiMo Editorial 调用失败：{type(exc).__name__}") from exc
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise ExternalToolError("MiMo Editorial 返回了空内容")
        return content

    def _verify_source_bound_content(
        self, draft: EditorialResponse, transcript: Transcript, schema: str
    ) -> EditorialResponse:
        prompt = _build_verification_prompt(draft, transcript, schema)
        verified = _parse_editorial_json(self._complete(prompt), "MiMo verification")
        _assert_source_identity(draft, verified)
        return verified


def _format_transcript(transcript: Transcript) -> str:
    return "\n".join(
        f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text}" for segment in transcript.segments
    )


def _parse_editorial_json(content: str, provider: str) -> EditorialResponse:
    cleaned = _clean_json_content(content, provider)

    try:
        return EditorialResponse.model_validate_json(cleaned)
    except ValidationError as exc:
        timeline = _safe_timeline_diagnostic(cleaned)
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}:{error['type']}:{error['msg']}"
            for error in exc.errors(include_url=False, include_input=False)[:5]
        )
        raise ExternalToolError(
            f"{provider} Editorial JSON 未通过数据模型校验（{details or 'unknown'}；{timeline}）"
        ) from exc


def _clean_json_content(content: str, provider: str) -> str:
    cleaned = JSON_FENCE_RE.sub("", content.strip()).strip()
    if not cleaned.startswith("{") or not cleaned.endswith("}"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ExternalToolError(f"{provider} Editorial 未返回 JSON 对象")
        cleaned = cleaned[start : end + 1]
    return cleaned


def _build_repair_prompt(content: str, transcript: Transcript, schema: str) -> str | None:
    try:
        payload = json.loads(_clean_json_content(content, "MiMo"))
        topic = payload["topics"][0]
        start = float(topic["source_start"])
        end = float(topic["source_end"])
    except (ExternalToolError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if end <= start or end - start > 600:
        return None
    excerpt_segments = [
        segment
        for segment in transcript.segments
        if segment.end >= max(0, start - 5) and segment.start <= end + 5
    ]
    if not excerpt_segments:
        return None
    excerpt = "\n".join(
        f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text}" for segment in excerpt_segments
    )
    return (
        "The previous proposal violated the hard timeline contract. Rebuild exactly ONE topic "
        "from scratch using only the excerpt below. The new source segment MUST be a continuous "
        "30-180 second subsegment. Rewrite all 6-10 slides and the entire caption so every claim "
        "is supported inside that shorter subsegment. Do not merely change source_start or "
        "source_end around the old content.\n\n"
        f"JSON Schema:\n{schema}\n\nPrevious invalid proposal (intent reference only):\n"
        f"{content}\n\nAllowed timestamped transcript excerpt:\n{excerpt}"
    )


def _build_verification_prompt(
    draft: EditorialResponse, transcript: Transcript, schema: str
) -> str:
    excerpts: list[str] = []
    for index, topic in enumerate(draft.topics, start=1):
        segments = [
            segment
            for segment in transcript.segments
            if segment.end >= topic.source_start and segment.start <= topic.source_end
        ]
        excerpt = "\n".join(
            f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text}" for segment in segments
        )
        excerpts.append(f"TOPIC {index} SOURCE:\n{excerpt}")
    return (
        "Audit and rewrite the Chinese editorial fields so every factual statement is explicitly "
        "supported by the supplied source excerpts. Remove dates, roles, impact claims, context, "
        "or interpretations not stated in the excerpts, even if you know they are true. Preserve "
        "topic count, source_start, source_end, every slide timestamp, and every source_quote "
        "exactly. You may only rewrite topic, zh_text, caption title, hook, body, and "
        "quality_score. "
        "Return only the corrected JSON.\n\n"
        f"JSON Schema:\n{schema}\n\nDRAFT:\n{draft.model_dump_json(indent=2)}\n\n"
        f"SOURCE EXCERPTS:\n{'\n\n'.join(excerpts)}"
    )


def _assert_source_identity(draft: EditorialResponse, verified: EditorialResponse) -> None:
    if len(draft.topics) != len(verified.topics):
        raise ExternalToolError("MiMo verification 改变了 Topic 数量")
    for index, (before, after) in enumerate(zip(draft.topics, verified.topics, strict=True)):
        before_source = (before.source_start, before.source_end)
        after_source = (after.source_start, after.source_end)
        before_slides = [(slide.timestamp, slide.source_quote) for slide in before.slides]
        after_slides = [(slide.timestamp, slide.source_quote) for slide in after.slides]
        if before_source != after_source or before_slides != after_slides:
            raise ExternalToolError(f"MiMo verification 改变了 Topic {index + 1} 的来源身份")


def _safe_timeline_diagnostic(content: str) -> str:
    try:
        payload = json.loads(content)
        topic = payload["topics"][0]
        start = float(topic["source_start"])
        end = float(topic["source_end"])
        timestamps = [float(slide["timestamp"]) for slide in topic.get("slides", [])]
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return "timeline=unavailable"
    slide_range = f"{min(timestamps):.2f}-{max(timestamps):.2f}" if timestamps else "none"
    return f"source={start:.2f}-{end:.2f},duration={end - start:.2f},slides={slide_range}"


class FixtureEditorialProvider:
    """Explicit offline provider for tests and no-key demonstrations."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.response = EditorialResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            raise ConfigurationError(f"Editorial fixture 无法读取或结构不合法：{path}") from exc

    @property
    def name(self) -> str:
        return f"fixture:{self.path.name}"

    def create_topics(
        self, metadata: VideoMetadata, transcript: Transcript, max_topics: int
    ) -> EditorialResponse:
        del metadata, transcript
        return EditorialResponse(topics=self.response.topics[:max_topics])
