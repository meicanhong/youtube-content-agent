from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError

from .errors import ConfigurationError, ExternalToolError, GroundingError
from .grounding import GroundingService
from .models import EditorialResponse, StoryCoherenceAudit, Transcript, VideoMetadata

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a source-first Chinese social-media editor.
Select independent, publishable ideas from an English interview. Every topic must use one
continuous source segment. Never invent a quote or idea. Slide timestamps must be ascending,
inside that segment, and correspond to the nearby source wording. Produce natural concise
Chinese, preserving meaning. Each topic must tell one complete continuous story in 6-14 slides.
Use as many slides as needed to preserve the setup, reasoning, and conclusion. Never truncate a
story merely to fit one image; the renderer automatically continues after every 10 visual beats.
Most topics should use 8-12 slides. Use 13-14 only when the conclusion would otherwise be lost.
Each zh_text must be a concise single-line visual beat, ideally 8-24 Chinese characters. Split a
long idea across adjacent slides instead of writing a paragraph into one slide.
The caption may reorganize the source idea but must not introduce factual claims absent from it.
Return zero topics when the transcript contains no strong standalone idea. When returning multiple
topics, use substantially different source segments and editorial ideas. Never repackage the same
quote or argument under a different title. Quality and distinctness always beat filling the quota.
HARD TIMELINE CONTRACT FOR EVERY TOPIC:
- source_end - source_start MUST be between 30 and 180 seconds; never return a longer segment.
- Return 6-14 slides with strictly ascending timestamps.
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


class MimoEditorialProvider:
    """Source-first Editorial workflow over an OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str,
        *,
        provider_id: str = "mimo",
        display_name: str = "MiMo",
        api_key_name: str = "MIMO_API_KEY",
    ) -> None:
        if not api_key:
            raise ConfigurationError(
                f"{display_name} Editorial 需要 {api_key_name}；请写入本地环境文件，"
                "或显式传 --editorial-fixture。"
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.provider_id = provider_id
        self.display_name = display_name

    @property
    def name(self) -> str:
        return f"{self.provider_id}:{self.model}"

    def create_topics(
        self, metadata: VideoMetadata, transcript: Transcript, max_topics: int
    ) -> EditorialResponse:
        transcript_text = _format_transcript(transcript)
        schema = json.dumps(EditorialResponse.model_json_schema(), ensure_ascii=False)
        user_prompt = (
            f"Video: {metadata.title}\nChannel: {metadata.channel}\n"
            f"{_topic_count_instruction(max_topics)}\n\nJSON Schema:\n{schema}\n\n"
            f"Timestamped transcript:\n{transcript_text}"
        )
        content = self._complete(user_prompt)
        try:
            draft = _parse_editorial_json(content, self.display_name)
        except ExternalToolError as initial_error:
            repair_prompt = _build_repair_prompt(content, transcript, schema)
            if repair_prompt is None:
                raise initial_error
            logger.warning(
                f"{self.display_name} editorial validation failed; attempting one bounded repair",
                extra={
                    "event": "external_retry",
                    "operation": "editorial_repair",
                    "provider": self.provider_id,
                    "resource_id": metadata.video_id,
                },
            )
            repaired_content = self._complete(repair_prompt)
            draft = _parse_editorial_json(
                repaired_content, f"{self.display_name} repair"
            )
        if not draft.topics:
            return draft
        return self.revise_topics(draft, transcript, metadata.video_id, schema)

    def revise_topics(
        self,
        draft: EditorialResponse,
        transcript: Transcript,
        video_id: str,
        schema: str | None = None,
    ) -> EditorialResponse:
        """Ground, fact-check, and coherence-edit an existing editorial draft."""
        if not draft.topics:
            return draft
        resolved_schema = schema or json.dumps(
            EditorialResponse.model_json_schema(), ensure_ascii=False
        )
        grounded = self._repair_ungrounded_draft(
            draft, transcript, resolved_schema, video_id
        )
        source_verified = self._verify_source_bound_content(
            grounded, transcript, resolved_schema
        )
        return self._edit_story_coherence(
            source_verified, transcript, resolved_schema, video_id
        )

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
            raise ExternalToolError(
                f"{self.display_name} Editorial 调用失败：{type(exc).__name__}"
            ) from exc
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise ExternalToolError(f"{self.display_name} Editorial 返回了空内容")
        return content

    def _verify_source_bound_content(
        self, draft: EditorialResponse, transcript: Transcript, schema: str
    ) -> EditorialResponse:
        prompt = _build_verification_prompt(draft, transcript, schema)
        stage = f"{self.display_name} source verification"
        verified = _parse_editorial_json(self._complete(prompt), stage)
        _assert_source_identity(draft, verified, stage)
        return verified

    def _repair_ungrounded_draft(
        self,
        draft: EditorialResponse,
        transcript: Transcript,
        schema: str,
        video_id: str,
    ) -> EditorialResponse:
        try:
            _assert_topics_grounded(draft, transcript)
            return draft
        except GroundingError:
            logger.warning(
                f"{self.display_name} source quotes failed grounding; "
                "attempting one bounded repair",
                extra={
                    "event": "external_retry",
                    "operation": "editorial_grounding_repair",
                    "provider": self.provider_id,
                    "resource_id": video_id,
                },
            )
        prompt = _build_grounding_repair_prompt(draft, transcript, schema)
        stage = f"{self.display_name} grounding repair"
        repaired = _parse_editorial_json(self._complete(prompt), stage)
        if len(repaired.topics) != len(draft.topics):
            raise ExternalToolError(f"{stage} 改变了 Topic 数量")
        _assert_topics_grounded(repaired, transcript)
        return repaired

    def _edit_story_coherence(
        self,
        draft: EditorialResponse,
        transcript: Transcript,
        schema: str,
        video_id: str,
    ) -> EditorialResponse:
        current = draft
        audit = self._audit_story_coherence(current, transcript)
        for attempt in range(1, 3):
            code_risks = _detect_hard_coherence_risks(current)
            if _audit_passed(audit, len(current.topics)) and not code_risks:
                if current != draft:
                    logger.info(
                        f"{self.display_name} story coherence edit completed",
                        extra={
                            "event": "story_coherence_complete",
                            "operation": "editorial_story_edit",
                            "provider": self.provider_id,
                            "resource_id": video_id,
                            "topic_count": len(current.topics),
                            "retry_count": attempt - 1,
                        },
                    )
                return current
            prompt = _build_story_coherence_prompt(
                current, transcript, schema, audit, code_risks
            )
            stage = f"{self.display_name} story coherence"
            edited = _parse_editorial_json(self._complete(prompt), stage)
            _assert_topic_scope_identity(draft, edited, stage)
            _assert_topics_grounded(edited, transcript)
            current = self._verify_source_bound_content(edited, transcript, schema)
            audit = self._audit_story_coherence(current, transcript)
        if _audit_passed(audit, len(current.topics)) and not _detect_hard_coherence_risks(current):
            return current
        filtered = _filter_coherent_topics(current, audit)
        logger.warning(
            f"{self.display_name} dropped topics that failed final coherence review",
            extra={
                "event": "story_coherence_topics_dropped",
                "operation": "editorial_story_filter",
                "provider": self.provider_id,
                "resource_id": video_id,
                "topic_count": len(current.topics),
                "kept_topic_count": len(filtered.topics),
                "dropped_topic_count": len(current.topics) - len(filtered.topics),
            },
        )
        return filtered

    def _audit_story_coherence(
        self, draft: EditorialResponse, transcript: Transcript
    ) -> StoryCoherenceAudit:
        content = self._complete(_build_story_coherence_audit_prompt(draft, transcript))
        try:
            audit = StoryCoherenceAudit.model_validate_json(
                _clean_json_content(content, f"{self.display_name} story audit")
            )
        except ValidationError as exc:
            raise ExternalToolError(
                f"{self.display_name} story audit JSON 未通过数据模型校验"
            ) from exc
        if len(audit.topics) != len(draft.topics):
            raise ExternalToolError(
                f"{self.display_name} story audit 返回的 Topic 数量不匹配"
            )
        if [topic.topic_index for topic in audit.topics] != list(
            range(1, len(draft.topics) + 1)
        ):
            raise ExternalToolError(
                f"{self.display_name} story audit 返回的 Topic 顺序不匹配"
            )
        return audit


class OpenAIEditorialProvider(MimoEditorialProvider):
    """OpenAI-compatible provider using the same verified Editorial workflow."""

    def __init__(self, api_key: str | None, model: str, base_url: str) -> None:
        super().__init__(
            api_key,
            model,
            base_url,
            provider_id="openai",
            display_name="OpenAI-compatible",
            api_key_name="OPENAI_API_KEY",
        )


def _format_transcript(transcript: Transcript) -> str:
    return "\n".join(
        f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text}" for segment in transcript.segments
    )


def _topic_count_instruction(max_topics: int) -> str:
    if max_topics == 0:
        return "Return zero topics."
    return (
        f"Return between 0 and {max_topics} topics. Find as many strong, distinct ideas as the "
        "transcript genuinely supports, without padding. Each topic must use a substantially "
        "different source segment and argument."
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
        "30-180 second subsegment. Rewrite all 6-14 slides and the entire caption so every claim "
        "is supported inside that shorter subsegment. Do not merely change source_start or "
        "source_end around the old content.\n\n"
        f"JSON Schema:\n{schema}\n\nPrevious invalid proposal (intent reference only):\n"
        f"{content}\n\nAllowed timestamped transcript excerpt:\n{excerpt}"
    )


def _build_verification_prompt(
    draft: EditorialResponse, transcript: Transcript, schema: str
) -> str:
    excerpts = _topic_source_excerpts(draft, transcript)
    return (
        "Audit and rewrite the Chinese editorial fields so every factual statement is explicitly "
        "supported by the supplied source excerpts. Remove dates, roles, impact claims, context, "
        "or interpretations not stated in the excerpts, even if you know they are true. Preserve "
        "topic count, source_start, source_end, every slide timestamp, and every source_quote "
        "exactly. You may only rewrite topic, zh_text, caption title, hook, body, and "
        "quality_score. "
        "Return only the corrected JSON.\n\n"
        f"JSON Schema:\n{schema}\n\nDRAFT:\n{draft.model_dump_json(indent=2)}\n\n"
        f"SOURCE EXCERPTS:\n{excerpts}"
    )


def _build_story_coherence_prompt(
    draft: EditorialResponse,
    transcript: Transcript,
    schema: str,
    audit: StoryCoherenceAudit,
    code_risks: list[str],
) -> str:
    excerpts = _topic_source_excerpts(draft, transcript)
    return (
        "Act as a senior Chinese carousel story editor. The reader will see only the zh_text "
        "values in order, without the source transcript. Fix every supplied audit issue and "
        "code-detected risk. Rewrite the editable Chinese fields so every topic forms one "
        "self-contained story: setup, development, decision or contrast, and conclusion.\n\n"
        "Hard coherence requirements:\n"
        "- Every zh_text must connect naturally to the previous and next slide.\n"
        "- Never leave dangling structures such as 第一/第二, 因此, 另一方面, 这, or 它 when the "
        "required antecedent or counterpart is absent.\n"
        "- Restore necessary reasoning bridges using only facts stated in the supplied source "
        "excerpt; never add outside context.\n"
        "- Each zh_text should express one complete visual beat in concise natural Chinese, "
        "ideally 8-24 Chinese characters, with no terminal punctuation. Split a longer idea "
        "across adjacent slides.\n"
        "- Preserve the speaker's distinction between examples, causes, criteria, and "
        "conclusions.\n"
        "- Preserve topic count, source_start, and source_end exactly. You may replace slide "
        "timestamps and source_quote values with stronger bridge sentences from the same supplied "
        "source excerpt, and may return 6-14 slides. Every replacement quote must be verbatim and "
        "timestamped at its first containing transcript segment.\n"
        "- If the approved source excerpt cannot support a coherent standalone story, set that "
        "topic's quality_score below 0.72 instead of inventing a bridge.\n"
        "Return only the corrected JSON.\n\n"
        f"AUDIT:\n{audit.model_dump_json(indent=2)}\n\n"
        f"CODE-DETECTED RISKS:\n{json.dumps(code_risks, ensure_ascii=False)}\n\n"
        f"JSON Schema:\n{schema}\n\nDRAFT:\n{draft.model_dump_json(indent=2)}\n\n"
        f"SOURCE EXCERPTS:\n{excerpts}"
    )


def _build_story_coherence_audit_prompt(
    draft: EditorialResponse, transcript: Transcript
) -> str:
    schema = json.dumps(StoryCoherenceAudit.model_json_schema(), ensure_ascii=False)
    excerpts = _topic_source_excerpts(draft, transcript)
    risks = _detect_hard_coherence_risks(draft)
    return (
        "Act only as an independent Chinese story-continuity reviewer. The reader sees only each "
        "topic's zh_text values in order. Mark a topic incoherent if it has a missing premise, "
        "dangling enumeration or connector, ambiguous pronoun, abrupt example switch, omitted "
        "decision criterion, overloaded slide, or unnatural translation. A factually correct "
        "sequence still fails when it literally translates an English metaphor or phrasal verb "
        "into unnatural Chinese, changes causality, or stacks redundant contrast connectors. A "
        "factually correct sequence can still be incoherent. Give concrete slide-level repair "
        "instructions. Treat zh_text longer than 24 Chinese characters as a visual-density risk "
        "unless shortening it would remove essential meaning. Score "
        "0.85 or above only when the sequence reads as a self-contained story without the source. "
        "Return only JSON matching the schema.\n\n"
        f"CODE-DETECTED RISKS TO VERIFY:\n{json.dumps(risks, ensure_ascii=False)}\n\n"
        f"JSON Schema:\n{schema}\n\nDRAFT:\n{draft.model_dump_json(indent=2)}\n\n"
        f"SOURCE EXCERPTS:\n{excerpts}"
    )


def _detect_hard_coherence_risks(draft: EditorialResponse) -> list[str]:
    return [risk for topic_risks in _hard_risks_by_topic(draft) for risk in topic_risks]


def _hard_risks_by_topic(draft: EditorialResponse) -> list[list[str]]:
    risks_by_topic: list[list[str]] = []
    for topic_index, topic in enumerate(draft.topics, start=1):
        risks: list[str] = []
        texts = [slide.zh_text.strip() for slide in topic.slides]
        has_first = any(text.startswith(("第一", "首先")) for text in texts)
        for slide_index, text in enumerate(texts, start=1):
            if text.startswith("第二") and not has_first:
                risks.append(
                    f"Topic {topic_index} slide {slide_index} starts with 第二 but no earlier "
                    "slide establishes 第一"
                )
            if text.startswith("但") and "另一方面" in text:
                risks.append(
                    f"Topic {topic_index} slide {slide_index} stacks 但 and 另一方面 in the "
                    "same sentence"
                )
            if len(text) > 34:
                risks.append(
                    f"Topic {topic_index} slide {slide_index} is too dense for one visual line "
                    f"({len(text)} characters)"
                )
        risks_by_topic.append(risks)
    return risks_by_topic


def _audit_passed(audit: StoryCoherenceAudit, expected_topics: int) -> bool:
    return len(audit.topics) == expected_topics and all(
        topic.coherent and topic.score >= 0.85 for topic in audit.topics
    )


def _filter_coherent_topics(
    draft: EditorialResponse,
    audit: StoryCoherenceAudit,
) -> EditorialResponse:
    risks_by_topic = _hard_risks_by_topic(draft)
    topics = [
        topic
        for topic, topic_audit, risks in zip(
            draft.topics,
            audit.topics,
            risks_by_topic,
            strict=True,
        )
        if topic_audit.coherent and topic_audit.score >= 0.85 and not risks
    ]
    return EditorialResponse(topics=topics)


def _build_grounding_repair_prompt(
    draft: EditorialResponse, transcript: Transcript, schema: str
) -> str:
    excerpts = _topic_source_excerpts(draft, transcript, padding=12)
    return (
        "The draft contains at least one source_quote or timestamp that cannot be grounded in "
        "its selected source segment. Rebuild every affected topic using only its supplied "
        "timestamped source excerpt. Preserve the topic count and editorial intent, but you may "
        "change source_start, source_end, slide timestamps, source_quote, and Chinese fields. "
        "Every source_quote must be copied verbatim from the excerpt and its timestamp must point "
        "to the first segment containing that quote. Keep each source segment continuous and "
        "30-180 seconds long. Return only the corrected JSON.\n\n"
        f"JSON Schema:\n{schema}\n\nDRAFT:\n{draft.model_dump_json(indent=2)}\n\n"
        f"ALLOWED SOURCE EXCERPTS:\n{excerpts}"
    )


def _topic_source_excerpts(
    draft: EditorialResponse, transcript: Transcript, padding: float = 0
) -> str:
    excerpts: list[str] = []
    for index, topic in enumerate(draft.topics, start=1):
        segments = [
            segment
            for segment in transcript.segments
            if segment.end >= max(0, topic.source_start - padding)
            and segment.start <= topic.source_end + padding
        ]
        excerpt = "\n".join(
            f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text}" for segment in segments
        )
        excerpts.append(f"TOPIC {index} SOURCE:\n{excerpt}")
    return "\n\n".join(excerpts)


def _assert_topics_grounded(draft: EditorialResponse, transcript: Transcript) -> None:
    grounding = GroundingService()
    for topic in draft.topics:
        grounding.ground(topic, transcript)


def _assert_source_identity(
    draft: EditorialResponse, verified: EditorialResponse, stage: str
) -> None:
    if len(draft.topics) != len(verified.topics):
        raise ExternalToolError(f"{stage} 改变了 Topic 数量")
    for index, (before, after) in enumerate(zip(draft.topics, verified.topics, strict=True)):
        before_source = (before.source_start, before.source_end)
        after_source = (after.source_start, after.source_end)
        before_slides = [(slide.timestamp, slide.source_quote) for slide in before.slides]
        after_slides = [(slide.timestamp, slide.source_quote) for slide in after.slides]
        if before_source != after_source or before_slides != after_slides:
            raise ExternalToolError(f"{stage} 改变了 Topic {index + 1} 的来源身份")


def _assert_topic_scope_identity(
    draft: EditorialResponse, edited: EditorialResponse, stage: str
) -> None:
    if len(draft.topics) != len(edited.topics):
        raise ExternalToolError(f"{stage} 改变了 Topic 数量")
    for index, (before, after) in enumerate(zip(draft.topics, edited.topics, strict=True)):
        if (before.source_start, before.source_end) != (after.source_start, after.source_end):
            raise ExternalToolError(f"{stage} 改变了 Topic {index + 1} 的来源区间")


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
