from __future__ import annotations

import unicodedata

from .models import EditorialResponse, TopicProposal


class TopicSelectionPolicy:
    """Keep only strong, distinct topics while preserving editorial rank order."""

    def __init__(self, min_quality_score: float = 0.72, max_overlap_ratio: float = 0.35) -> None:
        self.min_quality_score = min_quality_score
        self.max_overlap_ratio = max_overlap_ratio

    def select(self, response: EditorialResponse, max_topics: int) -> EditorialResponse:
        if not 0 <= max_topics <= 6:
            raise ValueError("max_topics must be between 0 and 6")
        if max_topics == 0:
            return EditorialResponse(topics=[])

        selected: list[TopicProposal] = []
        topic_names: set[str] = set()
        titles: set[str] = set()
        for candidate in response.topics:
            topic_name = _normalize(candidate.topic)
            title = _normalize(candidate.caption.title)
            is_duplicate = topic_name in topic_names or title in titles
            overlaps = any(
                _overlap_ratio(candidate, accepted) > self.max_overlap_ratio
                for accepted in selected
            )
            if candidate.quality_score < self.min_quality_score or is_duplicate or overlaps:
                continue

            selected.append(candidate)
            topic_names.add(topic_name)
            titles.add(title)
            if len(selected) == max_topics:
                break

        return EditorialResponse(topics=selected)


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _overlap_ratio(left: TopicProposal, right: TopicProposal) -> float:
    intersection = max(
        0.0,
        min(left.source_end, right.source_end) - max(left.source_start, right.source_start),
    )
    shorter_duration = min(
        left.source_end - left.source_start,
        right.source_end - right.source_start,
    )
    return intersection / shorter_duration
