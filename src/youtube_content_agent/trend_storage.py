from pathlib import Path

from .storage import write_json
from .trend_models import TrendReport


def write_trend_report(output_dir: Path, report: TrendReport) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "trend-report.json", report)
    lines = [
        f"# YouTube 播客与访谈 Top {len(report.ranked)}（{report.month}）",
        "",
        f"- 种子节目：{report.seed_count}",
        f"- 本月候选：{report.candidate_count}",
        f"- 规则通过：{report.eligible_count}",
        f"- AI：{report.ai_provider}",
        "",
        "| 排名 | 视频 | 频道 | 播放量 | 日均播放 | 综合分 | 入选理由 |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for item in report.ranked:
        video = item.video
        title = _table_text(video.title)
        channel = _table_text(video.channel)
        reason = _table_text(item.ai.reason)
        lines.append(
            f"| {item.rank} | [{title}]({video.youtube_url}) | {channel} | "
            f"{video.view_count:,} | {item.rule.views_per_day:,.0f} | "
            f"{item.final_score:.2f} | {reason} |"
        )
    (output_dir / "top10.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table_text(value: str) -> str:
    return " ".join(value.replace("|", "\\|").split())
