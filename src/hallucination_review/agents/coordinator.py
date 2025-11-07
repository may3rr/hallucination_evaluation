from __future__ import annotations

"""Coordinator agent orchestrating the multi-agent workflow."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from autogen_core import MessageContext, message_handler
from rich.console import Console

from ..config import get_settings
from ..models.messages import (
    ClaimAssessment,
    DraftTask,
    FinalReview,
    OutlineResult,
    ReviewReport,
    ReviewRequest,
    RevisionRequest,
    SearchResult,
    SearchTask,
    SectionDraft,
    SectionPlan,
    TopicRequest,
)
from ..models.papers import PaperMetadata
from .base import LiteratureAgent

console = Console()


@dataclass
class RequestState:
    topic: str
    instructions: str | None
    outline: List[SectionPlan]
    search_results: Dict[str, List[PaperMetadata]] = field(default_factory=dict)
    drafts: Dict[str, SectionDraft] = field(default_factory=dict)
    approvals: Dict[str, bool] = field(default_factory=dict)
    revisions: Dict[str, int] = field(default_factory=dict)
    review_stats: Dict[str, int] = field(default_factory=lambda: {
        "supported": 0,
        "unsupported": 0,
        "metadata_error": 0,
    })


class CoordinatorAgent(LiteratureAgent):
    """Coordinates agents and produces the final literature review."""

    def __init__(self, *, llm, paper_repo) -> None:  # type: ignore[override]
        super().__init__("Workflow coordinator", llm=llm, paper_repo=paper_repo)
        self._requests: Dict[str, RequestState] = {}
        self._settings = get_settings()

    @message_handler
    async def handle_topic(self, message: TopicRequest, ctx: MessageContext) -> None:
        console.rule(f"开始文献综述：{message.topic}")
        self._requests[message.request_id] = RequestState(
            topic=message.topic,
            instructions=message.instructions,
            outline=[],
        )

    @message_handler
    async def handle_outline(self, message: OutlineResult, ctx: MessageContext) -> None:
        state = self._requests[message.request_id]
        state.outline = message.sections
        console.print(f"[bold green]大纲完成[/]：{', '.join(section.title for section in message.sections)}")
        for section in message.sections:
            await self.publish_message(
                SearchTask(
                    request_id=message.request_id,
                    topic=state.topic,
                    section=section,
                ),
                topic_id=ctx.topic_id,
            )

    @message_handler
    async def handle_search(self, message: SearchResult, ctx: MessageContext) -> None:
        state = self._requests[message.request_id]
        state.search_results[message.section_id] = message.papers
        console.print(f"[cyan]检索完成[/]：{message.section_id} 共 {len(message.papers)} 篇")
        if len(state.search_results) == len(state.outline):
            await self._dispatch_drafting(message.request_id, ctx)

    async def _dispatch_drafting(self, request_id: str, ctx: MessageContext) -> None:
        state = self._requests[request_id]
        await self.publish_message(
            DraftTask(
                request_id=request_id,
                topic=state.topic,
                sections=state.outline,
                section_papers=state.search_results,
            ),
            topic_id=ctx.topic_id,
        )

    @message_handler
    async def handle_draft(self, message: SectionDraft, ctx: MessageContext) -> None:
        state = self._requests[message.request_id]
        state.drafts[message.section_id] = message
        console.print(f"[magenta]草稿生成[/]：{message.section_id}")
        await self.publish_message(
            ReviewRequest(
                request_id=message.request_id,
                section_id=message.section_id,
                content=message.content,
            ),
            topic_id=ctx.topic_id,
        )

    @message_handler
    async def handle_review(self, message: ReviewReport, ctx: MessageContext) -> None:
        state = self._requests[message.request_id]
        for assessment in message.assessments:
            state.review_stats[assessment.verdict] = state.review_stats.get(assessment.verdict, 0) + 1
        if message.approved:
            state.approvals[message.section_id] = True
            console.print(f"[bold green]审查通过[/]：{message.section_id}")
        else:
            count = state.revisions.get(message.section_id, 0)
            if count >= self._settings.revision_limit:
                console.print(f"[red]达到修订上限，保留最新版本[/]：{message.section_id}")
                state.approvals[message.section_id] = True
            else:
                state.revisions[message.section_id] = count + 1
                feedback = _format_feedback(message.assessments)
                await self.publish_message(
                    RevisionRequest(
                        request_id=message.request_id,
                        section_id=message.section_id,
                        original=state.drafts[message.section_id].content,
                        feedback=feedback,
                    ),
                    topic_id=ctx.topic_id,
                )
                return
        if len(state.approvals) == len(state.outline):
            await self._finalize(message.request_id, ctx)

    async def _finalize(self, request_id: str, ctx: MessageContext) -> None:
        state = self._requests[request_id]
        references = _collect_references(list(state.drafts.values()), self.papers)
        stats = {
            "supported": state.review_stats.get("supported", 0),
            "unsupported": state.review_stats.get("unsupported", 0),
            "metadata_error": state.review_stats.get("metadata_error", 0),
        }
        final = FinalReview(
            request_id=request_id,
            topic=state.topic,
            outline=state.outline,
            sections=list(state.drafts.values()),
            references=references,
            stats=stats,
        )
        console.rule("输出综述结果")
        console.print(_format_output(final))
        _write_output(final)
        await self.publish_message(final, topic_id=ctx.topic_id)


def _format_feedback(assessments: List[ClaimAssessment]) -> str:
    lines = []
    for item in assessments:
        if item.verdict == "supported":
            continue
        lines.append(f"- {item.claim} -> {item.verdict}: {item.rationale}")
    return "\n".join(lines) or "段落中所有引用均已通过审查。"


def _collect_references(drafts: List[SectionDraft], repo) -> List[str]:
    seen = []
    order = []
    for draft in drafts:
        for match in re.finditer(r"\[(?:arXiv:)?([0-9.]+)\]", draft.content):
            arxiv_id = match.group(1)
            if arxiv_id not in seen:
                seen.append(arxiv_id)
                paper = repo.get(arxiv_id)
                if paper:
                    order.append(f"[arXiv:{arxiv_id}] {paper.title} — {', '.join(paper.authors)}")
                else:
                    order.append(f"[arXiv:{arxiv_id}] 未找到元数据")
    return order


def _format_output(final: FinalReview) -> str:
    sections = "\n\n".join(
        f"## {draft.section_id}:\n{draft.content}" for draft in final.sections
    )
    references = "\n".join(final.references)
    stats = json.dumps(final.stats, ensure_ascii=False, indent=2)
    return (
        f"# {final.topic}\n\n" +
        sections +
        "\n\n# References\n" +
        references +
        "\n\n# 审查统计\n" +
        stats
    )


def _write_output(final: FinalReview) -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    path = output_dir / f"{final.request_id}.md"
    path.write_text(_format_output(final), encoding="utf-8")
