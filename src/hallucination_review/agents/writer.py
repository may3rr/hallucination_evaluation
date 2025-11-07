from __future__ import annotations

"""Agent responsible for writing and revising sections."""

from autogen_core import MessageContext, message_handler

from ..llm.prompts import REVISION_SYSTEM_PROMPT, REVISION_USER_PROMPT, WRITER_SYSTEM_PROMPT, WRITER_USER_PROMPT
from ..models.messages import DraftTask, RevisionRequest, SectionDraft
from .base import LiteratureAgent


class WritingAgent(LiteratureAgent):
    """Produces section drafts based on search results."""

    def __init__(self, *, llm, paper_repo) -> None:  # type: ignore[override]
        super().__init__("Literature synthesis specialist", llm=llm, paper_repo=paper_repo)
        self._drafts: dict[str, str] = {}

    @message_handler
    async def handle_draft(self, message: DraftTask, ctx: MessageContext) -> None:
        for section in message.sections:
            papers = message.section_papers.get(section.section_id, [])
            paper_summaries = "\n".join(
                f"- {paper.title} (arXiv:{paper.arxiv_id}): {paper.summary}" for paper in papers
            )
            prompt = WRITER_USER_PROMPT.format(
                topic=message.topic,
                title=section.title,
                focus=section.focus,
                paper_summaries=paper_summaries or "暂无文献，请说明缺失数据",
            )
            content = await self.llm.complete(
                system_prompt=WRITER_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
            self._drafts[section.section_id] = content
            await self.publish_message(
                SectionDraft(
                    request_id=message.request_id,
                    section_id=section.section_id,
                    content=content,
                ),
                topic_id=ctx.topic_id,
            )

    @message_handler
    async def handle_revision(self, message: RevisionRequest, ctx: MessageContext) -> None:
        prompt = REVISION_USER_PROMPT.format(original=message.original, feedback=message.feedback)
        content = await self.llm.complete(
            system_prompt=REVISION_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        self._drafts[message.section_id] = content
        await self.publish_message(
            SectionDraft(
                request_id=message.request_id,
                section_id=message.section_id,
                content=content,
            ),
            topic_id=ctx.topic_id,
        )
