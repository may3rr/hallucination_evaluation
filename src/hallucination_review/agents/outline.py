from __future__ import annotations

"""Agent responsible for producing literature review outlines."""

from autogen_core import MessageContext, message_handler

from ..llm.prompts import OUTLINE_SYSTEM_PROMPT, OUTLINE_USER_PROMPT
from ..models.messages import OutlineResult, SectionPlan, TopicRequest
from ..utils.parsing import ensure_json
from .base import LiteratureAgent


class OutlineAgent(LiteratureAgent):
    """Generates outline sections from a topic."""

    def __init__(self, *, llm, paper_repo) -> None:  # type: ignore[override]
        super().__init__("Outline planner", llm=llm, paper_repo=paper_repo)

    @message_handler
    async def handle_topic(self, message: TopicRequest, ctx: MessageContext) -> None:
        payload = await self.llm.structured(
            system_prompt=OUTLINE_SYSTEM_PROMPT,
            user_prompt=OUTLINE_USER_PROMPT.format(topic=message.topic),
        )
        data = ensure_json(payload)
        sections = []
        for idx, item in enumerate(data.get("sections", []), start=1):
            section_id = item.get("id") or f"sec{idx:02d}"
            title = item.get("title") or f"Section {idx}"
            focus = item.get("focus") or item.get("description") or ""
            key_questions = item.get("key_questions") or item.get("questions") or []
            sections.append(
                SectionPlan(
                    section_id=section_id,
                    title=title,
                    focus=focus,
                    key_questions=list(key_questions),
                )
            )

        await self.publish_message(
            OutlineResult(request_id=message.request_id, sections=sections),
            topic_id=ctx.topic_id,
        )
