from __future__ import annotations

"""Agent responsible for arXiv searches."""

from autogen_core import MessageContext, message_handler

from ..arxiv.client import ArxivClient
from ..llm.prompts import QUERY_SYSTEM_PROMPT, QUERY_USER_PROMPT
from ..models.messages import SearchResult, SearchTask
from ..utils.parsing import ensure_json
from .base import LiteratureAgent


class SearchAgent(LiteratureAgent):
    """Generates search queries and fetches papers from arXiv."""

    def __init__(self, *, llm, paper_repo, arxiv_client: ArxivClient) -> None:  # type: ignore[override]
        super().__init__("ArXiv search specialist", llm=llm, paper_repo=paper_repo)
        self._arxiv = arxiv_client

    @message_handler
    async def handle_search(self, message: SearchTask, ctx: MessageContext) -> None:
        prompt = QUERY_USER_PROMPT.format(
            topic=message.topic,
            title=message.section.title,
            focus=message.section.focus,
        )
        response = await self.llm.structured(
            system_prompt=QUERY_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        data = ensure_json(response)
        query = data.get("query") or message.section.title
        papers = await self._arxiv.search(query=query)
        self.papers.upsert_many(papers)

        await self.publish_message(
            SearchResult(
                request_id=message.request_id,
                section_id=message.section.section_id,
                papers=papers,
            ),
            topic_id=ctx.topic_id,
        )
