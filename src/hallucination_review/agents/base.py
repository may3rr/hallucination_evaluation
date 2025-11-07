from __future__ import annotations

"""Common agent base classes."""

from autogen_core import RoutedAgent

from ..llm.client import LLMClient
from ..models.papers import PaperRepository


class LiteratureAgent(RoutedAgent):
    """Base class providing shared resources."""

    def __init__(self, description: str, *, llm: LLMClient, paper_repo: PaperRepository) -> None:
        super().__init__(description)
        self._llm = llm
        self._papers = paper_repo

    @property
    def llm(self) -> LLMClient:
        return self._llm

    @property
    def papers(self) -> PaperRepository:
        return self._papers
