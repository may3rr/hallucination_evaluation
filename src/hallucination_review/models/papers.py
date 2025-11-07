from __future__ import annotations

"""Paper metadata models and repositories."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass(slots=True)
class PaperMetadata:
    arxiv_id: str
    title: str
    summary: str
    authors: List[str]
    url: str
    published: str
    updated: str | None = None

    def short_citation(self) -> str:
        return f"{self.title} ({', '.join(self.authors[:2])}{' et al.' if len(self.authors) > 2 else ''}, {self.published[:4]})"


class PaperRepository:
    """In-memory storage for paper metadata."""

    def __init__(self) -> None:
        self._papers: Dict[str, PaperMetadata] = {}

    def upsert_many(self, papers: Iterable[PaperMetadata]) -> None:
        for paper in papers:
            self._papers[paper.arxiv_id] = paper

    def get(self, arxiv_id: str) -> Optional[PaperMetadata]:
        return self._papers.get(arxiv_id)

    def all(self) -> List[PaperMetadata]:
        return list(self._papers.values())
