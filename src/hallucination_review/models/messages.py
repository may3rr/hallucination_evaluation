from __future__ import annotations

"""Message dataclasses shared across agents."""

from dataclasses import dataclass, field
from typing import List, Optional

from .papers import PaperMetadata


@dataclass(slots=True)
class SectionPlan:
    section_id: str
    title: str
    focus: str
    key_questions: List[str] = field(default_factory=list)


@dataclass(slots=True)
class TopicRequest:
    request_id: str
    topic: str
    instructions: Optional[str] = None


@dataclass(slots=True)
class OutlineResult:
    request_id: str
    sections: List[SectionPlan]


@dataclass(slots=True)
class SearchTask:
    request_id: str
    topic: str
    section: SectionPlan


@dataclass(slots=True)
class SearchResult:
    request_id: str
    section_id: str
    papers: List[PaperMetadata]


@dataclass(slots=True)
class DraftTask:
    request_id: str
    topic: str
    sections: List[SectionPlan]
    section_papers: dict[str, List[PaperMetadata]]


@dataclass(slots=True)
class SectionDraft:
    request_id: str
    section_id: str
    content: str


@dataclass(slots=True)
class ReviewRequest:
    request_id: str
    section_id: str
    content: str


@dataclass(slots=True)
class ClaimAssessment:
    claim: str
    arxiv_id: str
    verdict: str
    rationale: str


@dataclass(slots=True)
class ReviewReport:
    request_id: str
    section_id: str
    assessments: List[ClaimAssessment]
    approved: bool
    summary: str


@dataclass(slots=True)
class RevisionRequest:
    request_id: str
    section_id: str
    original: str
    feedback: str


@dataclass(slots=True)
class FinalReview:
    request_id: str
    topic: str
    outline: List[SectionPlan]
    sections: List[SectionDraft]
    references: List[str]
    stats: dict
