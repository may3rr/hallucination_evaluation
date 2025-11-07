from __future__ import annotations

"""Hallucination detection agent."""

import re
from collections import Counter

from autogen_core import MessageContext, message_handler

from ..llm.prompts import REVIEW_SYSTEM_PROMPT, REVIEW_USER_PROMPT
from ..models.messages import ClaimAssessment, ReviewReport, ReviewRequest
from ..utils.parsing import ensure_json
from .base import LiteratureAgent


_CITATION_PATTERN = re.compile(r"\[(?:arXiv:)?(?P<id>[0-9.]+)\]", re.IGNORECASE)


class ReviewAgent(LiteratureAgent):
    """Checks drafted sections for hallucinations."""

    def __init__(self, *, llm, paper_repo) -> None:  # type: ignore[override]
        super().__init__("Hallucination reviewer", llm=llm, paper_repo=paper_repo)

    @message_handler
    async def handle_review(self, message: ReviewRequest, ctx: MessageContext) -> None:
        claims = _extract_claims(message.content)
        assessments: list[ClaimAssessment] = []
        verdict_counter: Counter[str] = Counter()

        for claim_text, arxiv_id in claims:
            paper = self.papers.get(arxiv_id)
            if not paper:
                assessments.append(
                    ClaimAssessment(
                        claim=claim_text,
                        arxiv_id=arxiv_id,
                        verdict="metadata_error",
                        rationale="Citation not found in repository",
                    )
                )
                verdict_counter["metadata_error"] += 1
                continue

            prompt = REVIEW_USER_PROMPT.format(
                claim=claim_text,
                title=paper.title,
                authors=", ".join(paper.authors),
                arxiv_id=arxiv_id,
                abstract=paper.summary,
            )
            result = await self.llm.structured(
                system_prompt=REVIEW_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
            data = ensure_json(result)
            verdict = data.get("support", "metadata_error")
            rationale = data.get("rationale", "")
            assessments.append(
                ClaimAssessment(
                    claim=claim_text,
                    arxiv_id=arxiv_id,
                    verdict=verdict,
                    rationale=rationale,
                )
            )
            verdict_counter[verdict] += 1

        approved = verdict_counter.get("unsupported", 0) == 0 and verdict_counter.get("metadata_error", 0) == 0
        summary = (
            "All claims supported" if approved else "Issues detected: " + ", ".join(
                f"{k}={v}" for k, v in verdict_counter.items() if v
            )
        )

        await self.publish_message(
            ReviewReport(
                request_id=message.request_id,
                section_id=message.section_id,
                assessments=assessments,
                approved=approved,
                summary=summary,
            ),
            topic_id=ctx.topic_id,
        )


def _extract_claims(text: str) -> list[tuple[str, str]]:
    claims: list[tuple[str, str]] = []
    for match in _CITATION_PATTERN.finditer(text):
        arxiv_id = match.group("id")
        boundaries = [
            text.rfind(". ", 0, match.start()),
            text.rfind("。", 0, match.start()),
            text.rfind("; ", 0, match.start()),
            text.rfind("；", 0, match.start()),
            text.rfind("\n", 0, match.start()),
        ]
        start = max(b for b in boundaries if b != -1) if any(b != -1 for b in boundaries) else -1
        if start == -1:
            start = 0
        else:
            delimiter = text[start:start + 2]
            start = start + (2 if delimiter in {". ", "; "} else 1)
        claim = text[start:match.end()].strip()
        claims.append((claim, arxiv_id))
    return claims
