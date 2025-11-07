from __future__ import annotations

"""Async arXiv search utilities."""

import asyncio
import re
from typing import Iterable, List
import xml.etree.ElementTree as ET

import httpx

from ..config import get_settings
from ..models.papers import PaperMetadata


_ARXIV_API = "https://export.arxiv.org/api/query"


class ArxivSearchError(RuntimeError):
    """Raised when the arXiv API returns an error."""


class ArxivClient:
    """Simple asynchronous arXiv client with rate limiting."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)
        self._rate_lock = asyncio.Lock()
        self._last_call: float | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def _respect_rate_limit(self) -> None:
        async with self._rate_lock:
            now = asyncio.get_event_loop().time()
            if self._last_call is not None and now - self._last_call < 3.0:
                await asyncio.sleep(3.0 - (now - self._last_call))
            self._last_call = asyncio.get_event_loop().time()

    async def search(self, *, query: str) -> List[PaperMetadata]:
        settings = get_settings()
        params = {
            "search_query": query,
            "start": 0,
            "max_results": settings.arxiv_max_results,
            "sortBy": settings.arxiv_sort_by,
        }

        await self._respect_rate_limit()
        response = await self._client.get(_ARXIV_API, params=params)
        if response.status_code != 200:
            raise ArxivSearchError(f"arXiv API error: {response.status_code} {response.text}")

        return list(_parse_feed(response.text))

    async def fetch_metadata(self, arxiv_id: str) -> PaperMetadata | None:
        query = f"id:{arxiv_id}"
        results = await self.search(query=query)
        return results[0] if results else None


def _parse_feed(feed: str) -> Iterable[PaperMetadata]:
    root = ET.fromstring(feed)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.findtext("atom:id", default="", namespaces=ns)
        arxiv_id = arxiv_id.split("/")[-1]
        title = _clean(entry.findtext("atom:title", default="", namespaces=ns))
        summary = _clean(entry.findtext("atom:summary", default="", namespaces=ns))
        authors = [
            _clean(author.findtext("atom:name", default="", namespaces=ns))
            for author in entry.findall("atom:author", ns)
        ]
        link = ""
        for link_elem in entry.findall("atom:link", ns):
            if link_elem.attrib.get("type") == "text/html":
                link = link_elem.attrib.get("href", "")
                break
        published = entry.findtext("atom:published", default="", namespaces=ns)
        updated = entry.findtext("atom:updated", default="", namespaces=ns)

        yield PaperMetadata(
            arxiv_id=arxiv_id,
            title=title,
            summary=summary,
            authors=authors,
            url=link,
            published=published,
            updated=updated,
        )


def _clean(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "")
    return value.strip()
