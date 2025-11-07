from __future__ import annotations

"""Configuration helpers for the hallucination evaluation system."""

from dataclasses import dataclass
import os
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    api_key: str
    base_url: str = "https://api.gpt.ge/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.4
    max_output_tokens: int = 2048

    arxiv_max_results: int = 5
    arxiv_sort_by: str = "relevance"

    revision_limit: int = 2

    @staticmethod
    def from_env() -> "Settings":
        api_key = os.getenv("V3_API_KEY")
        if not api_key:
            raise RuntimeError("Missing V3_API_KEY environment variable")

        base_url = os.getenv("LLM_BASE_URL", "https://api.gpt.ge/v1")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.4"))
        max_output_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))

        arxiv_max_results = int(os.getenv("ARXIV_MAX_RESULTS", "5"))
        arxiv_sort_by = os.getenv("ARXIV_SORT_BY", "relevance")
        revision_limit = int(os.getenv("REVISION_LIMIT", "2"))

        return Settings(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            arxiv_max_results=arxiv_max_results,
            arxiv_sort_by=arxiv_sort_by,
            revision_limit=revision_limit,
        )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings."""

    return Settings.from_env()
