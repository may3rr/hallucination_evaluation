"""Utility helpers for parsing LLM outputs."""

from __future__ import annotations

import json
from typing import Any, Dict


class LLMJSONError(RuntimeError):
    """Raised when JSON parsing from the LLM fails."""


def ensure_json(payload: str) -> Dict[str, Any]:
    """Parse an LLM response expected to be JSON."""

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise LLMJSONError(f"Invalid JSON output: {payload}") from exc
