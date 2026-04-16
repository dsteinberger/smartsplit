"""Shared JSON helpers — used across planner, detector, and tools modules."""

from __future__ import annotations

import re


def extract_json(text: str) -> str:
    """Strip markdown code fences that LLMs frequently wrap around JSON."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    return match.group(1).strip() if match else text
