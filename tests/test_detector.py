"""Tests for detect_with_llm — LLM-based triage fallback."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from smartsplit.triage.detector import TriageDecision, detect_with_llm


@pytest.fixture
def mock_registry():
    """A MagicMock registry with call_free_llm as AsyncMock."""
    registry = MagicMock()
    registry.call_free_llm = AsyncMock()
    return registry


class TestDetectWithLlm:
    async def test_enrich_with_web_search(self, mock_registry: MagicMock):
        mock_registry.call_free_llm.return_value = json.dumps({"decision": "enrich", "enrichments": ["web_search"]})
        decision, enrichments = await detect_with_llm("What are the best Python web frameworks in 2026?", mock_registry)
        assert decision == TriageDecision.ENRICH
        assert enrichments == ["web_search"]

    async def test_enrich_with_multiple_enrichments(self, mock_registry: MagicMock):
        mock_registry.call_free_llm.return_value = json.dumps(
            {"decision": "enrich", "enrichments": ["web_search", "multi_perspective"]}
        )
        decision, enrichments = await detect_with_llm("Compare React vs Vue vs Svelte for a new project", mock_registry)
        assert decision == TriageDecision.ENRICH
        assert enrichments == ["web_search", "multi_perspective"]

    async def test_transparent_when_llm_says_transparent(self, mock_registry: MagicMock):
        mock_registry.call_free_llm.return_value = json.dumps({"decision": "transparent", "enrichments": []})
        decision, enrichments = await detect_with_llm("Write a hello world in Python", mock_registry)
        assert decision == TriageDecision.TRANSPARENT
        assert enrichments == []

    async def test_transparent_on_invalid_json(self, mock_registry: MagicMock):
        mock_registry.call_free_llm.return_value = "this is not json at all"
        decision, enrichments = await detect_with_llm("some prompt", mock_registry)
        assert decision == TriageDecision.TRANSPARENT
        assert enrichments == []

    async def test_transparent_on_non_dict_response(self, mock_registry: MagicMock):
        mock_registry.call_free_llm.return_value = json.dumps(["enrich", "web_search"])
        decision, enrichments = await detect_with_llm("some prompt", mock_registry)
        assert decision == TriageDecision.TRANSPARENT
        assert enrichments == []

    async def test_transparent_on_call_free_llm_exception(self, mock_registry: MagicMock):
        mock_registry.call_free_llm.side_effect = RuntimeError("provider unavailable")
        decision, enrichments = await detect_with_llm("some prompt", mock_registry)
        assert decision == TriageDecision.TRANSPARENT
        assert enrichments == []

    async def test_filters_invalid_enrichment_types(self, mock_registry: MagicMock):
        mock_registry.call_free_llm.return_value = json.dumps(
            {"decision": "enrich", "enrichments": ["web_search", "magic_answer", "pre_analysis", "hallucinate"]}
        )
        decision, enrichments = await detect_with_llm("some complex prompt", mock_registry)
        assert decision == TriageDecision.ENRICH
        assert enrichments == ["web_search", "pre_analysis"]

    async def test_transparent_when_all_enrichments_invalid(self, mock_registry: MagicMock):
        mock_registry.call_free_llm.return_value = json.dumps(
            {"decision": "enrich", "enrichments": ["magic_answer", "hallucinate"]}
        )
        decision, enrichments = await detect_with_llm("some prompt", mock_registry)
        assert decision == TriageDecision.TRANSPARENT
        assert enrichments == []
