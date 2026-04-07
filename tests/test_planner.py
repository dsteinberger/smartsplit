"""Tests for SmartSplit planner (domain detection, decomposition & synthesis)."""

from __future__ import annotations

import json

import pytest

from smartsplit.exceptions import ProviderError
from smartsplit.models import Mode, RouteResult, Subtask, TaskType
from smartsplit.planner import (
    MAX_SUBTASKS,
    Planner,
    _DecomposeCache,
    _extract_json,
    detect_domains,
)

# Prompts for testing — designed to trigger specific domain detection behaviors.
# Multi-domain prompt (code + writing): triggers decomposition
_MULTI_DOMAIN_PROMPT = (
    "Fix the Python bug in the login function and write a user-facing error message "
    "explaining what went wrong to the end user in a friendly tone"
)
# Single-domain prompt (code only): should NOT trigger decomposition (must be >80 chars)
_SINGLE_DOMAIN_PROMPT = (
    "Implement a Python function that sorts a list of dictionaries by a given key with error handling"
)
# Short prompt: below threshold, skips everything
_SHORT_PROMPT = "Hello"
# No clear domain: should default to general
_AMBIGUOUS_PROMPT = "Tell me something interesting about the world that I might not know about please"


# ── JSON extraction ──────────────────────────────────────────


class TestExtractJson:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            pytest.param('[{"type": "general"}]', '[{"type": "general"}]', id="plain-json"),
            pytest.param('```json\n[{"type": "code"}]\n```', '[{"type": "code"}]', id="markdown-json-fence"),
            pytest.param('```\n[{"type": "code"}]\n```', '[{"type": "code"}]', id="markdown-no-lang"),
            pytest.param('Here:\n```json\n[{"type": "code"}]\n```\nDone!', '[{"type": "code"}]', id="surrounded"),
            pytest.param('  \n  [{"type": "code"}]  \n  ', '[{"type": "code"}]', id="whitespace"),
        ],
    )
    def test_extraction(self, raw, expected):
        assert _extract_json(raw) == expected


# ── Keyword Fallback Detection ─────────────────────────────


class TestKeywordFallbackDetection:
    """Tests for the keyword-based fallback used when LLM classification fails."""

    def test_code_keywords(self):
        domains = detect_domains("Write a Python function to sort a list")
        domain_names = [d for d, _ in domains]
        assert "code" in domain_names

    def test_math_keywords(self):
        domains = detect_domains("Calculate the integral of x squared from 0 to 1")
        domain_names = [d for d, _ in domains]
        assert "math" in domain_names

    def test_empty_prompt(self):
        assert detect_domains("") == []

    def test_sorted_by_confidence(self):
        domains = detect_domains("Write a Python function to calculate the factorial")
        if len(domains) > 1:
            scores = [s for _, s in domains]
            assert scores == sorted(scores, reverse=True)


# ── LLM Domain Classification ──────────────────────────────


class TestClassifyDomains:
    @pytest.mark.asyncio
    async def test_llm_returns_valid_domains(self, mock_registry):
        mock_registry.call_free_llm.return_value = '["code", "math"]'
        planner = Planner(mock_registry)
        domains = await planner.classify_domains("Write a function to compute factorial")
        assert domains == ["code", "math"]

    @pytest.mark.asyncio
    async def test_llm_filters_invalid_domains(self, mock_registry):
        mock_registry.call_free_llm.return_value = '["code", "quantum_physics", "math"]'
        planner = Planner(mock_registry)
        domains = await planner.classify_domains("test prompt")
        assert domains == ["code", "math"]
        assert "quantum_physics" not in domains

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_keywords(self, mock_registry):
        mock_registry.call_free_llm.side_effect = ProviderError("test", "no LLM")
        planner = Planner(mock_registry)
        domains = await planner.classify_domains("Write a Python function to sort a list")
        assert "code" in domains

    @pytest.mark.asyncio
    async def test_llm_returns_garbage_falls_back(self, mock_registry):
        mock_registry.call_free_llm.return_value = "I don't understand"
        planner = Planner(mock_registry)
        # Keyword fallback should still work for English prompts
        domains = await planner.classify_domains("Calculate the integral of x squared")
        assert "math" in domains

    @pytest.mark.asyncio
    async def test_llm_returns_empty_list_falls_back(self, mock_registry):
        mock_registry.call_free_llm.return_value = "[]"
        planner = Planner(mock_registry)
        domains = await planner.classify_domains("Write a Python function")
        # Falls back to keywords since LLM returned no valid domains
        assert "code" in domains

    @pytest.mark.asyncio
    async def test_llm_strips_markdown_fences(self, mock_registry):
        mock_registry.call_free_llm.return_value = '```json\n["translation"]\n```'
        planner = Planner(mock_registry)
        domains = await planner.classify_domains("Traduisez ce texte en anglais")
        assert domains == ["translation"]


# ── Decompose ────────────────────────────────────────────────


class TestDecompose:
    @pytest.mark.asyncio
    async def test_short_prompt_skips_decomposition(self, mock_registry):
        """Prompts below the threshold should not call the LLM at all."""
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_SHORT_PROMPT)
        assert len(subtasks) == 1
        assert subtasks[0].content == _SHORT_PROMPT
        assert subtasks[0].complexity.value == "low"
        mock_registry.call_free_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_domain_skips_llm_decomposition(self, mock_registry):
        """Single-domain prompts should route direct without decomposition LLM call."""
        # classify_domains returns single domain → no decomposition
        mock_registry.call_free_llm.return_value = '["code"]'
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_SINGLE_DOMAIN_PROMPT)
        assert len(subtasks) == 1
        assert subtasks[0].type == TaskType.CODE
        assert subtasks[0].content == _SINGLE_DOMAIN_PROMPT
        # Only 1 call: classification. No decomposition call.
        assert mock_registry.call_free_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_multi_domain_triggers_decomposition(self, mock_registry):
        """Multi-domain prompts should call the LLM for decomposition."""
        mock_registry.call_free_llm.side_effect = [
            # 1st call: classification
            '["code", "writing"]',
            # 2nd call: decomposition
            json.dumps(
                [
                    {"type": "code", "content": "Fix the login bug", "complexity": "high"},
                    {"type": "general", "content": "Write error message", "complexity": "low"},
                ]
            ),
            # 3rd call: context injection
            "User wants to fix a login bug and write an error message.",
        ]
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert len(subtasks) == 2
        # classify + decompose + context injection = 3 calls
        assert mock_registry.call_free_llm.call_count == 3

    @pytest.mark.asyncio
    async def test_fallback_on_bad_json(self, mock_registry):
        """If decomposition LLM returns garbage, fall back to single task."""
        mock_registry.call_free_llm.side_effect = [
            '["code", "writing"]',  # classification
            "This is not JSON at all",  # decomposition fails
        ]
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert len(subtasks) == 1
        assert subtasks[0].content == _MULTI_DOMAIN_PROMPT

    @pytest.mark.asyncio
    async def test_fallback_on_provider_failure(self, mock_registry):
        """If the planner LLM is down, fall back to single task."""
        mock_registry.call_free_llm.side_effect = ProviderError("test", "no LLM")
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        # Classification fails → keyword fallback detects multi-domain → decomposition fails → single task
        assert len(subtasks) == 1

    @pytest.mark.asyncio
    async def test_markdown_wrapped_json(self, mock_registry):
        mock_registry.call_free_llm.side_effect = [
            '["code", "writing"]',
            '```json\n[{"type": "code", "content": "hello", "complexity": "low"}]\n```',
        ]
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert subtasks[0].type == TaskType.CODE

    @pytest.mark.asyncio
    async def test_dict_wrapped_in_list(self, mock_registry):
        mock_registry.call_free_llm.side_effect = [
            '["code", "writing"]',
            json.dumps({"type": "code", "content": "hello", "complexity": "low"}),
        ]
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert len(subtasks) == 1

    @pytest.mark.asyncio
    async def test_depends_on_parsed(self, mock_registry):
        """Subtasks with depends_on should preserve the dependency index."""
        mock_registry.call_free_llm.side_effect = [
            '["web_search", "summarize"]',
            json.dumps(
                [
                    {"type": "web_search", "content": "search X", "complexity": "low", "depends_on": None},
                    {"type": "summarize", "content": "summarize results", "complexity": "low", "depends_on": 0},
                ]
            ),
            "Context summary",
        ]
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert len(subtasks) == 2
        assert subtasks[0].depends_on is None
        assert subtasks[1].depends_on == 0

    @pytest.mark.asyncio
    async def test_invalid_task_type_fallback(self, mock_registry):
        """If the LLM returns an unknown task type, Pydantic validation fails -> fallback."""
        mock_registry.call_free_llm.side_effect = [
            '["code", "writing"]',
            json.dumps([{"type": "quantum_computing", "content": "test", "complexity": "low"}]),
        ]
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert len(subtasks) == 1
        assert subtasks[0].content == _MULTI_DOMAIN_PROMPT

    @pytest.mark.asyncio
    async def test_max_subtasks_economy(self, mock_registry):
        """Economy mode should cap at 3 subtasks."""
        mock_registry.call_free_llm.side_effect = [
            '["code", "writing"]',
            json.dumps(
                [
                    {"type": "code", "content": "task1", "complexity": "low"},
                    {"type": "general", "content": "task2", "complexity": "low"},
                    {"type": "reasoning", "content": "task3", "complexity": "low"},
                    {"type": "summarize", "content": "task4", "complexity": "low"},
                    {"type": "extraction", "content": "task5", "complexity": "low"},
                ]
            ),
            "Context summary for multi-subtask",
        ]
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT, mode=Mode.ECONOMY)
        assert len(subtasks) <= MAX_SUBTASKS[Mode.ECONOMY]

    @pytest.mark.asyncio
    async def test_truncation_invalidates_depends_on(self, mock_registry):
        """depends_on pointing to truncated subtasks should be set to None."""
        mock_registry.call_free_llm.side_effect = [
            '["code", "writing"]',
            json.dumps(
                [
                    {"type": "code", "content": "task1", "complexity": "low"},
                    {"type": "general", "content": "task2", "complexity": "low", "depends_on": 0},
                    {"type": "reasoning", "content": "task3", "complexity": "low", "depends_on": 1},
                    {"type": "summarize", "content": "task4", "complexity": "low", "depends_on": 3},
                    {"type": "extraction", "content": "task5", "complexity": "low", "depends_on": 4},
                ]
            ),
            "Context summary for truncated subtasks",
        ]
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT, mode=Mode.ECONOMY)
        assert len(subtasks) <= MAX_SUBTASKS[Mode.ECONOMY]
        # depends_on=0 and depends_on=1 are valid (within truncated range)
        # depends_on=3 and depends_on=4 would be invalid but those subtasks were truncated
        for st in subtasks:
            if st.depends_on is not None:
                assert st.depends_on < len(subtasks), f"depends_on={st.depends_on} >= {len(subtasks)}"

    @pytest.mark.asyncio
    async def test_messages_deep_copied_between_subtasks(self, mock_registry):
        """Messages should be deep-copied so subtasks don't share mutable state."""
        mock_registry.call_free_llm.side_effect = [
            '["code", "writing"]',
            json.dumps(
                [
                    {"type": "code", "content": "Fix bug", "complexity": "high"},
                    {"type": "general", "content": "Write message", "complexity": "low"},
                ]
            ),
            "Context summary",
        ]
        messages = [{"role": "user", "content": "original"}]
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT, messages=messages)
        assert len(subtasks) == 2
        # Mutating one subtask's messages should NOT affect the other
        subtasks[0].messages[0]["content"] = "MUTATED"
        assert subtasks[1].messages[0]["content"] == "original"
        # Original messages should also be unchanged
        assert messages[0]["content"] == "original"

    @pytest.mark.asyncio
    async def test_max_subtasks_quality(self, mock_registry):
        """Quality mode allows up to 8 subtasks."""
        assert MAX_SUBTASKS[Mode.QUALITY] == 8

    @pytest.mark.asyncio
    async def test_no_domain_defaults_to_general(self, mock_registry):
        """Prompts with no clear domain should route as general."""
        mock_registry.call_free_llm.return_value = "[]"
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_AMBIGUOUS_PROMPT)
        assert len(subtasks) == 1
        assert subtasks[0].type == TaskType.GENERAL


# ── Context Injection ───────────────────────────────────────


class TestContextInjection:
    @pytest.mark.asyncio
    async def test_context_injected_on_multi_subtask(self, mock_registry):
        """When decomposition produces 2+ subtasks, context should be injected."""
        mock_registry.call_free_llm.side_effect = [
            # 1st call: classification
            '["code", "writing"]',
            # 2nd call: decomposition
            json.dumps(
                [
                    {"type": "code", "content": "Fix bug", "complexity": "high"},
                    {"type": "general", "content": "Write message", "complexity": "low"},
                ]
            ),
            # 3rd call: context summary
            "User wants to fix a login bug and write an error message.",
        ]
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert len(subtasks) == 2
        for st in subtasks:
            assert "[Context:" in st.content

    @pytest.mark.asyncio
    async def test_context_injection_failure_uses_raw_subtasks(self, mock_registry):
        """If context generation fails, subtasks should be returned as-is."""
        mock_registry.call_free_llm.side_effect = [
            # 1st call: classification
            '["code", "writing"]',
            # 2nd call: decomposition succeeds
            json.dumps(
                [
                    {"type": "code", "content": "Fix bug", "complexity": "high"},
                    {"type": "general", "content": "Write message", "complexity": "low"},
                ]
            ),
            # 3rd call: context generation fails
            ProviderError("test", "down"),
        ]
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert len(subtasks) == 2
        assert "[Context:" not in subtasks[0].content


# ── Synthesize ───────────────────────────────────────────────


class TestSynthesize:
    @pytest.mark.asyncio
    async def test_synthesis_calls_free_llm(self, mock_registry):
        mock_registry.call_free_llm.return_value = "Unified response."
        planner = Planner(mock_registry)
        results = [
            RouteResult(type=TaskType.WEB_SEARCH, response="Found info", provider="serper"),
            RouteResult(type=TaskType.CODE, response="def foo(): ...", provider="groq"),
        ]
        synthesis = await planner.synthesize("Research and code", results)
        assert synthesis == "Unified response."
        mock_registry.call_free_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesis_fallback_concatenates(self, mock_registry):
        mock_registry.call_free_llm.side_effect = ProviderError("test", "down")
        planner = Planner(mock_registry)
        results = [
            RouteResult(type=TaskType.WEB_SEARCH, response="Result A", provider="serper"),
            RouteResult(type=TaskType.CODE, response="Result B", provider="groq"),
        ]
        synthesis = await planner.synthesize("test", results)
        assert "Result A" in synthesis
        assert "Result B" in synthesis


# ── Decompose Cache ────────────────────────────────────────────


class TestDecomposeCache:
    def test_miss_then_hit(self):
        cache = _DecomposeCache()
        subtasks = [Subtask(content="hello", complexity="low")]
        assert cache.get("key1") is None
        assert cache.misses == 1
        cache.put("key1", subtasks)
        assert cache.get("key1") == subtasks
        assert cache.hits == 1

    def test_ttl_expiry(self):
        import time as time_mod

        cache = _DecomposeCache(ttl=1)
        cache.put("key1", [Subtask(content="hello")])
        assert cache.get("key1") is not None
        # Fake expiry by backdating
        cache._store["key1"] = (time_mod.time() - 2, cache._store["key1"][1])
        assert cache.get("key1") is None
        assert cache.misses == 1

    def test_max_size_eviction(self):
        cache = _DecomposeCache(max_size=3)
        for i in range(4):
            cache.put(f"key{i}", [Subtask(content=f"task{i}")])
        # First key should be evicted
        assert cache.get("key0") is None
        assert cache.get("key3") is not None

    @pytest.mark.asyncio
    async def test_cache_integrated_in_planner(self, mock_registry):
        """Second call with same prompt should hit cache and not call LLM."""
        mock_registry.call_free_llm.return_value = '["code"]'
        planner = Planner(mock_registry)
        # Prompt must be >80 chars to trigger domain detection
        prompt = "Explain in detail what a Python decorator is and how to use it in real world projects today"
        result1 = await planner.decompose(prompt)
        result2 = await planner.decompose(prompt)
        assert result1 == result2
        assert planner.cache.hits == 1
        # LLM called once for classification on first decompose, cached on second
        assert mock_registry.call_free_llm.call_count == 1
