"""Tests for SmartSplit planner (domain detection, decomposition & synthesis)."""

from __future__ import annotations

import json

import pytest

from smartsplit.models import Mode, RouteResult, Subtask, TaskType
from smartsplit.planner import MAX_SUBTASKS, Planner, _DecomposeCache, _extract_json, detect_domains

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


# ── Domain Detection ────────────────────────────────────────


class TestDomainDetection:
    def test_code_domain_detected(self):
        domains = detect_domains("Write a Python function to sort a list")
        domain_names = [d for d, _ in domains]
        assert "code" in domain_names

    def test_math_domain_detected(self):
        domains = detect_domains("Calculate the integral of x squared from 0 to 1")
        domain_names = [d for d, _ in domains]
        assert "math" in domain_names

    def test_translation_domain_detected(self):
        domains = detect_domains("Translate this text to French: hello world")
        domain_names = [d for d, _ in domains]
        assert "translation" in domain_names

    def test_creative_domain_detected(self):
        domains = detect_domains("Write a short story about a dragon and a knight")
        domain_names = [d for d, _ in domains]
        assert "creative" in domain_names

    def test_extraction_domain_detected(self):
        domains = detect_domains("Extract all email addresses from this text and list them")
        domain_names = [d for d, _ in domains]
        assert "extraction" in domain_names

    def test_factual_domain_detected(self):
        domains = detect_domains("What is the capital of France?")
        domain_names = [d for d, _ in domains]
        assert "factual" in domain_names

    def test_multi_domain_detected(self):
        domains = detect_domains(_MULTI_DOMAIN_PROMPT)
        domain_names = [d for d, _ in domains]
        assert len(domain_names) >= 2

    def test_empty_prompt_returns_empty(self):
        domains = detect_domains("")
        assert domains == []

    def test_returns_sorted_by_confidence(self):
        domains = detect_domains("Write a Python function to calculate the factorial")
        if len(domains) > 1:
            scores = [s for _, s in domains]
            assert scores == sorted(scores, reverse=True)


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
        """Single-domain prompts should route direct without calling the decomposition LLM."""
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_SINGLE_DOMAIN_PROMPT)
        assert len(subtasks) == 1
        assert subtasks[0].type == TaskType.CODE
        assert subtasks[0].content == _SINGLE_DOMAIN_PROMPT
        mock_registry.call_free_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_multi_domain_triggers_decomposition(self, mock_registry):
        """Multi-domain prompts should call the LLM for decomposition."""
        mock_registry.call_free_llm.return_value = json.dumps(
            [
                {"type": "code", "content": "Fix the login bug", "complexity": "high"},
                {"type": "general", "content": "Write error message", "complexity": "low"},
            ]
        )
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert len(subtasks) == 2
        # call_free_llm called for decomposition + context injection
        assert mock_registry.call_free_llm.call_count >= 1

    @pytest.mark.asyncio
    async def test_fallback_on_bad_json(self, mock_registry):
        """If decomposition LLM returns garbage, fall back to single task."""
        mock_registry.call_free_llm.return_value = "This is not JSON at all"
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert len(subtasks) == 1
        assert subtasks[0].content == _MULTI_DOMAIN_PROMPT

    @pytest.mark.asyncio
    async def test_fallback_on_provider_failure(self, mock_registry):
        """If the planner LLM is down, fall back to single task."""
        mock_registry.call_free_llm.side_effect = RuntimeError("no LLM")
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert len(subtasks) == 1

    @pytest.mark.asyncio
    async def test_markdown_wrapped_json(self, mock_registry):
        mock_registry.call_free_llm.return_value = (
            '```json\n[{"type": "code", "content": "hello", "complexity": "low"}]\n```'
        )
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert subtasks[0].type == TaskType.CODE

    @pytest.mark.asyncio
    async def test_dict_wrapped_in_list(self, mock_registry):
        mock_registry.call_free_llm.return_value = json.dumps({"type": "code", "content": "hello", "complexity": "low"})
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert len(subtasks) == 1

    @pytest.mark.asyncio
    async def test_depends_on_parsed(self, mock_registry):
        """Subtasks with depends_on should preserve the dependency index."""
        mock_registry.call_free_llm.return_value = json.dumps(
            [
                {"type": "web_search", "content": "search X", "complexity": "low", "depends_on": None},
                {"type": "summarize", "content": "summarize results", "complexity": "low", "depends_on": 0},
            ]
        )
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert len(subtasks) == 2
        assert subtasks[0].depends_on is None
        assert subtasks[1].depends_on == 0

    @pytest.mark.asyncio
    async def test_invalid_task_type_fallback(self, mock_registry):
        """If the LLM returns an unknown task type, Pydantic validation fails -> fallback."""
        mock_registry.call_free_llm.return_value = json.dumps(
            [{"type": "quantum_computing", "content": "test", "complexity": "low"}]
        )
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        # Fallback: single task with original prompt
        assert len(subtasks) == 1
        assert subtasks[0].content == _MULTI_DOMAIN_PROMPT

    @pytest.mark.asyncio
    async def test_max_subtasks_economy(self, mock_registry):
        """Economy mode should cap at 3 subtasks."""
        mock_registry.call_free_llm.return_value = json.dumps(
            [
                {"type": "code", "content": "task1", "complexity": "low"},
                {"type": "general", "content": "task2", "complexity": "low"},
                {"type": "reasoning", "content": "task3", "complexity": "low"},
                {"type": "summarize", "content": "task4", "complexity": "low"},
                {"type": "extraction", "content": "task5", "complexity": "low"},
            ]
        )
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT, mode=Mode.ECONOMY)
        assert len(subtasks) <= MAX_SUBTASKS[Mode.ECONOMY]

    @pytest.mark.asyncio
    async def test_max_subtasks_quality(self, mock_registry):
        """Quality mode allows up to 8 subtasks."""
        assert MAX_SUBTASKS[Mode.QUALITY] == 8

    @pytest.mark.asyncio
    async def test_ambiguous_prompt_defaults_to_general(self, mock_registry):
        """Prompts with no clear domain should route as general."""
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_AMBIGUOUS_PROMPT)
        assert len(subtasks) == 1
        # Should not call LLM for decomposition (single or no domain)
        mock_registry.call_free_llm.assert_not_called()


# ── Context Injection ───────────────────────────────────────


class TestContextInjection:
    @pytest.mark.asyncio
    async def test_context_injected_on_multi_subtask(self, mock_registry):
        """When decomposition produces 2+ subtasks, context should be injected."""
        mock_registry.call_free_llm.side_effect = [
            # First call: decomposition
            json.dumps(
                [
                    {"type": "code", "content": "Fix bug", "complexity": "high"},
                    {"type": "general", "content": "Write message", "complexity": "low"},
                ]
            ),
            # Second call: context summary
            "User wants to fix a login bug and write an error message.",
        ]
        planner = Planner(mock_registry)
        subtasks = await planner.decompose(_MULTI_DOMAIN_PROMPT)
        assert len(subtasks) == 2
        # Both subtasks should contain the context prefix
        for st in subtasks:
            assert "[Context:" in st.content

    @pytest.mark.asyncio
    async def test_context_injection_failure_uses_raw_subtasks(self, mock_registry):
        """If context generation fails, subtasks should be returned as-is."""
        mock_registry.call_free_llm.side_effect = [
            # First call: decomposition succeeds
            json.dumps(
                [
                    {"type": "code", "content": "Fix bug", "complexity": "high"},
                    {"type": "general", "content": "Write message", "complexity": "low"},
                ]
            ),
            # Second call: context generation fails
            RuntimeError("down"),
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
        mock_registry.call_free_llm.side_effect = RuntimeError("down")
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
        planner = Planner(mock_registry)
        # Use a prompt long enough to trigger domain detection (>80 chars)
        prompt = "Explain in detail what a Python decorator is and how to use it in real projects"
        result1 = await planner.decompose(prompt)
        result2 = await planner.decompose(prompt)
        assert result1 == result2
        assert planner.cache.hits == 1
        # LLM should not be called on the second decompose
        assert mock_registry.call_free_llm.call_count == 0  # single domain, no LLM needed
