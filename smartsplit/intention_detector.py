"""Intention detector — predicts read-only tool calls an LLM will make.

SmartSplit can pre-execute anticipated read-only tools (file reads, searches,
directory listings) while the brain is still thinking, reducing round-trip
latency in agentic loops.

Pipeline:
1. Inspect the last message in the conversation.
2. If it's a user message → predict which read tools the LLM will call.
3. If it's a tool result → predict what the LLM will read next.
4. Return a ``Prediction`` with anticipated tools (filtered to SAFE_TOOLS).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from smartsplit.exceptions import SmartSplitError
from smartsplit.planner import _extract_json
from smartsplit.tool_registry import FILE_REF_RE as _FILE_REF_RE
from smartsplit.tool_registry import SAFE_TOOLS
from smartsplit.tool_registry import WELL_KNOWN_FILES as _WELL_KNOWN_FILES

if TYPE_CHECKING:
    from smartsplit.providers.registry import ProviderRegistry
    from smartsplit.tool_pattern_learner import ToolPatternLearner

logger = logging.getLogger("smartsplit.intention_detector")

_MIN_CONFIDENCE = 0.7  # conservative: 10% irrelevant content → -23% quality (context rot research)
_FAKE_CONFIDENCE = 0.85  # threshold for FAKE tool_use — skip LLM if rules already reach this

# ── Data classes ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AnticipatedTool:
    """A single tool call the LLM is predicted to make."""

    tool: str
    args: dict[str, str | int | bool | list[str]] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class Prediction:
    """Result of intention detection — zero or more anticipated tool calls."""

    should_anticipate: bool
    confidence: float
    tools: list[AnticipatedTool] = field(default_factory=list)


# ── Null prediction (reused to avoid allocations) ──────────

_NULL_PREDICTION = Prediction(should_anticipate=False, confidence=0.0, tools=[])

# ── Rule-based prediction ──────────────────────────────────
#
# Hardcoded patterns from research on coding agent usage (SWE-bench traces,
# Claude Code/Cursor/Aider usage data). These cover 90%+ of first tool calls
# without needing an LLM prediction.

# Intent keywords — English base + multilingual from i18n_keywords.py
_FIX_KEYWORDS = [
    "fix",
    "bug",
    "error",
    "crash",
    "broken",
    "failing",
    "traceback",
    "exception",
    "TypeError",
    "ValueError",
    "undefined",
]
_SEARCH_KEYWORDS = [
    "find",
    "where",
    "locate",
    "search",
    "who calls",
    "imported",
    "usage of",
    "defined",
]
_TEST_KEYWORDS = ["test", "tests", "spec", "coverage", "unit test", "integration test"]

# Merge multilingual intent keywords
from smartsplit.i18n_keywords import INTENT_KEYWORDS_I18N  # noqa: E402

for _lang_kw in INTENT_KEYWORDS_I18N.values():
    _FIX_KEYWORDS.extend(_lang_kw.get("fix_debug", []))
    _SEARCH_KEYWORDS.extend(_lang_kw.get("search", []))
    _TEST_KEYWORDS.extend(_lang_kw.get("test", []))
# Deduplicate
_FIX_KEYWORDS = list(dict.fromkeys(_FIX_KEYWORDS))
_SEARCH_KEYWORDS = list(dict.fromkeys(_SEARCH_KEYWORDS))
_TEST_KEYWORDS = list(dict.fromkeys(_TEST_KEYWORDS))

_STACKTRACE_RE = re.compile(r"(File \"([^\"]+)\", line \d+|at .+\((.+\.\w+:\d+)\))", re.MULTILINE)


def _predict_from_rules(user_content: str) -> Prediction:
    """Predict tool calls from prompt patterns — no LLM needed.

    Based on research: 90%+ of first tool calls are predictable from the prompt.
    Priority: file mentions > stacktrace > search intent > explain/fix/test intent.
    """
    tools: list[AnticipatedTool] = []

    # 1. File paths mentioned → Read them (95% confidence)
    # Only match paths that look like real project files:
    # Only keep paths that look like real project files.
    # Skip bare names like "Next.js", "settings.json" that appear in documentation.
    _CODE_EXTENSIONS = {
        ".py",
        ".ts",
        ".tsx",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".rb",
        ".c",
        ".cpp",
        ".h",
        ".cs",
        ".swift",
        ".vue",
        ".svelte",
        ".lua",
        ".sh",
        ".sql",
        ".r",
    }
    raw_paths = list(dict.fromkeys(_FILE_REF_RE.findall(user_content)))
    paths = []
    for p in raw_paths:
        ext = "." + p.rsplit(".", 1)[-1] if "." in p else ""
        if "/" in p or p.startswith(".") or p in _WELL_KNOWN_FILES or ext in _CODE_EXTENSIONS:
            paths.append(p)
    for path in paths[:3]:
        tools.append(
            AnticipatedTool(
                tool="read_file",
                args={"path": path},
                reason="file mentioned in prompt",
                confidence=0.95,
            )
        )

    # 2. Stacktrace pasted → Read files from the trace (95% confidence)
    trace_files: list[str] = []
    for match in _STACKTRACE_RE.finditer(user_content):
        # Python: File "path", line N
        if match.group(2):
            trace_files.append(match.group(2))
        # JS/TS: at foo (path:line)
        elif match.group(3):
            trace_files.append(match.group(3).split(":")[0])
    for path in list(dict.fromkeys(trace_files))[:2]:
        if not any(t.args.get("path") == path for t in tools):
            tools.append(
                AnticipatedTool(
                    tool="read_file",
                    args={"path": path},
                    reason="file from stacktrace",
                    confidence=0.93,
                )
            )

    # 3. Search intent → Grep (95% confidence)
    prompt_lower = user_content.lower()
    if any(kw in prompt_lower for kw in _SEARCH_KEYWORDS) and not tools:
        tools.append(
            AnticipatedTool(
                tool="grep",
                args={},
                reason="search intent detected",
                confidence=0.85,
            )
        )

    # 4. Error intent without files → Grep for error pattern (90% confidence)
    if any(kw in prompt_lower for kw in _FIX_KEYWORDS) and not tools:
        tools.append(
            AnticipatedTool(
                tool="grep",
                args={},
                reason="error/debug intent detected",
                confidence=0.80,
            )
        )

    # 5. Test intent → Read source file then look for test files (90% confidence)
    if any(kw in prompt_lower for kw in _TEST_KEYWORDS) and paths:
        for path in paths[:1]:
            # Predict reading the test file too
            name = path.rsplit("/", 1)[-1]
            stem = name.rsplit(".", 1)[0]
            test_path = "test_" + stem + ".py" if name.endswith(".py") else ""
            if test_path and not any(t.args.get("path") == test_path for t in tools):
                tools.append(
                    AnticipatedTool(
                        tool="read_file",
                        args={"path": test_path},
                        reason="test intent: likely test file",
                        confidence=0.80,
                    )
                )

    # Cap at 3
    tools = tools[:3]

    if not tools:
        return _NULL_PREDICTION

    logger.info(
        "Rule-based prediction: %d tool(s): %s",
        len(tools),
        [(t.tool, t.args) for t in tools],
    )
    top_confidence = max(t.confidence for t in tools)
    return Prediction(should_anticipate=True, confidence=top_confidence, tools=tools)


# ── Prompts ─────────────────────────────────────────────────

_PREDICT_FROM_USER_TEMPLATE = (
    "You are a tool-call predictor for a coding assistant. Given a user message and a "
    "list of available tools, predict which READ-ONLY tools the assistant will call first.\n\n"
    "You may ONLY predict these safe tools: {safe_tools}\n\n"
    "Available tools in this session: {available_tools}\n\n"
    "Rules:\n"
    "- Predict ONLY read-only tools (file reads, searches, directory listings)\n"
    "- Do NOT predict write tools (edit, write, execute, delete, etc.)\n"
    "- Be conservative — only predict calls you are highly confident about\n"
    "- Maximum 3 predictions\n\n"
    "Respond with ONLY this JSON (no other text):\n"
    '{{"should_anticipate": true/false, "confidence": 0.0-1.0, "anticipated_tools": ['
    '{{"tool": "tool_name", "args": {{}}, "reason": "why", "confidence": 0.0-1.0}}]}}\n\n'
    "--- USER MESSAGE ---\n"
)

_PREDICT_FROM_TOOL_RESULT_TEMPLATE = (
    "You are a tool-call predictor for a coding assistant. The assistant just received "
    "a tool result. Predict which READ-ONLY tool the assistant will call next.\n\n"
    "You may ONLY predict these safe tools: {safe_tools}\n\n"
    "Rules:\n"
    "- Predict ONLY read-only tools (file reads, searches, directory listings)\n"
    "- Think about what a coding assistant would logically read next\n"
    "- Be conservative — only predict if you are highly confident\n"
    "- Maximum 3 predictions\n\n"
    "Respond with ONLY this JSON (no other text):\n"
    '{{"should_anticipate": true/false, "confidence": 0.0-1.0, "anticipated_tools": ['
    '{{"tool": "tool_name", "args": {{}}, "reason": "why", "confidence": 0.0-1.0}}]}}\n\n'
)


# ── Detector ────────────────────────────────────────────────


class IntentionDetector:
    """Predicts read-only tool calls an LLM is likely to make next.

    Combines LLM-based prediction with learned patterns from past observations.
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        pattern_learner: ToolPatternLearner | None = None,
    ) -> None:
        self._registry = registry
        self._pattern_learner = pattern_learner

    async def predict(
        self,
        messages: list[dict[str, str]],
        available_tools: list[dict[str, str]] | None,
    ) -> Prediction:
        """Main entry point — inspect the last message and predict tool calls.

        Merges LLM predictions with pattern-based suggestions from the learner.
        """
        if not messages:
            return _NULL_PREDICTION

        last = messages[-1]
        role = last.get("role", "")

        # Phase 1: Rule-based prediction — if user mentions files, predict read_file
        rule_prediction = _NULL_PREDICTION
        if role == "user":
            content = last.get("content", "")
            if content:
                rule_prediction = _predict_from_rules(content)

        # Phase 2: LLM-based prediction — skip if:
        # - Rules already have high confidence (saves free LLM quota)
        # - All free workers are in circuit breaker (avoids cascading 429s)
        llm_prediction = _NULL_PREDICTION
        if rule_prediction.confidence < _FAKE_CONFIDENCE:
            try:
                priority = self._registry._free_llm_priority
                providers = self._registry._providers
                if not isinstance(priority, list) or not isinstance(providers, dict):
                    has_healthy_worker = True
                else:
                    has_healthy_worker = any(
                        self._registry.circuit_breaker.is_healthy(name)
                        for name in priority
                        if providers.get(name) is not None
                    )
            except (AttributeError, TypeError):
                has_healthy_worker = True  # assume available if registry doesn't support check
            if has_healthy_worker:
                if role == "user":
                    content = last.get("content", "")
                    if content:
                        llm_prediction = await self._predict_from_user(content, available_tools)
                elif role == "tool":
                    llm_prediction = await self._predict_from_tool_result(messages)
            else:
                logger.debug("Skipping LLM prediction — no healthy free workers")

        # Phase 3: Pattern-based suggestions
        pattern_suggestions = []
        if self._pattern_learner:
            pattern_suggestions = self._pattern_learner.suggest_tools(messages)

        # Merge all sources: rules > patterns > LLM
        merged = self._merge_all(rule_prediction, llm_prediction, pattern_suggestions)
        return merged

    def _merge_all(
        self,
        rule_prediction: Prediction,
        llm_prediction: Prediction,
        pattern_suggestions: list[dict],
    ) -> Prediction:
        """Merge rule-based, LLM-based, and pattern-based predictions.

        Priority: rules (highest confidence, most reliable) > patterns > LLM.
        Deduplicates by tool+path to avoid reading the same file twice.
        """
        all_tools: list[AnticipatedTool] = []
        seen_keys: set[str] = set()

        # Add rule predictions first (highest priority)
        for t in rule_prediction.tools:
            key = t.tool + ":" + str(t.args.get("path", ""))
            if key not in seen_keys:
                all_tools.append(t)
                seen_keys.add(key)

        # Add LLM predictions (skip duplicates)
        for t in llm_prediction.tools:
            key = t.tool + ":" + str(t.args.get("path", ""))
            if key not in seen_keys:
                all_tools.append(t)
                seen_keys.add(key)

        # Add pattern suggestions (skip duplicates)
        for s in pattern_suggestions:
            raw_args = s.get("args", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    raw_args = {}
            if not isinstance(raw_args, dict):
                raw_args = {}
            key = s["tool"] + ":" + str(raw_args.get("path", ""))
            if key not in seen_keys:
                all_tools.append(
                    AnticipatedTool(
                        tool=s["tool"],
                        args=raw_args,
                        reason="pattern:" + s.get("source", "learned"),
                        confidence=s.get("confidence", 0.7),
                    )
                )
                seen_keys.add(key)

        # Sort by confidence, cap at 3
        all_tools.sort(key=lambda t: t.confidence, reverse=True)
        all_tools = all_tools[:3]

        if not all_tools:
            return _NULL_PREDICTION

        top_confidence = max(t.confidence for t in all_tools)
        return Prediction(should_anticipate=True, confidence=top_confidence, tools=all_tools)

    async def _predict_from_user(
        self,
        user_content: str,
        available_tools: list[dict[str, str]] | None,
    ) -> Prediction:
        """Predict which read tools the LLM will call given a user message."""
        tool_names = _extract_tool_names(available_tools)
        safe_tool_names = sorted(SAFE_TOOLS & set(tool_names)) if tool_names else sorted(SAFE_TOOLS)

        if not safe_tool_names:
            return _NULL_PREDICTION

        safe_str = ", ".join(sorted(SAFE_TOOLS))
        avail_str = ", ".join(tool_names) if tool_names else "(not specified)"
        prompt = (
            _PREDICT_FROM_USER_TEMPLATE.replace("{safe_tools}", safe_str).replace("{available_tools}", avail_str)
            + user_content[:2000]
        )

        return await self._call_and_parse(prompt)

    async def _predict_from_tool_result(self, messages: list[dict[str, str]]) -> Prediction:
        """Predict the next read tool given a tool result and preceding call."""
        tool_name, tool_args, tool_result = _extract_last_tool_exchange(messages)
        if not tool_name:
            return _NULL_PREDICTION

        safe_str = ", ".join(sorted(SAFE_TOOLS))
        prompt = (
            _PREDICT_FROM_TOOL_RESULT_TEMPLATE.replace("{safe_tools}", safe_str)
            + "The assistant called: "
            + tool_name
            + "\n"
            + "With arguments: "
            + json.dumps(tool_args)[:500]
            + "\n"
            + "And got this result (truncated): "
            + tool_result[:1000]
        )

        return await self._call_and_parse(prompt)

    async def _call_and_parse(self, prompt: str) -> Prediction:
        """Send prompt to free LLM, parse JSON response into Prediction."""
        try:
            raw = await self._registry.call_free_llm(prompt, prefer="cerebras")
            cleaned = _extract_json(raw)
            data = json.loads(cleaned)

            should_anticipate = bool(data.get("should_anticipate", False))
            confidence = float(data.get("confidence", 0.0))

            if not should_anticipate or confidence < _MIN_CONFIDENCE:
                return _NULL_PREDICTION

            anticipated = data.get("anticipated_tools", [])
            if not isinstance(anticipated, list):
                return _NULL_PREDICTION

            tools: list[AnticipatedTool] = []
            for entry in anticipated:
                if not isinstance(entry, dict):
                    continue
                tool = str(entry.get("tool", ""))
                if tool not in SAFE_TOOLS:
                    logger.debug("Filtered out non-safe tool prediction: %s", tool)
                    continue
                tool_confidence = float(entry.get("confidence", 0.0))
                if tool_confidence < _MIN_CONFIDENCE:
                    continue
                tools.append(
                    AnticipatedTool(
                        tool=tool,
                        args=entry.get("args", {}),
                        reason=str(entry.get("reason", "")),
                        confidence=tool_confidence,
                    )
                )

            if not tools:
                return _NULL_PREDICTION

            logger.info(
                "Predicted %d tool call(s): %s (confidence=%.2f)",
                len(tools),
                [t.tool for t in tools],
                confidence,
            )
            return Prediction(should_anticipate=True, confidence=confidence, tools=tools)

        except (SmartSplitError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Intention prediction failed: %s: %s", type(exc).__name__, exc)
            return _NULL_PREDICTION
        except Exception as exc:
            # Never crash — prediction is best-effort.
            logger.error("Unexpected error in intention prediction: %s: %s", type(exc).__name__, exc)
            return _NULL_PREDICTION


# ── Helpers ─────────────────────────────────────────────────


def _extract_tool_names(available_tools: list[dict[str, str]] | None) -> list[str]:
    """Extract tool names from the OpenAI-format tools list."""
    if not available_tools:
        return []
    names: list[str] = []
    for tool in available_tools:
        if isinstance(tool, dict):
            # OpenAI format: {"type": "function", "function": {"name": "...", ...}}
            func = tool.get("function", {})
            if isinstance(func, dict):
                name = func.get("name", "")
                if name:
                    names.append(name)
    return names


def _extract_last_tool_exchange(
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, str], str]:
    """Find the last tool result and its preceding assistant tool_call.

    Returns (tool_name, tool_args, tool_result). Returns empty strings
    on failure — caller checks ``tool_name`` truthiness.
    """
    if len(messages) < 2:
        return "", {}, ""

    last = messages[-1]
    tool_result = str(last.get("content", ""))

    # Walk backwards to find the assistant message with the tool_call
    for msg in reversed(messages[:-1]):
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue
        # Use the last tool_call from the assistant message
        call = tool_calls[-1]
        if isinstance(call, dict):
            func = call.get("function", {})
            if isinstance(func, dict):
                tool_name = str(func.get("name", ""))
                args_raw = func.get("arguments", "{}")
                try:
                    tool_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}
                return tool_name, tool_args, tool_result
        break

    return "", {}, ""
