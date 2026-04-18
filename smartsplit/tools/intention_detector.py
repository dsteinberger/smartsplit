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
from smartsplit.json_utils import extract_json
from smartsplit.tools.registry import FILE_REF_RE as _FILE_REF_RE
from smartsplit.tools.registry import SAFE_TOOLS
from smartsplit.tools.registry import WELL_KNOWN_FILES as _WELL_KNOWN_FILES
from smartsplit.triage.i18n_keywords import INTENT_KEYWORDS_I18N

if TYPE_CHECKING:
    from smartsplit.providers.registry import ProviderRegistry
    from smartsplit.tools.pattern_learner import ToolPatternLearner

logger = logging.getLogger("smartsplit.intention_detector")

_MIN_CONFIDENCE = 0.7  # conservative: 10% irrelevant content → -23% quality (context rot research)
FAKE_TOOL_CONFIDENCE = 0.85  # threshold for FAKE tool_use — shared with proxy pipeline

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
_FIX_KEYWORDS_BASE = [
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
_SEARCH_KEYWORDS_BASE = [
    "find",
    "where",
    "locate",
    "search",
    "who calls",
    "imported",
    "usage of",
    "defined",
]
_TEST_KEYWORDS_BASE = ["test", "tests", "spec", "coverage", "unit test", "integration test"]


def _merge_intent_keywords(base: list[str], key: str) -> list[str]:
    """Merge per-language intent keywords from INTENT_KEYWORDS_I18N under a named bucket."""
    merged = list(base)
    for lang_kw in INTENT_KEYWORDS_I18N.values():
        merged.extend(lang_kw.get(key, []))
    return list(dict.fromkeys(merged))


_FIX_KEYWORDS = _merge_intent_keywords(_FIX_KEYWORDS_BASE, "fix_debug")
_SEARCH_KEYWORDS = _merge_intent_keywords(_SEARCH_KEYWORDS_BASE, "search")
_TEST_KEYWORDS = _merge_intent_keywords(_TEST_KEYWORDS_BASE, "test")

_STACKTRACE_RE = re.compile(r"(File \"([^\"]+)\", line \d+|at .+\((.+\.\w+:\d+)\))", re.MULTILINE)


_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
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
)


def _looks_like_project_path(path: str) -> bool:
    """Return True if ``path`` looks like a real project file (not doc jargon)."""
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
    return "/" in path or path.startswith(".") or path in _WELL_KNOWN_FILES or ext in _CODE_EXTENSIONS


def _pick_read_tool(tool_names: set[str]) -> str:
    """Return the read-file tool name the client exposes, or the canonical handler."""
    return "Read" if "Read" in tool_names else "read_file"


def _pick_grep_tool(tool_names: set[str]) -> str:
    """Return the grep tool name the client exposes, or the canonical handler."""
    return "Grep" if "Grep" in tool_names else "grep"


def _predict_file_mentions(user_content: str, tool_names: set[str]) -> tuple[list[AnticipatedTool], list[str]]:
    """Rule 1: file paths referenced in the prompt → read them."""
    raw = list(dict.fromkeys(_FILE_REF_RE.findall(user_content)))
    paths = [p for p in raw if _looks_like_project_path(p)]
    read_tool = _pick_read_tool(tool_names)
    tools = [
        AnticipatedTool(tool=read_tool, args={"path": path}, reason="file mentioned in prompt", confidence=0.95)
        for path in paths[:3]
    ]
    return tools, paths


def _predict_stacktrace(
    user_content: str, existing: list[AnticipatedTool], tool_names: set[str]
) -> list[AnticipatedTool]:
    """Rule 2: stacktrace pasted → read files from the trace (skip dups)."""
    trace_files: list[str] = []
    for match in _STACKTRACE_RE.finditer(user_content):
        if match.group(2):  # Python: File "path", line N
            trace_files.append(match.group(2))
        elif match.group(3):  # JS/TS: at foo (path:line)
            trace_files.append(match.group(3).split(":")[0])
    known = {t.args.get("path") for t in existing}
    read_tool = _pick_read_tool(tool_names)
    return [
        AnticipatedTool(tool=read_tool, args={"path": path}, reason="file from stacktrace", confidence=0.93)
        for path in list(dict.fromkeys(trace_files))[:2]
        if path not in known
    ]


def _predict_grep_intent(prompt_lower: str, tool_names: set[str]) -> AnticipatedTool | None:
    """Rules 3 & 4: search or error intent without known paths → grep."""
    grep_tool = _pick_grep_tool(tool_names)
    if any(kw in prompt_lower for kw in _SEARCH_KEYWORDS):
        return AnticipatedTool(tool=grep_tool, args={}, reason="search intent detected", confidence=0.85)
    if any(kw in prompt_lower for kw in _FIX_KEYWORDS):
        return AnticipatedTool(tool=grep_tool, args={}, reason="error/debug intent detected", confidence=0.80)
    return None


def _predict_test_files(
    prompt_lower: str, paths: list[str], existing: list[AnticipatedTool], tool_names: set[str]
) -> list[AnticipatedTool]:
    """Rule 5: test intent → also read the likely test file for mentioned .py sources."""
    if not (any(kw in prompt_lower for kw in _TEST_KEYWORDS) and paths):
        return []
    known = {t.args.get("path") for t in existing}
    read_tool = _pick_read_tool(tool_names)
    out: list[AnticipatedTool] = []
    for path in paths[:1]:
        name = path.rsplit("/", 1)[-1]
        if not name.endswith(".py"):
            continue
        stem = name.rsplit(".", 1)[0]
        test_path = "test_" + stem + ".py"
        if test_path not in known:
            out.append(
                AnticipatedTool(
                    tool=read_tool,
                    args={"path": test_path},
                    reason="test intent: likely test file",
                    confidence=0.80,
                )
            )
    return out


def _predict_from_rules(user_content: str, tool_names: set[str]) -> Prediction:
    """Predict tool calls from prompt patterns — no LLM needed.

    Based on research: 90%+ of first tool calls are predictable from the prompt.
    Priority: file mentions > stacktrace > search intent > explain/fix/test intent.
    Tool names emitted match what the client exposes (e.g. ``Read`` vs ``read_file``).
    """
    tools, paths = _predict_file_mentions(user_content, tool_names)
    tools.extend(_predict_stacktrace(user_content, tools, tool_names))

    prompt_lower = user_content.lower()
    if not tools:
        grep = _predict_grep_intent(prompt_lower, tool_names)
        if grep is not None:
            tools.append(grep)

    tools.extend(_predict_test_files(prompt_lower, paths, tools, tool_names))

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

        # Phase 1: Rule-based prediction — if user mentions files, predict read_file.
        # We pass the client's tool names so the rules emit the exact name the
        # client exposes (e.g. ``Read`` vs canonical ``read_file``). Otherwise the
        # FAKE tool_use would carry a name the agent doesn't recognise.
        tool_names = set(_extract_tool_names(available_tools))
        rule_prediction = _NULL_PREDICTION
        if role == "user":
            content = last.get("content", "")
            if content:
                rule_prediction = _predict_from_rules(content, tool_names)

        # Phase 2: LLM-based prediction — skip if:
        # - Rules already have high confidence (saves free LLM quota)
        # - All free workers are in circuit breaker (avoids cascading 429s)
        llm_prediction = _NULL_PREDICTION
        if rule_prediction.confidence < FAKE_TOOL_CONFIDENCE:
            try:
                priority = self._registry.free_llm_priority
                if not isinstance(priority, list):
                    has_healthy_worker = True  # mock registry — assume available
                else:
                    has_healthy_worker = any(
                        self._registry.circuit_breaker.is_healthy(name)
                        for name in priority
                        if self._registry.get(name) is not None
                    )
            except (AttributeError, TypeError):
                has_healthy_worker = True  # tolerate registries without this API
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Merged prediction: %s",
                [(t.tool, t.args, f"{t.confidence:.2f}", t.reason) for t in all_tools],
            )
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

        safe_str = ", ".join(safe_tool_names)
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
            cleaned = extract_json(raw)
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
