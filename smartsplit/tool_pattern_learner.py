"""Tool pattern learner — observes LLM tool calls and builds patterns for prediction.

Watches what tools the LLM actually calls, compares against predictions, and learns
patterns that improve future suggestions. Patterns include:

- File mention patterns: user mentions .py → LLM reads a .py file
- Sequential patterns: after tool A, LLM calls tool B
- Result content patterns: error in output → LLM reads specific files
- Project first-read patterns: files the LLM reads first in a project
- Per-tool accuracy tracking: correct/wrong/missed rates

Scoring uses Wilson score lower bound for reliable confidence intervals.
Staleness decay removes old patterns and halves stale ones on load.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from smartsplit.tool_registry import FILE_REF_RE as _FILE_PATH_RE
from smartsplit.tool_registry import WELL_KNOWN_FILES as _WELL_KNOWN_FILES

logger = logging.getLogger("smartsplit.tool_pattern_learner")

# ── Constants ───────────────────────────────────────────────

_HALF_LIFE = 7 * 86400  # 7 days — halve stale pattern hits after this
_EVICT_AGE = 30 * 86400  # 30 days — remove patterns older than this
_MAX_SEQUENTIAL = 200  # max sequential pattern entries
_MAX_CONTENT = 50  # max result-content pattern entries
_MIN_OBSERVATIONS = 3  # minimum observations before scoring
_SAVE_INTERVAL = 60  # persist every 60 seconds
_MIN_SUGGEST_CONFIDENCE = 0.65  # default minimum confidence for suggestions

_ERROR_KEYWORDS: tuple[str, ...] = (
    "ImportError",
    "ModuleNotFoundError",
    "FileNotFoundError",
    "SyntaxError",
    "TypeError",
    "ValueError",
    "KeyError",
    "AttributeError",
    "NameError",
    "IndexError",
    "FAILED",
    "Traceback",
    "Permission denied",
    "No such file",
    "command not found",
)


# ── Data classes ────────────────────────────────────────────


@dataclass(frozen=True)
class PendingPrediction:
    """A prediction awaiting outcome observation."""

    request_id: str
    timestamp: float
    predicted_tools: list[dict[str, str]]
    context_signals: dict[str, object]


# ── Helpers (module-level) ──────────────────────────────────


def _extract_file_paths(text: str) -> list[str]:
    """Extract file paths from text using regex."""
    if not text:
        return []
    try:
        matches = _FILE_PATH_RE.findall(text)
        # Deduplicate preserving order
        seen: set[str] = set()
        result: list[str] = []
        for m in matches:
            if m not in seen and not m.startswith("http"):
                seen.add(m)
                result.append(m)
        return result
    except (TypeError, re.error):
        return []


def _extract_error_keywords(text: str) -> list[str]:
    """Extract known error keywords from text."""
    if not text:
        return []
    found: list[str] = []
    for kw in _ERROR_KEYWORDS:
        if kw in text:
            found.append(kw)
    return found


def _abstract_tool_call(tool: str, args: dict[str, str]) -> str:
    """Abstract a tool call to a generalised pattern key.

    read_file(auth.py) → read_file:*.py
    read_file(requirements.txt) → read_file:requirements.txt  (well-known)
    grep(*) → grep:*
    """
    if tool in ("read_file", "Read", "cat", "head", "tail"):
        file_arg = args.get("file_path") or args.get("path") or args.get("file", "")
        if file_arg:
            basename = Path(file_arg).name
            if basename in _WELL_KNOWN_FILES:
                return tool + ":" + basename
            ext = Path(basename).suffix
            if ext:
                return tool + ":*" + ext
        return tool + ":*"
    return tool + ":*"


def _normalize_tool_call(tc: dict[str, str]) -> str:
    """Normalize a tool call dict to a comparison string like 'read_file:auth.py'."""
    tool = tc.get("tool") or tc.get("name", "")
    args = tc.get("args", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {}
    if not isinstance(args, dict):
        args = {}

    if tool in ("read_file", "Read", "cat", "head", "tail"):
        file_arg = args.get("file_path") or args.get("path") or args.get("file", "")
        if file_arg:
            return tool + ":" + Path(file_arg).name
    return tool + ":*"


def _extract_context_signals(messages: list[dict[str, str]]) -> dict[str, object]:
    """Extract context signals from a message history."""
    signals: dict[str, object] = {
        "mentioned_files": [],
        "mentioned_extensions": [],
        "last_tool": "",
        "last_tool_args": {},
        "result_has_error": False,
        "result_error_keywords": [],
        "is_first_turn": False,
    }

    if not messages:
        return signals

    # Count user messages to determine first turn
    user_count = sum(1 for m in messages if m.get("role") == "user")
    signals["is_first_turn"] = user_count <= 1

    # Find last user message for file mentions
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = str(msg.get("content", ""))
            files = _extract_file_paths(content)
            signals["mentioned_files"] = files
            exts: list[str] = []
            seen_exts: set[str] = set()
            for f in files:
                ext = Path(f).suffix
                if ext and ext not in seen_exts:
                    seen_exts.add(ext)
                    exts.append(ext)
            signals["mentioned_extensions"] = exts
            break

    # Find last tool call and its result
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "tool":
            content = str(msg.get("content", ""))
            error_kws = _extract_error_keywords(content)
            signals["result_has_error"] = len(error_kws) > 0
            signals["result_error_keywords"] = error_kws
            # Look back for the assistant tool_call
            for j in range(i - 1, -1, -1):
                prev = messages[j]
                if prev.get("role") != "assistant":
                    continue
                tool_calls = prev.get("tool_calls")
                if not isinstance(tool_calls, list) or not tool_calls:
                    continue
                call = tool_calls[-1]
                if isinstance(call, dict):
                    func = call.get("function", {})
                    if isinstance(func, dict):
                        signals["last_tool"] = str(func.get("name", ""))
                        raw_args = func.get("arguments", "{}")
                        try:
                            parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                            signals["last_tool_args"] = parsed if isinstance(parsed, dict) else {}
                        except (json.JSONDecodeError, TypeError):
                            signals["last_tool_args"] = {}
                break
            break

    return signals


# ── Typed accessors ────────────────────────────────────────


def _as_dict_of_dicts(data: dict[str, object], key: str) -> dict[str, dict[str, object]]:
    """Retrieve a nested dict from the data store, defaulting to empty."""
    val = data.get(key, {})
    return val if isinstance(val, dict) else {}


def _as_str_list(signals: dict[str, object], key: str) -> list[str]:
    """Retrieve a list of strings from a signals dict, defaulting to empty."""
    val = signals.get(key, [])
    return val if isinstance(val, list) else []


def _as_str_dict(signals: dict[str, object], key: str) -> dict[str, str]:
    """Retrieve a dict of strings from a signals dict, defaulting to empty."""
    val = signals.get(key, {})
    return val if isinstance(val, dict) else {}


def _as_int_dict(data: dict[str, object], key: str) -> dict[str, dict[str, int]]:
    """Retrieve a dict of int-valued dicts from the data store."""
    val = data.get(key, {})
    return val if isinstance(val, dict) else {}


# ── Main class ──────────────────────────────────────────────


class ToolPatternLearner:
    """Learns tool-call patterns from observations and suggests future tools.

    Thread-safe: all reads and mutations are protected by a lock.
    """

    def __init__(self, persistence_path: str | None = None, project_dir: str = ".") -> None:
        self._lock = threading.Lock()
        self._dirty = False
        self._last_save: float = 0.0
        self._project_dir = project_dir
        self._project_hash = hashlib.sha256(project_dir.encode()).hexdigest()[:8]
        self._pending: dict[str, PendingPrediction] = {}
        self._path = Path(persistence_path or (Path.home() / ".smartsplit" / "tool_patterns.json"))

        # Pattern stores
        self._data: dict[str, object] = {
            "version": 1,
            "total_observations": 0,
            "file_mention_patterns": {},
            "sequential_patterns": {},
            "result_content_patterns": {},
            "project_first_read": {},
            "tool_accuracy": {},
        }

        self._load()

    def record_prediction(
        self,
        request_id: str,
        predicted_tools: list[dict[str, str]],
        context_signals: dict[str, object],
    ) -> None:
        """Store a prediction for later comparison with actual outcome."""
        with self._lock:
            self._pending[request_id] = PendingPrediction(
                request_id=request_id,
                timestamp=time.time(),
                predicted_tools=predicted_tools,
                context_signals=context_signals,
            )
            # Keep pending map bounded — evict oldest if too large
            if len(self._pending) > 100:
                oldest_key = min(self._pending, key=lambda k: self._pending[k].timestamp)
                del self._pending[oldest_key]

    def observe_outcome(
        self,
        actual_tools: list[dict[str, str]],
        messages: list[dict[str, str]],
    ) -> None:
        """Compare pending prediction vs actual tools called; learn patterns."""
        with self._lock:
            try:
                signals = _extract_context_signals(messages)
                self._learn_patterns(actual_tools, signals)
                self._update_accuracy(actual_tools)
                data = self._data
                total = int(data.get("total_observations", 0))
                data["total_observations"] = total + 1
                self._dirty = True

                now = time.time()
                if now - self._last_save >= _SAVE_INTERVAL:
                    self._save()
                    self._last_save = now
            except Exception as exc:
                logger.warning("Error in observe_outcome: %s: %s", type(exc).__name__, exc)

    def suggest_tools(
        self,
        messages: list[dict[str, str]],
        min_confidence: float = _MIN_SUGGEST_CONFIDENCE,
    ) -> list[dict[str, str | float]]:
        """Return tool suggestions based on learned patterns."""
        with self._lock:
            try:
                return self._suggest_tools_locked(messages, min_confidence)
            except Exception as exc:
                logger.warning("Error in suggest_tools: %s: %s", type(exc).__name__, exc)
                return []

    def get_stats(self) -> dict[str, object]:
        """Return a deep copy of pattern data for observability."""
        with self._lock:
            return copy.deepcopy(self._data)

    def flush(self) -> None:
        """Persist data if dirty."""
        with self._lock:
            if self._dirty:
                self._save()

    # ── Internal: pattern learning ──────────────────────────

    def _learn_patterns(
        self,
        actual_tools: list[dict[str, str]],
        signals: dict[str, object],
    ) -> None:
        """Extract patterns from observed tool calls. Caller must hold self._lock."""
        now = time.time()
        file_patterns = _as_dict_of_dicts(self._data, "file_mention_patterns")
        seq_patterns = _as_dict_of_dicts(self._data, "sequential_patterns")
        content_patterns = _as_dict_of_dicts(self._data, "result_content_patterns")
        proj_first = _as_dict_of_dicts(self._data, "project_first_read")

        mentioned_exts = _as_str_list(signals, "mentioned_extensions")
        is_first: bool = bool(signals.get("is_first_turn", False))
        error_kws = _as_str_list(signals, "result_error_keywords")

        # Pattern 1: file mention → tool read
        for tc in actual_tools:
            abstract = _abstract_tool_call(tc.get("tool", tc.get("name", "")), tc.get("args", {}))
            # Check if the tool reads a file with an extension the user mentioned
            for ext in mentioned_exts:
                if "*" + ext in abstract:
                    key = ext + "→" + abstract
                    entry = file_patterns.get(key, {"hits": 0, "misses": 0, "last_seen": now})
                    entry["hits"] = int(entry.get("hits", 0)) + 1
                    entry["last_seen"] = now
                    file_patterns[key] = entry

        # Pattern 2: sequential tool calls (A → B)
        for i in range(1, len(actual_tools)):
            prev_abstract = _abstract_tool_call(
                actual_tools[i - 1].get("tool", actual_tools[i - 1].get("name", "")),
                actual_tools[i - 1].get("args", {}),
            )
            curr_abstract = _abstract_tool_call(
                actual_tools[i].get("tool", actual_tools[i].get("name", "")),
                actual_tools[i].get("args", {}),
            )
            key = prev_abstract + "→" + curr_abstract
            entry = seq_patterns.get(key, {"hits": 0, "misses": 0, "last_seen": now})
            entry["hits"] = int(entry.get("hits", 0)) + 1
            entry["last_seen"] = now
            seq_patterns[key] = entry

        # Trim sequential patterns to max
        if len(seq_patterns) > _MAX_SEQUENTIAL:
            sorted_keys = sorted(seq_patterns, key=lambda k: seq_patterns[k].get("last_seen", 0))
            for k in sorted_keys[: len(seq_patterns) - _MAX_SEQUENTIAL]:
                del seq_patterns[k]

        # Pattern 3: error keywords in result → next tool
        if error_kws and actual_tools:
            first_tool = _abstract_tool_call(
                actual_tools[0].get("tool", actual_tools[0].get("name", "")),
                actual_tools[0].get("args", {}),
            )
            for kw in error_kws:
                key = kw + "→" + first_tool
                entry = content_patterns.get(key, {"hits": 0, "misses": 0, "last_seen": now})
                entry["hits"] = int(entry.get("hits", 0)) + 1
                entry["last_seen"] = now
                content_patterns[key] = entry

        # Trim content patterns to max
        if len(content_patterns) > _MAX_CONTENT:
            sorted_keys = sorted(content_patterns, key=lambda k: content_patterns[k].get("last_seen", 0))
            for k in sorted_keys[: len(content_patterns) - _MAX_CONTENT]:
                del content_patterns[k]

        # Pattern 4: project first-read files
        if is_first and actual_tools:
            proj_entry = _as_dict_of_dicts(proj_first, self._project_hash)
            for tc in actual_tools[:3]:  # only first 3 tools
                abstract = _abstract_tool_call(tc.get("tool", tc.get("name", "")), tc.get("args", {}))
                normalized = _normalize_tool_call(tc)
                file_key = normalized or abstract
                sub_val = proj_entry.get(file_key)
                sub: dict[str, object] = (
                    sub_val if isinstance(sub_val, dict) else {"hits": 0, "misses": 0, "last_seen": now}
                )
                sub["hits"] = int(sub.get("hits", 0)) + 1
                sub["last_seen"] = now
                proj_entry[file_key] = sub
            proj_first[self._project_hash] = proj_entry

        self._data["file_mention_patterns"] = file_patterns
        self._data["sequential_patterns"] = seq_patterns
        self._data["result_content_patterns"] = content_patterns
        self._data["project_first_read"] = proj_first

    def _update_accuracy(self, actual_tools: list[dict[str, str]]) -> None:
        """Update per-tool accuracy stats. Caller must hold self._lock."""
        accuracy = _as_int_dict(self._data, "tool_accuracy")

        # Find matching pending prediction (most recent)
        prediction: PendingPrediction | None = None
        if self._pending:
            # Use most recent pending prediction
            latest_key = max(self._pending, key=lambda k: self._pending[k].timestamp)
            prediction = self._pending.pop(latest_key, None)

        actual_set: set[str] = set()
        for tc in actual_tools:
            norm = _normalize_tool_call(tc)
            actual_set.add(norm)
            tool_name = tc.get("tool", tc.get("name", ""))
            entry = accuracy.get(tool_name, {"correct": 0, "wrong": 0, "missed": 0})
            accuracy[tool_name] = entry

        if prediction is not None:
            predicted_set: set[str] = set()
            for pt in prediction.predicted_tools:
                predicted_set.add(_normalize_tool_call(pt))

            # Correct: predicted and actually called
            for p in predicted_set & actual_set:
                tool_name = p.split(":")[0] if ":" in p else p
                entry = accuracy.get(tool_name, {"correct": 0, "wrong": 0, "missed": 0})
                entry["correct"] = int(entry.get("correct", 0)) + 1
                accuracy[tool_name] = entry

            # Wrong: predicted but not called
            for p in predicted_set - actual_set:
                tool_name = p.split(":")[0] if ":" in p else p
                entry = accuracy.get(tool_name, {"correct": 0, "wrong": 0, "missed": 0})
                entry["wrong"] = int(entry.get("wrong", 0)) + 1
                accuracy[tool_name] = entry

            # Missed: called but not predicted
            for a in actual_set - predicted_set:
                tool_name = a.split(":")[0] if ":" in a else a
                entry = accuracy.get(tool_name, {"correct": 0, "wrong": 0, "missed": 0})
                entry["missed"] = int(entry.get("missed", 0)) + 1
                accuracy[tool_name] = entry

        self._data["tool_accuracy"] = accuracy

    # ── Internal: suggesting ────────────────────────────────

    def _suggest_tools_locked(
        self,
        messages: list[dict[str, str]],
        min_confidence: float,
    ) -> list[dict[str, str | float]]:
        """Compute suggestions from all pattern types. Caller must hold self._lock."""
        signals = _extract_context_signals(messages)
        suggestions: list[dict[str, str | float]] = []

        mentioned_files = _as_str_list(signals, "mentioned_files")
        mentioned_exts = _as_str_list(signals, "mentioned_extensions")
        last_tool: str = str(signals.get("last_tool", ""))
        last_tool_args = _as_str_dict(signals, "last_tool_args")
        is_first: bool = bool(signals.get("is_first_turn", False))
        error_kws = _as_str_list(signals, "result_error_keywords")

        file_patterns = _as_dict_of_dicts(self._data, "file_mention_patterns")
        seq_patterns = _as_dict_of_dicts(self._data, "sequential_patterns")
        content_patterns = _as_dict_of_dicts(self._data, "result_content_patterns")
        proj_first = _as_dict_of_dicts(self._data, "project_first_read")

        # File mention patterns
        for ext in mentioned_exts:
            for key, entry in file_patterns.items():
                if not key.startswith(ext + "→"):
                    continue
                score = self._pattern_score(int(entry.get("hits", 0)), int(entry.get("misses", 0)))
                if score < min_confidence:
                    continue
                # Concretize: find actual files with this extension from context
                for fpath in mentioned_files:
                    if fpath.endswith(ext):
                        tool_name = key.split("→")[1].split(":")[0] if "→" in key else "read_file"
                        suggestions.append(
                            {
                                "tool": tool_name,
                                "args": json.dumps({"file_path": fpath}),
                                "confidence": score,
                                "source": "file_mention",
                            }
                        )

        # Sequential patterns
        if last_tool:
            last_abstract = _abstract_tool_call(last_tool, last_tool_args)
            for key, entry in seq_patterns.items():
                if not key.startswith(last_abstract + "→"):
                    continue
                score = self._pattern_score(int(entry.get("hits", 0)), int(entry.get("misses", 0)))
                if score < min_confidence:
                    continue
                next_part = key.split("→")[1] if "→" in key else ""
                if not next_part:
                    continue
                tool_name = next_part.split(":")[0]
                # Concretize abstract patterns back to specific files from context
                target = next_part.split(":")[1] if ":" in next_part else "*"
                args_dict: dict[str, str] = {}
                if target.startswith("*.") and mentioned_files:
                    target_ext = target[1:]  # e.g. ".py"
                    for fpath in mentioned_files:
                        if fpath.endswith(target_ext):
                            args_dict = {"file_path": fpath}
                            break
                suggestions.append(
                    {
                        "tool": tool_name,
                        "args": json.dumps(args_dict),
                        "confidence": score,
                        "source": "sequential",
                    }
                )

        # Project first-read patterns
        if is_first:
            proj_entry = _as_dict_of_dicts(proj_first, self._project_hash)
            for file_key, entry in proj_entry.items():
                if not isinstance(entry, dict):
                    continue
                score = self._pattern_score(int(entry.get("hits", 0)), int(entry.get("misses", 0)))
                if score < min_confidence:
                    continue
                tool_name = file_key.split(":")[0] if ":" in file_key else "read_file"
                target = file_key.split(":")[1] if ":" in file_key else ""
                args_dict = {"file_path": target} if target and target != "*" else {}
                suggestions.append(
                    {
                        "tool": tool_name,
                        "args": json.dumps(args_dict),
                        "confidence": score,
                        "source": "project_first_read",
                    }
                )

        # Result content patterns
        for kw in error_kws:
            for key, entry in content_patterns.items():
                if not key.startswith(kw + "→"):
                    continue
                score = self._pattern_score(int(entry.get("hits", 0)), int(entry.get("misses", 0)))
                if score < min_confidence:
                    continue
                next_part = key.split("→")[1] if "→" in key else ""
                if not next_part:
                    continue
                tool_name = next_part.split(":")[0]
                target = next_part.split(":")[1] if ":" in next_part else "*"
                args_dict = {}
                if target.startswith("*.") and mentioned_files:
                    target_ext = target[1:]
                    for fpath in mentioned_files:
                        if fpath.endswith(target_ext):
                            args_dict = {"file_path": fpath}
                            break
                suggestions.append(
                    {
                        "tool": tool_name,
                        "args": json.dumps(args_dict),
                        "confidence": score,
                        "source": "result_content",
                    }
                )

        # Deduplicate by (tool, args), keeping highest confidence
        seen: dict[str, dict[str, str | float]] = {}
        for s in suggestions:
            dedup_key = str(s.get("tool", "")) + "|" + str(s.get("args", ""))
            existing = seen.get(dedup_key)
            if existing is None or float(s.get("confidence", 0)) > float(existing.get("confidence", 0)):
                seen[dedup_key] = s

        # Sort by confidence descending, limit to 3
        result = sorted(seen.values(), key=lambda s: float(s.get("confidence", 0)), reverse=True)
        return result[:3]

    # ── Scoring ─────────────────────────────────────────────

    @staticmethod
    def _pattern_score(hits: int, misses: int) -> float:
        """Wilson score lower bound — conservative confidence interval."""
        n = hits + misses
        if n < _MIN_OBSERVATIONS:
            return 0.0
        z = 1.96
        p = hits / n
        denom = 1 + z * z / n
        center = p + z * z / (2 * n)
        spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
        return (center - spread) / denom

    # ── Persistence ─────────────────────────────────────────

    def _save(self) -> None:
        """Persist data to disk. Caller must hold self._lock."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._path.write_text(json.dumps(self._data, indent=2))
            self._path.chmod(0o600)
            self._dirty = False
        except OSError as e:
            logger.warning("Could not save tool patterns: %s", e)

    def _load(self) -> None:
        """Load persisted data and apply staleness decay."""
        try:
            if not self._path.exists():
                return
            raw = json.loads(self._path.read_text())
            if not isinstance(raw, dict):
                raise ValueError("Root is not a dict")
            if raw.get("version") != 1:
                logger.info("Unknown tool_patterns version, starting fresh")
                return

            # Validate and accept known keys
            for key in (
                "file_mention_patterns",
                "sequential_patterns",
                "result_content_patterns",
                "project_first_read",
                "tool_accuracy",
            ):
                val = raw.get(key)
                if val is not None and not isinstance(val, dict):
                    raise ValueError("Invalid structure for '" + key + "'")

            self._data = raw
            self._data.setdefault("total_observations", 0)
            self._apply_decay()
            total = int(self._data.get("total_observations", 0))
            logger.info("Loaded tool patterns: %d observations", total)

        except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("Could not load tool patterns, starting fresh: %s", e)
            self._data = {
                "version": 1,
                "total_observations": 0,
                "file_mention_patterns": {},
                "sequential_patterns": {},
                "result_content_patterns": {},
                "project_first_read": {},
                "tool_accuracy": {},
            }

    def _apply_decay(self) -> None:
        """Remove evicted patterns and halve stale ones. Called on load only."""
        now = time.time()
        pattern_keys = (
            "file_mention_patterns",
            "sequential_patterns",
            "result_content_patterns",
        )

        for pkey in pattern_keys:
            patterns = _as_dict_of_dicts(self._data, pkey)
            if not isinstance(patterns, dict):
                continue
            to_delete: list[str] = []
            for key, entry in patterns.items():
                if not isinstance(entry, dict):
                    to_delete.append(key)
                    continue
                last_seen = float(entry.get("last_seen", 0))
                age = now - last_seen
                if age > _EVICT_AGE:
                    to_delete.append(key)
                elif age > _HALF_LIFE:
                    entry["hits"] = max(1, int(entry.get("hits", 0)) // 2)
            for key in to_delete:
                del patterns[key]

        # Also decay project_first_read entries
        proj_first = _as_dict_of_dicts(self._data, "project_first_read")
        if isinstance(proj_first, dict):
            for proj_hash, entries in list(proj_first.items()):
                if not isinstance(entries, dict):
                    del proj_first[proj_hash]
                    continue
                to_delete = []
                for key, entry in entries.items():
                    if not isinstance(entry, dict):
                        to_delete.append(key)
                        continue
                    last_seen = float(entry.get("last_seen", 0))
                    age = now - last_seen
                    if age > _EVICT_AGE:
                        to_delete.append(key)
                    elif age > _HALF_LIFE:
                        entry["hits"] = max(1, int(entry.get("hits", 0)) // 2)
                for key in to_delete:
                    del entries[key]
