"""Tests for SmartSplit tool pattern learner."""

from __future__ import annotations

import json
import time
from pathlib import Path

from smartsplit.tools.pattern_learner import (
    ToolPatternLearner,
    _abstract_tool_call,
    _extract_error_keywords,
    _extract_file_paths,
    _normalize_tool_call,
    extract_context_signals,
)

# ── _extract_file_paths regex ────────────────────────────────


class TestExtractFilePaths:
    def test_python_file(self):
        paths = _extract_file_paths("Look at smartsplit/proxy.py for the issue")
        assert "smartsplit/proxy.py" in paths

    def test_multiple_files(self):
        paths = _extract_file_paths("Check auth.py and models.py")
        assert "auth.py" in paths
        assert "models.py" in paths

    def test_dotfile_with_extension(self):
        # Pure dotfiles like .gitignore have no extension — the regex requires *.ext
        paths = _extract_file_paths("Check .env.example for secrets")
        assert ".env.example" in paths

    def test_pure_dotfile(self):
        # Pure dotfiles like .gitignore are valid file paths
        paths = _extract_file_paths("Edit the .gitignore file")
        assert ".gitignore" in paths

    def test_nested_path(self):
        paths = _extract_file_paths("Read tests/test_proxy.py")
        assert "tests/test_proxy.py" in paths

    def test_no_paths(self):
        assert _extract_file_paths("Just a normal sentence") == []

    def test_empty_string(self):
        assert _extract_file_paths("") == []

    def test_none_returns_empty(self):
        assert _extract_file_paths(None) == []

    def test_skips_http_urls(self):
        paths = _extract_file_paths("Visit http://example.com/page.html and check config.py")
        # http URL should be filtered out
        assert "config.py" in paths
        for p in paths:
            assert not p.startswith("http")

    def test_deduplicates(self):
        paths = _extract_file_paths("Read auth.py then check auth.py again")
        assert paths.count("auth.py") == 1


# ── _abstract_tool_call ──────────────────────────────────────


class TestAbstractToolCall:
    def test_python_file_abstracts_to_star_py(self):
        result = _abstract_tool_call("read_file", {"path": "auth.py"})
        assert result == "read_file:*.py"

    def test_well_known_file_kept_as_is(self):
        result = _abstract_tool_call("read_file", {"path": "requirements.txt"})
        assert result == "read_file:requirements.txt"

    def test_well_known_with_dir_prefix(self):
        result = _abstract_tool_call("read_file", {"path": "/project/pyproject.toml"})
        assert result == "read_file:pyproject.toml"

    def test_nested_python_file(self):
        result = _abstract_tool_call("read_file", {"file_path": "src/models/user.py"})
        assert result == "read_file:*.py"

    def test_grep_always_star(self):
        result = _abstract_tool_call("grep", {"pattern": "TODO"})
        assert result == "grep:*"

    def test_list_directory_always_star(self):
        result = _abstract_tool_call("list_directory", {})
        assert result == "list_directory:*"

    def test_read_with_file_path_key(self):
        result = _abstract_tool_call("Read", {"file_path": "config.yaml"})
        assert result == "Read:*.yaml"

    def test_cat_tool(self):
        result = _abstract_tool_call("cat", {"file": "Dockerfile"})
        assert result == "cat:Dockerfile"

    def test_no_args_returns_star(self):
        result = _abstract_tool_call("read_file", {})
        assert result == "read_file:*"

    def test_file_without_extension(self):
        result = _abstract_tool_call("read_file", {"path": "Makefile"})
        assert result == "read_file:Makefile"


# ── _pattern_score (Wilson) ──────────────────────────────────


class TestPatternScore:
    def test_zero_observations(self):
        """Below _MIN_OBSERVATIONS → 0."""
        assert ToolPatternLearner._pattern_score(0, 0) == 0.0

    def test_below_min_observations(self):
        """2 total < _MIN_OBSERVATIONS(3) → 0."""
        assert ToolPatternLearner._pattern_score(1, 1) == 0.0

    def test_small_sample_low_score(self):
        """3 hits, 0 misses — still a small sample so score < 1."""
        score = ToolPatternLearner._pattern_score(3, 0)
        assert 0 < score < 1.0

    def test_many_hits_high_score(self):
        """100 hits, 0 misses → high score."""
        score = ToolPatternLearner._pattern_score(100, 0)
        assert score > 0.9

    def test_many_misses_low_score(self):
        """0 hits, 100 misses → very low score."""
        score = ToolPatternLearner._pattern_score(0, 100)
        assert score < 0.05

    def test_mixed_results(self):
        """50 hits, 50 misses → moderate score."""
        score = ToolPatternLearner._pattern_score(50, 50)
        assert 0.3 < score < 0.6

    def test_monotonic_with_hits(self):
        """More hits (same misses) → higher score."""
        s1 = ToolPatternLearner._pattern_score(5, 5)
        s2 = ToolPatternLearner._pattern_score(15, 5)
        s3 = ToolPatternLearner._pattern_score(50, 5)
        assert s1 < s2 < s3


# ── record_prediction + observe_outcome ──────────────────────


class TestFeedbackLoop:
    def test_record_and_observe(self, tmp_path: Path):
        learner = ToolPatternLearner(
            persistence_path=str(tmp_path / "patterns.json"),
            project_dir=str(tmp_path),
        )
        # Record a prediction
        learner.record_prediction(
            request_id="req-1",
            predicted_tools=[{"tool": "read_file", "args": {"path": "auth.py"}}],
            context_signals={"mentioned_files": ["auth.py"]},
        )

        # Observe the actual tools called
        learner.observe_outcome(
            actual_tools=[{"tool": "read_file", "args": {"path": "auth.py"}}],
            messages=[{"role": "user", "content": "Fix the auth module in auth.py"}],
        )

        stats = learner.get_stats()
        assert int(stats.get("total_observations", 0)) == 1
        accuracy = stats.get("tool_accuracy", {})
        assert "read_file" in accuracy

    def test_wrong_prediction_tracked(self, tmp_path: Path):
        learner = ToolPatternLearner(
            persistence_path=str(tmp_path / "patterns.json"),
            project_dir=str(tmp_path),
        )
        learner.record_prediction(
            request_id="req-1",
            predicted_tools=[{"tool": "grep", "args": {"pattern": "TODO"}}],
            context_signals={},
        )
        # LLM actually called read_file, not grep
        learner.observe_outcome(
            actual_tools=[{"tool": "read_file", "args": {"path": "main.py"}}],
            messages=[{"role": "user", "content": "Look at main.py"}],
        )

        stats = learner.get_stats()
        accuracy = stats.get("tool_accuracy", {})
        # grep was predicted but not called → wrong
        grep_acc = accuracy.get("grep", {})
        assert int(grep_acc.get("wrong", 0)) > 0


# ── suggest_tools (cold start) ───────────────────────────────


class TestSuggestTools:
    def test_no_data_returns_empty(self, tmp_path: Path):
        learner = ToolPatternLearner(
            persistence_path=str(tmp_path / "patterns.json"),
            project_dir=str(tmp_path),
        )
        suggestions = learner.suggest_tools([{"role": "user", "content": "Hello world"}])
        assert suggestions == []

    def test_no_messages_returns_empty(self, tmp_path: Path):
        learner = ToolPatternLearner(
            persistence_path=str(tmp_path / "patterns.json"),
            project_dir=str(tmp_path),
        )
        assert learner.suggest_tools([]) == []


# ── Persistence round-trip ───────────────────────────────────


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path):
        path = str(tmp_path / "patterns.json")
        learner = ToolPatternLearner(persistence_path=path, project_dir=str(tmp_path))

        # Generate some data
        learner.observe_outcome(
            actual_tools=[{"tool": "read_file", "args": {"path": "config.py"}}],
            messages=[{"role": "user", "content": "Check config.py"}],
        )
        learner.flush()

        # Verify file exists
        assert Path(path).exists()

        # Load into new instance
        learner2 = ToolPatternLearner(persistence_path=path, project_dir=str(tmp_path))
        stats = learner2.get_stats()
        assert int(stats.get("total_observations", 0)) == 1

    def test_load_nonexistent_file(self, tmp_path: Path):
        path = str(tmp_path / "does_not_exist.json")
        learner = ToolPatternLearner(persistence_path=path, project_dir=str(tmp_path))
        stats = learner.get_stats()
        assert int(stats.get("total_observations", 0)) == 0

    def test_load_corrupt_file(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        # Should not crash — starts fresh
        learner = ToolPatternLearner(persistence_path=str(path), project_dir=str(tmp_path))
        stats = learner.get_stats()
        assert int(stats.get("total_observations", 0)) == 0

    def test_load_wrong_version(self, tmp_path: Path):
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"version": 99}), encoding="utf-8")
        learner = ToolPatternLearner(persistence_path=str(path), project_dir=str(tmp_path))
        stats = learner.get_stats()
        assert int(stats.get("total_observations", 0)) == 0


# ── Staleness decay ──────────────────────────────────────────


class TestStalenessDecay:
    def test_old_patterns_evicted_on_load(self, tmp_path: Path):
        path = tmp_path / "stale.json"
        now = time.time()
        old_time = now - 31 * 86400  # 31 days ago (beyond _EVICT_AGE of 30 days)

        data = {
            "version": 1,
            "total_observations": 5,
            "file_mention_patterns": {
                ".py->read_file:*.py": {"hits": 10, "misses": 0, "last_seen": old_time},
            },
            "sequential_patterns": {},
            "result_content_patterns": {},
            "project_first_read": {},
            "tool_accuracy": {},
        }
        path.write_text(json.dumps(data), encoding="utf-8")

        learner = ToolPatternLearner(persistence_path=str(path), project_dir=str(tmp_path))
        stats = learner.get_stats()
        file_patterns = stats.get("file_mention_patterns", {})
        # The 31-day-old pattern should have been evicted
        assert ".py->read_file:*.py" not in file_patterns

    def test_stale_patterns_halved_on_load(self, tmp_path: Path):
        path = tmp_path / "stale.json"
        now = time.time()
        stale_time = now - 8 * 86400  # 8 days ago (beyond _HALF_LIFE of 7 days but within _EVICT_AGE)

        data = {
            "version": 1,
            "total_observations": 5,
            "file_mention_patterns": {
                ".py->read_file:*.py": {"hits": 20, "misses": 0, "last_seen": stale_time},
            },
            "sequential_patterns": {},
            "result_content_patterns": {},
            "project_first_read": {},
            "tool_accuracy": {},
        }
        path.write_text(json.dumps(data), encoding="utf-8")

        learner = ToolPatternLearner(persistence_path=str(path), project_dir=str(tmp_path))
        stats = learner.get_stats()
        file_patterns = stats.get("file_mention_patterns", {})
        entry = file_patterns.get(".py->read_file:*.py", {})
        # Hits should be halved: 20 // 2 = 10
        assert int(entry.get("hits", 0)) == 10

    def test_recent_patterns_not_decayed(self, tmp_path: Path):
        path = tmp_path / "fresh.json"
        now = time.time()
        recent_time = now - 3600  # 1 hour ago

        data = {
            "version": 1,
            "total_observations": 5,
            "file_mention_patterns": {
                ".py->read_file:*.py": {"hits": 20, "misses": 0, "last_seen": recent_time},
            },
            "sequential_patterns": {},
            "result_content_patterns": {},
            "project_first_read": {},
            "tool_accuracy": {},
        }
        path.write_text(json.dumps(data), encoding="utf-8")

        learner = ToolPatternLearner(persistence_path=str(path), project_dir=str(tmp_path))
        stats = learner.get_stats()
        file_patterns = stats.get("file_mention_patterns", {})
        entry = file_patterns.get(".py->read_file:*.py", {})
        # Hits should remain unchanged
        assert int(entry.get("hits", 0)) == 20


# ── _extract_error_keywords ─────────────────────────────────


class TestExtractErrorKeywords:
    def test_finds_keywords(self):
        result = _extract_error_keywords("Traceback (most recent call last):\n  ImportError: no module")
        assert "Traceback" in result
        assert "ImportError" in result

    def test_empty_string(self):
        assert _extract_error_keywords("") == []

    def test_none_returns_empty(self):
        assert _extract_error_keywords(None) == []

    def test_no_keywords(self):
        assert _extract_error_keywords("All tests passed") == []


# ── _normalize_tool_call ────────────────────────────────────


class TestNormalizeToolCall:
    def test_read_file_with_path(self):
        result = _normalize_tool_call({"tool": "read_file", "args": {"path": "src/auth.py"}})
        assert result == "read_file:auth.py"

    def test_read_file_with_file_path(self):
        result = _normalize_tool_call({"tool": "Read", "args": {"file_path": "config.yaml"}})
        assert result == "Read:config.yaml"

    def test_grep_star(self):
        result = _normalize_tool_call({"tool": "grep", "args": {"pattern": "TODO"}})
        assert result == "grep:*"

    def test_no_tool_key(self):
        result = _normalize_tool_call({"name": "Read", "args": {"file_path": "a.py"}})
        assert result == "Read:a.py"

    def test_args_as_json_string(self):
        result = _normalize_tool_call({"tool": "read_file", "args": '{"path": "test.py"}'})
        assert result == "read_file:test.py"

    def test_args_as_invalid_json(self):
        result = _normalize_tool_call({"tool": "read_file", "args": "not json"})
        assert result == "read_file:*"

    def test_args_as_non_dict(self):
        result = _normalize_tool_call({"tool": "read_file", "args": [1, 2, 3]})
        assert result == "read_file:*"


# ── extract_context_signals ────────────────────────────────


class TestExtractContextSignals:
    def test_empty_messages(self):
        signals = extract_context_signals([])
        assert signals["mentioned_files"] == []
        assert signals["is_first_turn"] is False

    def test_first_turn(self):
        signals = extract_context_signals([{"role": "user", "content": "Hello"}])
        assert signals["is_first_turn"] is True

    def test_not_first_turn(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ]
        signals = extract_context_signals(messages)
        assert signals["is_first_turn"] is False

    def test_file_mentions_extracted(self):
        signals = extract_context_signals([{"role": "user", "content": "Fix auth.py and models.py"}])
        assert "auth.py" in signals["mentioned_files"]
        assert ".py" in signals["mentioned_extensions"]

    def test_error_keywords_from_tool_result(self):
        messages = [
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [{"function": {"name": "Bash", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "Traceback (most recent call last):\n  ImportError: no module"},
        ]
        signals = extract_context_signals(messages)
        assert signals["result_has_error"] is True
        assert "Traceback" in signals["result_error_keywords"]
        assert signals["last_tool"] == "Bash"

    def test_no_tool_calls_in_assistant(self):
        messages = [
            {"role": "assistant", "content": "just text"},
            {"role": "tool", "content": "some output"},
        ]
        signals = extract_context_signals(messages)
        assert signals["last_tool"] == ""

    def test_tool_call_with_json_args(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "x.py"}'}}],
            },
            {"role": "tool", "content": "file contents"},
        ]
        signals = extract_context_signals(messages)
        assert signals["last_tool"] == "read_file"
        assert signals["last_tool_args"] == {"path": "x.py"}

    def test_tool_call_with_invalid_args(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "Bash", "arguments": "not json"}}],
            },
            {"role": "tool", "content": "output"},
        ]
        signals = extract_context_signals(messages)
        assert signals["last_tool"] == "Bash"
        assert signals["last_tool_args"] == {}


# ── _learn_patterns (via observe_outcome) ───────────────────


class TestLearnPatterns:
    def test_file_mention_pattern(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        learner.observe_outcome(
            actual_tools=[{"tool": "read_file", "args": {"path": "auth.py"}}],
            messages=[{"role": "user", "content": "Fix the auth module in auth.py"}],
        )
        stats = learner.get_stats()
        fp = stats.get("file_mention_patterns", {})
        # .py extension mentioned → read_file:*.py pattern
        assert any(".py" in k and "read_file" in k for k in fp)

    def test_sequential_pattern(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        learner.observe_outcome(
            actual_tools=[
                {"tool": "read_file", "args": {"path": "auth.py"}},
                {"tool": "grep", "args": {"pattern": "login"}},
            ],
            messages=[{"role": "user", "content": "Search auth.py for login"}],
        )
        stats = learner.get_stats()
        sp = stats.get("sequential_patterns", {})
        assert len(sp) >= 1
        # Should have read_file:*.py→grep:* pattern
        assert any("read_file" in k and "grep" in k for k in sp)

    def test_project_first_read_pattern(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        # First turn (single user message)
        learner.observe_outcome(
            actual_tools=[{"tool": "read_file", "args": {"path": "README.md"}}],
            messages=[{"role": "user", "content": "What is this project?"}],
        )
        stats = learner.get_stats()
        pf = stats.get("project_first_read", {})
        assert len(pf) >= 1

    def test_error_content_pattern(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "Bash", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "ImportError: no module named 'foo'"},
            {"role": "user", "content": "Fix this error in main.py"},
        ]
        learner.observe_outcome(
            actual_tools=[{"tool": "read_file", "args": {"path": "main.py"}}],
            messages=messages,
        )
        stats = learner.get_stats()
        cp = stats.get("result_content_patterns", {})
        assert any("ImportError" in k for k in cp)

    def test_sequential_pattern_trimming(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        # Force many sequential patterns
        for i in range(210):
            learner.observe_outcome(
                actual_tools=[
                    {"tool": "read_file", "args": {"path": f"file{i}.py"}},
                    {"tool": f"tool_{i}", "args": {}},
                ],
                messages=[{"role": "user", "content": f"Fix file{i}.py"}],
            )
        stats = learner.get_stats()
        sp = stats.get("sequential_patterns", {})
        assert len(sp) <= 200


# ── suggest_tools with data ─────────────────────────────────


class TestSuggestToolsWithData:
    def test_file_mention_suggestion(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        # Build enough data for Wilson score
        for _ in range(10):
            learner.observe_outcome(
                actual_tools=[{"tool": "read_file", "args": {"path": "auth.py"}}],
                messages=[{"role": "user", "content": "Fix auth.py"}],
            )
        suggestions = learner.suggest_tools([{"role": "user", "content": "Fix auth.py"}])
        # Should suggest read_file for .py files
        assert any(s.get("source") == "file_mention" for s in suggestions)

    def test_sequential_suggestion(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        # Build sequential pattern: after read_file → grep
        for _ in range(10):
            learner.observe_outcome(
                actual_tools=[
                    {"tool": "read_file", "args": {"path": "x.py"}},
                    {"tool": "grep", "args": {"pattern": "TODO"}},
                ],
                messages=[{"role": "user", "content": "Check x.py"}],
            )
        # Now simulate context where last_tool is read_file
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "x.py"}'}}],
            },
            {"role": "tool", "content": "file contents"},
            {"role": "user", "content": "Search for TODOs in x.py"},
        ]
        suggestions = learner.suggest_tools(messages)
        assert any(s.get("source") == "sequential" for s in suggestions)

    def test_project_first_read_suggestion(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        # Build first-read pattern
        for _ in range(10):
            learner.observe_outcome(
                actual_tools=[{"tool": "read_file", "args": {"path": "README.md"}}],
                messages=[{"role": "user", "content": "What is this?"}],
            )
        # Suggest on first turn
        suggestions = learner.suggest_tools([{"role": "user", "content": "Tell me about this project"}])
        assert any(s.get("source") == "project_first_read" for s in suggestions)

    def test_error_content_builds_pattern(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        # Build error pattern: ImportError → grep
        for _ in range(10):
            messages = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "Bash", "arguments": "{}"}}],
                },
                {"role": "tool", "content": "ImportError: no module named 'foo'"},
                {"role": "user", "content": "Fix main.py"},
            ]
            learner.observe_outcome(
                actual_tools=[{"tool": "grep", "args": {"pattern": "import"}}],
                messages=messages,
            )
        # Verify the pattern was stored
        stats = learner.get_stats()
        cp = stats.get("result_content_patterns", {})
        assert any("ImportError" in k and "grep" in k for k in cp)

    def test_suggestions_capped_at_3(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        # Build many patterns
        for i in range(20):
            learner.observe_outcome(
                actual_tools=[{"tool": "read_file", "args": {"path": f"f{i}.py"}}],
                messages=[{"role": "user", "content": f"Fix f{i}.py"}],
            )
        suggestions = learner.suggest_tools([{"role": "user", "content": "Fix f1.py"}])
        assert len(suggestions) <= 3

    def test_deduplicates_by_tool_and_args(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        # Build duplicate patterns from different sources
        for _ in range(10):
            learner.observe_outcome(
                actual_tools=[{"tool": "read_file", "args": {"path": "auth.py"}}],
                messages=[{"role": "user", "content": "Fix auth.py"}],
            )
        suggestions = learner.suggest_tools([{"role": "user", "content": "Fix auth.py"}])
        # Should not have duplicate read_file suggestions for the same file
        keys = [s.get("tool", "") + "|" + str(s.get("args", "")) for s in suggestions]
        assert len(keys) == len(set(keys))


# ── _update_accuracy ────────────────────────────────────────


class TestUpdateAccuracy:
    def test_correct_prediction(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        learner.record_prediction(
            "req1",
            [{"tool": "read_file", "args": {"path": "a.py"}}],
            {},
        )
        learner.observe_outcome(
            actual_tools=[{"tool": "read_file", "args": {"path": "a.py"}}],
            messages=[{"role": "user", "content": "Read a.py"}],
        )
        stats = learner.get_stats()
        acc = stats["tool_accuracy"]["read_file"]
        assert acc["correct"] >= 1

    def test_missed_prediction(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        learner.record_prediction("req1", [], {})
        learner.observe_outcome(
            actual_tools=[{"tool": "grep", "args": {"pattern": "x"}}],
            messages=[{"role": "user", "content": "Search for x"}],
        )
        stats = learner.get_stats()
        acc = stats["tool_accuracy"]["grep"]
        assert acc["missed"] >= 1

    def test_no_pending_prediction(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        # No record_prediction, just observe
        learner.observe_outcome(
            actual_tools=[{"tool": "read_file", "args": {"path": "a.py"}}],
            messages=[{"role": "user", "content": "Read a.py"}],
        )
        stats = learner.get_stats()
        assert "read_file" in stats["tool_accuracy"]

    def test_pending_eviction(self, tmp_path: Path):
        learner = ToolPatternLearner(persistence_path=str(tmp_path / "p.json"), project_dir=str(tmp_path))
        # Add 101 pending predictions to trigger eviction
        for i in range(101):
            learner.record_prediction(f"req{i}", [{"tool": "read_file", "args": {}}], {})
        assert len(learner._pending) <= 100


# ── Persistence edge cases ──────────────────────────────────


class TestPersistenceEdgeCases:
    def test_load_non_dict_root(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text('"just a string"', encoding="utf-8")
        learner = ToolPatternLearner(persistence_path=str(path), project_dir=str(tmp_path))
        assert int(learner.get_stats().get("total_observations", 0)) == 0

    def test_load_invalid_structure(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        data = {"version": 1, "file_mention_patterns": "not a dict"}
        path.write_text(json.dumps(data), encoding="utf-8")
        learner = ToolPatternLearner(persistence_path=str(path), project_dir=str(tmp_path))
        assert int(learner.get_stats().get("total_observations", 0)) == 0

    def test_save_creates_directory(self, tmp_path: Path):
        path = str(tmp_path / "subdir" / "deep" / "patterns.json")
        learner = ToolPatternLearner(persistence_path=path, project_dir=str(tmp_path))
        learner.observe_outcome(
            actual_tools=[{"tool": "read_file", "args": {"path": "x.py"}}],
            messages=[{"role": "user", "content": "Read x.py"}],
        )
        learner.flush()
        assert Path(path).exists()


# ── Decay edge cases ────────────────────────────────────────


class TestDecayEdgeCases:
    def test_sequential_patterns_decayed(self, tmp_path: Path):
        path = tmp_path / "decay.json"
        old = time.time() - 31 * 86400
        data = {
            "version": 1,
            "total_observations": 5,
            "file_mention_patterns": {},
            "sequential_patterns": {
                "read_file:*.py→grep:*": {"hits": 10, "misses": 0, "last_seen": old},
            },
            "result_content_patterns": {},
            "project_first_read": {},
            "tool_accuracy": {},
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        learner = ToolPatternLearner(persistence_path=str(path), project_dir=str(tmp_path))
        stats = learner.get_stats()
        assert "read_file:*.py→grep:*" not in stats.get("sequential_patterns", {})

    def test_content_patterns_decayed(self, tmp_path: Path):
        path = tmp_path / "decay.json"
        old = time.time() - 31 * 86400
        data = {
            "version": 1,
            "total_observations": 5,
            "file_mention_patterns": {},
            "sequential_patterns": {},
            "result_content_patterns": {
                "ImportError→read_file:*.py": {"hits": 10, "misses": 0, "last_seen": old},
            },
            "project_first_read": {},
            "tool_accuracy": {},
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        learner = ToolPatternLearner(persistence_path=str(path), project_dir=str(tmp_path))
        stats = learner.get_stats()
        assert "ImportError→read_file:*.py" not in stats.get("result_content_patterns", {})

    def test_project_first_read_decayed(self, tmp_path: Path):
        path = tmp_path / "decay.json"
        old = time.time() - 31 * 86400
        data = {
            "version": 1,
            "total_observations": 5,
            "file_mention_patterns": {},
            "sequential_patterns": {},
            "result_content_patterns": {},
            "project_first_read": {
                "proj123": {
                    "read_file:README.md": {"hits": 10, "misses": 0, "last_seen": old},
                }
            },
            "tool_accuracy": {},
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        learner = ToolPatternLearner(persistence_path=str(path), project_dir=str(tmp_path))
        stats = learner.get_stats()
        proj = stats.get("project_first_read", {}).get("proj123", {})
        assert "read_file:README.md" not in proj

    def test_project_first_read_halved(self, tmp_path: Path):
        path = tmp_path / "decay.json"
        stale = time.time() - 8 * 86400
        data = {
            "version": 1,
            "total_observations": 5,
            "file_mention_patterns": {},
            "sequential_patterns": {},
            "result_content_patterns": {},
            "project_first_read": {
                "proj123": {
                    "read_file:README.md": {"hits": 20, "misses": 0, "last_seen": stale},
                }
            },
            "tool_accuracy": {},
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        learner = ToolPatternLearner(persistence_path=str(path), project_dir=str(tmp_path))
        stats = learner.get_stats()
        entry = stats["project_first_read"]["proj123"]["read_file:README.md"]
        assert entry["hits"] == 10

    def test_invalid_entry_in_patterns_removed(self, tmp_path: Path):
        path = tmp_path / "decay.json"
        data = {
            "version": 1,
            "total_observations": 5,
            "file_mention_patterns": {"bad_key": "not a dict"},
            "sequential_patterns": {},
            "result_content_patterns": {},
            "project_first_read": {"proj": "not a dict"},
            "tool_accuracy": {},
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        learner = ToolPatternLearner(persistence_path=str(path), project_dir=str(tmp_path))
        stats = learner.get_stats()
        assert "bad_key" not in stats.get("file_mention_patterns", {})

    def test_invalid_entry_in_project_first_read(self, tmp_path: Path):
        path = tmp_path / "decay.json"
        data = {
            "version": 1,
            "total_observations": 5,
            "file_mention_patterns": {},
            "sequential_patterns": {},
            "result_content_patterns": {},
            "project_first_read": {
                "proj123": {"key": "not a dict entry"},
            },
            "tool_accuracy": {},
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        learner = ToolPatternLearner(persistence_path=str(path), project_dir=str(tmp_path))
        stats = learner.get_stats()
        proj = stats.get("project_first_read", {}).get("proj123", {})
        assert "key" not in proj
