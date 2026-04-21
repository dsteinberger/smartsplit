"""Tests for SmartSplit tool anticipator."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smartsplit.exceptions import SmartSplitError
from smartsplit.tools.anticipator import (
    _DIR_ENTRY_LIMIT,
    _GREP_OUTPUT_LIMIT,
    _READ_FILE_LIMIT,
    AnticipatedTool,
    ToolAnticipator,
    ToolResult,
)

# ── _read_file ──────────────────────────────────────────────


class TestReadFile:
    def test_reads_real_temp_file(self, tmp_path: Path):
        f = tmp_path / "hello.py"
        f.write_text("print('hello')\n", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        content = anticipator._read_file(str(f))
        assert "print('hello')" in content

    def test_reads_relative_path(self, tmp_path: Path):
        f = tmp_path / "sub" / "hello.py"
        f.parent.mkdir()
        f.write_text("x = 1\n", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        content = anticipator._read_file("sub/hello.py")
        assert "x = 1" in content

    def test_rejects_binary_file(self, tmp_path: Path):
        f = tmp_path / "image.bin"
        f.write_bytes(b"\x00\x01\x02\x03binary data")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(Exception, match="binary file"):
            anticipator._read_file(str(f))

    def test_sandbox_blocks_path_traversal(self, tmp_path: Path):
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(Exception, match="outside working directory"):
            anticipator._read_file("../../etc/passwd")

    def test_empty_path_raises(self, tmp_path: Path):
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(Exception, match="empty path"):
            anticipator._read_file("")

    def test_file_not_found(self, tmp_path: Path):
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            anticipator._read_file("nonexistent.py")

    def test_fallback_search_finds_file_in_subdir(self, tmp_path: Path):
        """When path doesn't exist at root, _read_file searches recursively."""
        sub = tmp_path / "src" / "pkg"
        sub.mkdir(parents=True)
        target = sub / "proxy.py"
        target.write_text("# found me\n", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        content = anticipator._read_file("proxy.py")
        assert "# found me" in content

    def test_not_a_regular_file_raises(self, tmp_path: Path):
        """A directory path should raise 'not a regular file'."""
        subdir = tmp_path / "mydir"
        subdir.mkdir()

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(SmartSplitError, match="not a regular file"):
            anticipator._read_file(str(subdir))

    def test_permission_denied_on_read_bytes(self, tmp_path: Path):
        """Permission denied during binary check."""
        f = tmp_path / "secret.py"
        f.write_text("secret", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        original_read_bytes = Path.read_bytes

        def mock_read_bytes(self_path, *args, **kwargs):
            if self_path.name == "secret.py":
                raise PermissionError("nope")
            return original_read_bytes(self_path, *args, **kwargs)

        with patch.object(Path, "read_bytes", mock_read_bytes):
            with pytest.raises(SmartSplitError, match="permission denied"):
                anticipator._read_file(str(f))

    def test_unicode_decode_error(self, tmp_path: Path):
        """Non-UTF-8 text file raises decode error."""
        f = tmp_path / "latin.txt"
        f.write_bytes(b"caf\xe9 au lait")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(SmartSplitError, match="unable to decode as UTF-8"):
            anticipator._read_file(str(f))

    def test_permission_denied_on_read_text(self, tmp_path: Path):
        """Permission denied during read_text."""
        f = tmp_path / "locked.py"
        f.write_text("locked", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        original_read_text = Path.read_text

        def mock_read_text(self_path, **kwargs):
            if self_path.name == "locked.py":
                raise PermissionError("nope")
            return original_read_text(self_path, **kwargs)

        with patch.object(Path, "read_text", mock_read_text):
            with pytest.raises(SmartSplitError, match="permission denied"):
                anticipator._read_file(str(f))

    def test_truncates_large_file(self, tmp_path: Path):
        """Files exceeding _READ_FILE_LIMIT are truncated."""
        f = tmp_path / "huge.py"
        content = "x" * (_READ_FILE_LIMIT + 1000)
        f.write_text(content, encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        result = anticipator._read_file(str(f))
        assert "[truncated" in result
        assert len(result) < len(content)


# ── _list_directory ──────────────────────────────────────────


class TestListDirectory:
    def test_returns_entries(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("a", encoding="utf-8")
        (tmp_path / "b.txt").write_text("b", encoding="utf-8")
        (tmp_path / "subdir").mkdir()

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        result = anticipator._list_directory(str(tmp_path))
        assert "a.py" in result
        assert "b.txt" in result
        assert "subdir/" in result

    def test_sandbox_blocks_outside_dir(self, tmp_path: Path):
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(Exception, match="outside working directory"):
            anticipator._list_directory("../../")

    def test_not_found_raises(self, tmp_path: Path):
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            anticipator._list_directory("no_such_dir")

    def test_not_a_directory_raises(self, tmp_path: Path):
        """A file path raises 'not a directory'."""
        f = tmp_path / "file.txt"
        f.write_text("hello", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(SmartSplitError, match="not a directory"):
            anticipator._list_directory(str(f))

    def test_permission_denied(self, tmp_path: Path):
        """Permission denied when iterating directory."""
        target = tmp_path / "restricted"
        target.mkdir()

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        with patch.object(Path, "iterdir", side_effect=PermissionError("nope")):
            with pytest.raises(SmartSplitError, match="permission denied"):
                anticipator._list_directory(str(target))

    def test_truncates_large_directory(self, tmp_path: Path):
        """Directories with more than _DIR_ENTRY_LIMIT entries are truncated."""
        for i in range(_DIR_ENTRY_LIMIT + 10):
            (tmp_path / f"file_{i:04d}.txt").write_text("x", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        result = anticipator._list_directory(str(tmp_path))
        assert "[truncated" in result


# ── _grep ────────────────────────────────────────────────────


class TestGrep:
    def test_grep_finds_pattern(self, tmp_path: Path):
        """Basic grep finds matching lines."""
        f = tmp_path / "code.py"
        f.write_text("import os\nimport sys\n", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        result = anticipator._grep("import os")
        assert "import os" in result

    def test_grep_empty_pattern_raises(self, tmp_path: Path):
        """Empty pattern raises SmartSplitError."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(SmartSplitError, match="empty pattern"):
            anticipator._grep("")

    def test_grep_with_specific_path(self, tmp_path: Path):
        """Grep with a specific path argument."""
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "mod.py").write_text("def hello():\n    pass\n", encoding="utf-8")
        (tmp_path / "other.py").write_text("def hello():\n    pass\n", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        result = anticipator._grep("hello", str(sub))
        assert "mod.py" in result

    def test_grep_sandbox_blocks_outside_path(self, tmp_path: Path):
        """Grep with path outside working_dir is rejected."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(SmartSplitError, match="outside working directory"):
            anticipator._grep("pattern", "/etc")

    def test_grep_truncates_long_output(self, tmp_path: Path):
        """Grep output exceeding limit is truncated."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        long_output = "x" * (_GREP_OUTPUT_LIMIT + 500)
        mock_result = MagicMock()
        mock_result.stdout = long_output

        with patch("subprocess.run", return_value=mock_result):
            result = anticipator._grep("pattern")
        assert "[truncated" in result

    def test_grep_timeout_raises(self, tmp_path: Path):
        """Grep subprocess timeout raises TimeoutError."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("grep", 5)):
            with pytest.raises(TimeoutError, match="grep timed out"):
                anticipator._grep("pattern")


# ── _web_search ──────────────────────────────────────────────


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_web_search_uses_serper_first(self, tmp_path: Path):
        """web_search tries serper first."""
        registry = MagicMock()
        registry.call_search = AsyncMock(return_value="serper results")

        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        result = await anticipator._web_search("python asyncio")
        assert result == "serper results"
        registry.call_search.assert_called_once_with("serper", "python asyncio")

    @pytest.mark.asyncio
    async def test_web_search_falls_back_to_tavily(self, tmp_path: Path):
        """web_search falls back to tavily when serper fails."""
        registry = MagicMock()

        async def mock_call_search(name: str, query: str) -> str:
            if name == "serper":
                raise Exception("serper down")
            return "tavily results"

        registry.call_search = mock_call_search

        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        result = await anticipator._web_search("python asyncio")
        assert result == "tavily results"

    @pytest.mark.asyncio
    async def test_web_search_no_provider_available(self, tmp_path: Path):
        """web_search raises when all providers fail."""
        registry = MagicMock()
        registry.call_search = AsyncMock(side_effect=Exception("down"))

        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(SmartSplitError, match="no search provider available"):
            await anticipator._web_search("test query")

    @pytest.mark.asyncio
    async def test_web_search_empty_query_raises(self, tmp_path: Path):
        """web_search with empty query raises."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(SmartSplitError, match="empty query"):
            await anticipator._web_search("")


# ── _web_fetch ───────────────────────────────────────────────


class TestWebFetch:
    @pytest.mark.asyncio
    async def test_web_fetch_success(self, tmp_path: Path):
        """web_fetch returns content on success."""
        mock_response = MagicMock()
        mock_response.text = "<html>hello</html>"
        mock_response.raise_for_status = MagicMock()

        mock_http = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_response)

        registry = MagicMock()
        registry.http_client = mock_http

        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        result = await anticipator._web_fetch("https://example.com")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_web_fetch_empty_url_raises(self, tmp_path: Path):
        """web_fetch with empty URL raises."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(SmartSplitError, match="empty URL"):
            await anticipator._web_fetch("")

    @pytest.mark.asyncio
    async def test_web_fetch_http_error(self, tmp_path: Path):
        """web_fetch wraps HTTP errors in SmartSplitError."""
        mock_http = MagicMock()
        mock_http.get = AsyncMock(side_effect=Exception("connection refused"))

        registry = MagicMock()
        registry.http_client = mock_http

        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(SmartSplitError, match="web_fetch"):
            await anticipator._web_fetch("https://bad.example.com")

    @pytest.mark.asyncio
    async def test_web_fetch_truncates_large_content(self, tmp_path: Path):
        """web_fetch truncates content exceeding limit."""
        mock_response = MagicMock()
        mock_response.text = "x" * (_READ_FILE_LIMIT + 1000)
        mock_response.raise_for_status = MagicMock()

        mock_http = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_response)

        registry = MagicMock()
        registry.http_client = mock_http

        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        result = await anticipator._web_fetch("https://example.com/huge")
        assert "[truncated]" in result
        assert len(result) <= _READ_FILE_LIMIT + 50


# ── _git_command ─────────────────────────────────────────────


class TestGitCommand:
    def test_git_status(self, tmp_path: Path):
        """git_status runs 'git status --short'."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        mock_result = MagicMock()
        mock_result.stdout = " M file.py\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = anticipator._git_command("git_status", {})
        assert "file.py" in result
        cmd_args = mock_run.call_args[0][0]
        assert cmd_args == ["git", "status", "--short"]

    def test_git_log(self, tmp_path: Path):
        """git_log runs 'git log --oneline -20'."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        mock_result = MagicMock()
        mock_result.stdout = "abc1234 initial commit\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = anticipator._git_command("git_log", {})
        assert "initial commit" in result
        cmd_args = mock_run.call_args[0][0]
        assert cmd_args == ["git", "log", "--oneline", "-20"]

    def test_git_blame_with_path(self, tmp_path: Path):
        """git_blame appends file path to command."""
        f = tmp_path / "code.py"
        f.write_text("pass\n", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        mock_result = MagicMock()
        mock_result.stdout = "abc1234 (author 2024-01-01) pass\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = anticipator._git_command("git_blame", {"path": str(f)})
        assert "abc1234" in result
        cmd_args = mock_run.call_args[0][0]
        assert str(f) in cmd_args[-1]

    def test_git_blame_relative_path(self, tmp_path: Path):
        """git_blame resolves relative path to absolute."""
        f = tmp_path / "src" / "main.py"
        f.parent.mkdir()
        f.write_text("pass\n", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        mock_result = MagicMock()
        mock_result.stdout = "blame output\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            anticipator._git_command("git_blame", {"path": "src/main.py"})
        cmd_args = mock_run.call_args[0][0]
        assert str(tmp_path / "src" / "main.py") == cmd_args[-1]

    def test_git_blame_sandbox_blocks_traversal(self, tmp_path: Path):
        """git_blame rejects paths outside working_dir."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(SmartSplitError, match="outside working directory"):
            anticipator._git_command("git_blame", {"path": "/etc/passwd"})

    def test_git_unknown_command_raises(self, tmp_path: Path):
        """Unknown git subcommand raises SmartSplitError."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        with pytest.raises(SmartSplitError, match="unknown command"):
            anticipator._git_command("git_push", {})

    def test_git_timeout(self, tmp_path: Path):
        """Git subprocess timeout raises TimeoutError."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 5)):
            with pytest.raises(TimeoutError, match="git command timed out"):
                anticipator._git_command("git_status", {})

    def test_git_truncates_long_output(self, tmp_path: Path):
        """Git output exceeding limit is truncated."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        mock_result = MagicMock()
        mock_result.stdout = "x" * (_GREP_OUTPUT_LIMIT + 500)

        with patch("subprocess.run", return_value=mock_result):
            result = anticipator._git_command("git_diff", {})
        assert "[truncated]" in result

    def test_git_diff_with_file_arg(self, tmp_path: Path):
        """git_diff also accepts 'file' arg key."""
        f = tmp_path / "changed.py"
        f.write_text("new\n", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        mock_result = MagicMock()
        mock_result.stdout = "diff output\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            anticipator._git_command("git_diff", {"file": str(f)})
        cmd_args = mock_run.call_args[0][0]
        assert str(f) in cmd_args[-1]

    def test_git_show_with_path(self, tmp_path: Path):
        """git_show appends file path when provided."""
        f = tmp_path / "show_me.py"
        f.write_text("show\n", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        mock_result = MagicMock()
        mock_result.stdout = "show stat output\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            anticipator._git_command("git_show", {"path": str(f)})
        cmd_args = mock_run.call_args[0][0]
        assert str(f) in cmd_args[-1]


# ── execute ─────────────────────────────────────────────────


class TestExecute:
    @pytest.mark.asyncio
    async def test_filters_non_safe_tools(self, tmp_path: Path):
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        tools = [
            AnticipatedTool(tool="write_file", args={"path": "x.py", "content": "x"}),
            AnticipatedTool(tool="execute_bash", args={"command": "rm -rf /"}),
        ]
        results = await anticipator.execute(tools)
        assert results == []

    @pytest.mark.asyncio
    async def test_executes_read_file(self, tmp_path: Path):
        f = tmp_path / "code.py"
        f.write_text("import os\n", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        tools = [AnticipatedTool(tool="read_file", args={"path": str(f)})]
        results = await anticipator.execute(tools)
        assert len(results) == 1
        assert results[0].success
        assert "import os" in results[0].content
        assert results[0].tokens_estimate > 0

    @pytest.mark.asyncio
    async def test_handles_file_not_found_gracefully(self, tmp_path: Path):
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        tools = [AnticipatedTool(tool="read_file", args={"path": "missing.py"})]
        results = await anticipator.execute(tools)
        assert len(results) == 1
        assert not results[0].success
        assert results[0].tokens_estimate == 0

    @pytest.mark.asyncio
    async def test_executes_list_directory(self, tmp_path: Path):
        (tmp_path / "file.txt").write_text("hello", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        tools = [AnticipatedTool(tool="list_directory", args={"path": str(tmp_path)})]
        results = await anticipator.execute(tools)
        assert len(results) == 1
        assert results[0].success
        assert "file.txt" in results[0].content

    @pytest.mark.asyncio
    async def test_empty_anticipated_returns_empty(self, tmp_path: Path):
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        results = await anticipator.execute([])
        assert results == []

    @pytest.mark.asyncio
    async def test_mixed_safe_unsafe(self, tmp_path: Path):
        f = tmp_path / "ok.py"
        f.write_text("pass\n", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        tools = [
            AnticipatedTool(tool="write_file", args={"path": "bad.py"}),
            AnticipatedTool(tool="read_file", args={"path": str(f)}),
        ]
        results = await anticipator.execute(tools)
        # Only the safe tool should be executed
        assert len(results) == 1
        assert results[0].tool == "read_file"
        assert results[0].success

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_internal_error(self, tmp_path: Path):
        """When gather returns a BaseException, result is internal error."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        async def _boom(tool: str, args: dict) -> ToolResult:
            raise RuntimeError("unexpected failure")

        anticipator._execute_one = _boom  # type: ignore[assignment]

        tools = [AnticipatedTool(tool="read_file", args={"path": "any.py"})]
        results = await anticipator.execute(tools)
        assert len(results) == 1
        assert not results[0].success
        assert results[0].content == "internal error"
        assert results[0].tokens_estimate == 0

    @pytest.mark.asyncio
    async def test_executes_grep_via_execute(self, tmp_path: Path):
        """Execute routes to grep handler correctly."""
        f = tmp_path / "search.py"
        f.write_text("hello world\n", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        tools = [AnticipatedTool(tool="grep", args={"pattern": "hello"})]
        results = await anticipator.execute(tools)
        assert len(results) == 1
        assert results[0].success
        assert "hello" in results[0].content

    @pytest.mark.asyncio
    async def test_executes_web_search_via_execute(self, tmp_path: Path):
        """Execute routes to web_search handler correctly."""
        registry = MagicMock()
        registry.call_search = AsyncMock(return_value="search results here")

        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        tools = [AnticipatedTool(tool="web_search", args={"query": "python"})]
        results = await anticipator.execute(tools)
        assert len(results) == 1
        assert results[0].success
        assert "search results here" in results[0].content

    @pytest.mark.asyncio
    async def test_executes_web_fetch_via_execute(self, tmp_path: Path):
        """Execute routes to web_fetch handler correctly."""
        mock_response = MagicMock()
        mock_response.text = "fetched content"
        mock_response.raise_for_status = MagicMock()

        mock_http = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_response)

        registry = MagicMock()
        registry.http_client = mock_http

        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        tools = [AnticipatedTool(tool="web_fetch", args={"url": "https://example.com"})]
        results = await anticipator.execute(tools)
        assert len(results) == 1
        assert results[0].success
        assert "fetched content" in results[0].content

    @pytest.mark.asyncio
    async def test_executes_git_status_via_execute(self, tmp_path: Path):
        """Execute routes to git_command handler correctly."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        mock_result = MagicMock()
        mock_result.stdout = " M proxy.py\n"

        with patch("subprocess.run", return_value=mock_result):
            tools = [AnticipatedTool(tool="git_status", args={})]
            results = await anticipator.execute(tools)
        assert len(results) == 1
        assert results[0].success
        assert "proxy.py" in results[0].content

    @pytest.mark.asyncio
    async def test_unsupported_tool_returns_failure(self, tmp_path: Path):
        """Unknown canonical tool returns unsupported."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        # Force a tool that passes the filter but has no handler
        original_safe = anticipator.SAFE_TOOLS
        anticipator.SAFE_TOOLS = frozenset(original_safe | {"unknown_read_tool"})

        tools = [AnticipatedTool(tool="unknown_read_tool", args={})]
        results = await anticipator.execute(tools)

        anticipator.SAFE_TOOLS = original_safe

        assert len(results) == 1
        assert not results[0].success
        assert results[0].content == "unsupported tool"


# ── _execute_one error handling ──────────────────────────────


class TestExecuteOneErrors:
    @pytest.mark.asyncio
    async def test_timeout_returns_failure(self, tmp_path: Path, monkeypatch):
        """A tool that exceeds _TOOL_TIMEOUT returns success=False."""
        import time

        import smartsplit.tools.anticipator as mod

        monkeypatch.setattr(mod, "_TOOL_TIMEOUT", 0.1)

        def _slow_read(path: str) -> str:
            time.sleep(1)
            return "should not reach"

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        monkeypatch.setattr(anticipator, "_read_file", _slow_read)

        result = await anticipator._execute_one("read_file", {"path": "slow.py"})
        assert not result.success
        assert result.content == "timeout"
        assert result.tokens_estimate == 0

    @pytest.mark.asyncio
    async def test_generic_exception_returns_error(self, tmp_path: Path, monkeypatch):
        """A tool that raises a generic exception returns a failure result."""
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        def _bad_read(path: str) -> str:
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(anticipator, "_read_file", _bad_read)

        result = await anticipator._execute_one("read_file", {"path": "crash.py"})
        assert not result.success
        assert "disk on fire" in result.content
        assert result.tokens_estimate == 0


# ── Tool alias resolution ────────────────────────────────────


class TestToolAlias:
    @pytest.mark.asyncio
    async def test_alias_Read_resolves_to_read_file(self, tmp_path: Path):
        """'Read' alias is resolved to 'read_file' handler."""
        f = tmp_path / "test.py"
        f.write_text("aliased\n", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        result = await anticipator._execute_one("Read", {"path": str(f)})
        assert result.success
        assert "aliased" in result.content

    @pytest.mark.asyncio
    async def test_alias_Grep_resolves_to_grep(self, tmp_path: Path):
        """'Grep' alias is resolved to 'grep' handler."""
        f = tmp_path / "data.txt"
        f.write_text("pattern_match\n", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        result = await anticipator._execute_one("Grep", {"pattern": "pattern_match"})
        assert result.success
        assert "pattern_match" in result.content

    @pytest.mark.asyncio
    async def test_alias_Glob_resolves_to_list_directory(self, tmp_path: Path):
        """'Glob' alias is resolved to 'list_directory' handler."""
        (tmp_path / "file.txt").write_text("x", encoding="utf-8")

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        result = await anticipator._execute_one("Glob", {"path": str(tmp_path)})
        assert result.success
        assert "file.txt" in result.content

    @pytest.mark.asyncio
    async def test_alias_WebSearch_resolves_to_web_search(self, tmp_path: Path):
        """'WebSearch' alias is resolved to 'web_search' handler."""
        registry = MagicMock()
        registry.call_search = AsyncMock(return_value="web results")

        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        result = await anticipator._execute_one("WebSearch", {"query": "test"})
        assert result.success
        assert "web results" in result.content

    @pytest.mark.asyncio
    async def test_alias_WebFetch_resolves_to_web_fetch(self, tmp_path: Path):
        """'WebFetch' alias is resolved to 'web_fetch' handler."""
        mock_response = MagicMock()
        mock_response.text = "page content"
        mock_response.raise_for_status = MagicMock()

        mock_http = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_response)

        registry = MagicMock()
        registry.http_client = mock_http

        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        result = await anticipator._execute_one("WebFetch", {"url": "https://example.com"})
        assert result.success
        assert "page content" in result.content


# ── ToolResult dataclass ────────────────────────────────────


class TestToolResult:
    def test_tool_result_is_frozen(self):
        """ToolResult is a frozen dataclass — attributes cannot be reassigned."""
        result = ToolResult(
            tool="read_file",
            args={"path": "x.py"},
            content="hello",
            success=True,
            tokens_estimate=1,
        )
        with pytest.raises(AttributeError):
            result.content = "modified"  # type: ignore[misc]


# ── Graceful failures (end-to-end) ──────────────────────────


class TestGracefulFailures:
    @pytest.mark.asyncio
    async def test_web_search_empty_query_fails_gracefully(self, tmp_path: Path):
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        tools = [AnticipatedTool(tool="web_search", args={"query": ""})]
        results = await anticipator.execute(tools)
        assert len(results) == 1
        assert not results[0].success

    @pytest.mark.asyncio
    async def test_read_file_empty_path_fails_gracefully(self, tmp_path: Path):
        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))

        tools = [AnticipatedTool(tool="read_file", args={"path": ""})]
        results = await anticipator.execute(tools)
        assert len(results) == 1
        assert not results[0].success


class TestTimeout:
    @pytest.mark.asyncio
    async def test_slow_tool_returns_timeout(self, tmp_path: Path, monkeypatch):
        """A tool that exceeds _TOOL_TIMEOUT returns success=False with content='timeout'."""
        import smartsplit.tools.anticipator as mod

        # Patch timeout to 0.1s so the test runs fast
        monkeypatch.setattr(mod, "_TOOL_TIMEOUT", 0.1)

        import time

        def _slow_read(path: str) -> str:
            time.sleep(1)  # way over 0.1s
            return "should not reach"

        registry = MagicMock()
        anticipator = ToolAnticipator(registry, working_dir=str(tmp_path))
        monkeypatch.setattr(anticipator, "_read_file", _slow_read)

        tools = [AnticipatedTool(tool="read_file", args={"path": "slow.py"})]
        results = await anticipator.execute(tools)
        assert len(results) == 1
        assert not results[0].success
        assert results[0].content == "timeout"
        assert results[0].tokens_estimate == 0


class TestEstimateTokens:
    def test_rough_estimate(self):
        assert ToolAnticipator._estimate_tokens("a" * 100) == 25
        assert ToolAnticipator._estimate_tokens("") == 0
