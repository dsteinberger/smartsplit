"""Tests for SmartSplit tool registry — cross-file consistency checks."""

from __future__ import annotations

from smartsplit.tools.registry import (
    CANONICAL_HANDLERS,
    DECISIONAL_TOOLS,
    DUMB_TOOLS,
    EXECUTABLE_TOOLS,
    GREP_TOOLS,
    LIST_DIR_TOOLS,
    READ_TOOLS,
    RECOGNIZE_ONLY_TOOLS,
    SAFE_TOOLS,
    SMART_TOOLS,
    TOOL_ALIAS,
    WRITE_TOOLS,
)


class TestRegistryConsistency:
    def test_executable_is_canonical_plus_aliases(self):
        """EXECUTABLE_TOOLS must be exactly canonical handlers + alias keys."""
        expected = CANONICAL_HANDLERS | frozenset(TOOL_ALIAS.keys())
        assert expected == EXECUTABLE_TOOLS

    def test_safe_tools_is_executable_plus_recognize_only(self):
        """SAFE_TOOLS must be exactly executable + recognize-only."""
        assert SAFE_TOOLS == EXECUTABLE_TOOLS | RECOGNIZE_ONLY_TOOLS

    def test_no_overlap_executable_and_recognize_only(self):
        """Executable and recognize-only must not overlap."""
        overlap = EXECUTABLE_TOOLS & RECOGNIZE_ONLY_TOOLS
        assert not overlap, f"Overlap: {overlap}"

    def test_all_aliases_point_to_canonical_handlers(self):
        """Every alias value must be a canonical handler name."""
        for alias, canonical in TOOL_ALIAS.items():
            assert canonical in CANONICAL_HANDLERS, (
                f"Alias '{alias}' points to '{canonical}' which is not in CANONICAL_HANDLERS"
            )

    def test_no_self_aliases(self):
        """No alias should point to itself."""
        for alias, canonical in TOOL_ALIAS.items():
            assert alias != canonical, f"Self-alias: '{alias}'"

    def test_canonical_handlers_not_in_aliases(self):
        """Canonical handler names should not appear as alias keys (they need no alias)."""
        in_both = CANONICAL_HANDLERS & frozenset(TOOL_ALIAS.keys())
        assert not in_both, f"Canonical handler also in TOOL_ALIAS: {in_both}"


class TestCompressionCategories:
    def test_no_overlap_dumb_smart(self):
        overlap = DUMB_TOOLS & SMART_TOOLS
        assert not overlap, f"Overlap DUMB/SMART: {overlap}"

    def test_no_overlap_dumb_decisional(self):
        overlap = DUMB_TOOLS & DECISIONAL_TOOLS
        assert not overlap, f"Overlap DUMB/DECISIONAL: {overlap}"

    def test_no_overlap_smart_decisional(self):
        overlap = SMART_TOOLS & DECISIONAL_TOOLS
        assert not overlap, f"Overlap SMART/DECISIONAL: {overlap}"

    def test_bash_is_decisional_not_smart(self):
        """Bash/execute must be in DECISIONAL (write tools), never in SMART."""
        for tool in ("Bash", "bash", "execute"):
            assert tool in DECISIONAL_TOOLS, f"'{tool}' missing from DECISIONAL_TOOLS"
            assert tool not in SMART_TOOLS, f"'{tool}' should NOT be in SMART_TOOLS"

    def test_git_status_and_show_are_dumb(self):
        """git_status and git_show have small output — should be DUMB."""
        for tool in ("git_status", "git_show"):
            assert tool in DUMB_TOOLS, f"'{tool}' missing from DUMB_TOOLS"

    def test_git_large_output_is_smart(self):
        """git_log, git_diff, git_blame can have large output — should be SMART."""
        for tool in ("git_log", "git_diff", "git_blame"):
            assert tool in SMART_TOOLS, f"'{tool}' missing from SMART_TOOLS"


class TestCategoryToolSets:
    def test_read_tools_are_file_readers(self):
        """READ_TOOLS should contain only file-reading tools."""
        for tool in READ_TOOLS:
            assert tool in SAFE_TOOLS, f"'{tool}' in READ_TOOLS but not SAFE_TOOLS"

    def test_write_tools_not_in_safe(self):
        """WRITE_TOOLS must never appear in SAFE_TOOLS."""
        overlap = WRITE_TOOLS & SAFE_TOOLS
        assert not overlap, f"Write tools in SAFE_TOOLS: {overlap}"

    def test_grep_tools_are_safe(self):
        for tool in GREP_TOOLS:
            assert tool in SAFE_TOOLS, f"'{tool}' in GREP_TOOLS but not SAFE_TOOLS"

    def test_list_dir_tools_are_safe(self):
        for tool in LIST_DIR_TOOLS:
            assert tool in SAFE_TOOLS, f"'{tool}' in LIST_DIR_TOOLS but not SAFE_TOOLS"

    def test_write_tools_are_decisional(self):
        """All WRITE_TOOLS should be in DECISIONAL_TOOLS."""
        missing = WRITE_TOOLS - DECISIONAL_TOOLS
        assert not missing, f"WRITE_TOOLS not in DECISIONAL: {missing}"


class TestAnticipatorAlignment:
    def test_anticipator_safe_tools_matches_registry(self):
        """ToolAnticipator.SAFE_TOOLS must equal EXECUTABLE_TOOLS from registry."""
        from smartsplit.tools.anticipator import ToolAnticipator

        assert ToolAnticipator.SAFE_TOOLS == EXECUTABLE_TOOLS

    def test_anticipator_alias_matches_registry(self):
        """ToolAnticipator._TOOL_ALIAS must equal TOOL_ALIAS from registry."""
        from smartsplit.tools.anticipator import ToolAnticipator

        assert ToolAnticipator._TOOL_ALIAS == TOOL_ALIAS


class TestDetectorAlignment:
    def test_detector_safe_tools_matches_registry(self):
        """intention_detector.SAFE_TOOLS must be the same object from registry."""
        from smartsplit.tools.intention_detector import SAFE_TOOLS as DETECTOR_SAFE_TOOLS

        assert DETECTOR_SAFE_TOOLS is SAFE_TOOLS
