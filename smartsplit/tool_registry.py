"""Tool registry — single source of truth for all tool definitions.

Every tool name, alias, category, and handler mapping lives here.
Other modules import what they need from this registry instead of
maintaining their own independent lists.

Categories:
- SAFE_TOOLS: Read-only tools that can be predicted/anticipated (never side effects)
- EXECUTABLE_TOOLS: Subset of SAFE_TOOLS we can actually run locally
- RECOGNIZE_ONLY_TOOLS: Safe tools we classify but cannot execute (IDE/LSP/MCP)

Compression categories (for Tool-Aware Proxy):
- DUMB_TOOLS: Small results, pass through as-is
- SMART_TOOLS: Potentially large results, compress before sending to brain
- DECISIONAL_TOOLS: Brain decides content, never touch

Alias mapping:
- Maps agent-specific tool names to canonical handler names
"""

from __future__ import annotations

import re

# ── Canonical handler names ───────────────────────────────────
# These are the base tool names that have actual execution handlers
# in tool_anticipator.py.

CANONICAL_HANDLERS = frozenset(
    {
        "read_file",
        "list_directory",
        "grep",
        "web_search",
        "web_fetch",
        "git_status",
        "git_log",
        "git_diff",
        "git_show",
        "git_blame",
    }
)

# ── Alias mapping ─────────────────────────────────────────────
# Maps every known tool name to its canonical handler.
# Keys are agent-specific names (Claude Code, Cursor, Cline, Aider, OpenCode).
# Values are the canonical handler name from CANONICAL_HANDLERS.

TOOL_ALIAS: dict[str, str] = {
    # File reading → read_file
    "Read": "read_file",
    "cat": "read_file",
    "head": "read_file",
    "tail": "read_file",
    "NotebookRead": "read_file",
    # Directory listing → list_directory
    "list_files": "list_directory",
    "list_dir": "list_directory",
    "Glob": "list_directory",
    "LS": "list_directory",
    "ls": "list_directory",
    # Content search → grep
    "Grep": "grep",
    "grep_search": "grep",
    "search_files": "grep",
    "find": "grep",
    "file_search": "grep",
    # Web search → web_search
    "WebSearch": "web_search",
    # Web fetch → web_fetch
    "WebFetch": "web_fetch",
    "fetch": "web_fetch",
}

# ── Executable tools ──────────────────────────────────────────
# All tools we can actually execute locally (canonical + aliases).

EXECUTABLE_TOOLS: frozenset[str] = CANONICAL_HANDLERS | frozenset(TOOL_ALIAS.keys())

# ── Recognize-only tools ──────────────────────────────────────
# Read-only tools that we classify as safe (to avoid misidentifying as write
# tools) but cannot execute locally. These are IDE/LSP features or MCP tools
# that only the agent itself can run.

RECOGNIZE_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        # IDE/LSP navigation
        "codebase_search",
        "goToDefinition",
        "findReferences",
        "hover",
        "documentSymbol",
        "workspaceSymbol",
        "incomingCalls",
        "outgoingCalls",
        "list_code_definition_names",
        # MCP tools
        "query-docs",
        "resolve-library-id",
        "access_mcp_resource",
        "ReadMcpResource",
    }
)

# ── SAFE_TOOLS ────────────────────────────────────────────────
# Complete set of read-only tools (executable + recognize-only).
# Used by intention_detector to filter LLM predictions: if the LLM
# predicts a tool not in this set, it's rejected.

SAFE_TOOLS: frozenset[str] = EXECUTABLE_TOOLS | RECOGNIZE_ONLY_TOOLS

# ── Compression categories (Tool-Aware Proxy) ────────────────
# Used by the proxy to decide how to handle tool results.

# Small/structured results, pass through untouched.
# Grep is here because the LLM needs full search results to make decisions
# (e.g. "which modules exist", "find all usages"). Truncating grep results
# removes context the LLM needs. Grep output is already structured and compact.
DUMB_TOOLS: frozenset[str] = frozenset(
    {
        "Read",
        "read_file",
        "cat",
        "head",
        "tail",
        "NotebookRead",
        "Glob",
        "list_files",
        "list_directory",
        "list_dir",
        "ls",
        "LS",
        "git_status",
        "git_show",
        # Grep/search — structured results, LLM needs full context
        "Grep",
        "grep",
        "grep_search",
        "search_files",
        "find",
        "file_search",
        "codebase_search",
    }
)

# Potentially large unstructured results — compress if above threshold.
SMART_TOOLS: frozenset[str] = frozenset(
    {
        # Web (HTML pages, search results — verbose, compressible)
        "WebSearch",
        "WebFetch",
        "web_search",
        "web_fetch",
        "fetch",
        # Git (large output — diffs, logs, blame)
        "git_log",
        "git_diff",
        "git_blame",
    }
)

# Brain decides content — never touch results.
DECISIONAL_TOOLS: frozenset[str] = frozenset(
    {
        # Write/edit
        "Write",
        "write_file",
        "Edit",
        "edit_file",
        "create_file",
        "delete_file",
        "delete",
        # Execution (side effects — brain must see full output)
        "Bash",
        "bash",
        "execute",
        # Communication
        "send_message",
        "create_pr",
    }
)

# ── Read/write category sets (used by anticipation.py) ────────
# For filtering already-read and recently-written paths.

READ_TOOLS: frozenset[str] = frozenset({"read_file", "Read", "cat", "head", "tail", "NotebookRead"})
WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "write_file",
        "Write",
        "Edit",
        "edit_file",
        "create_file",
        "delete_file",
        "delete",
        "Bash",
        "bash",
        "execute",
    }
)
LIST_DIR_TOOLS: frozenset[str] = frozenset({"list_directory", "list_files", "list_dir", "Glob", "ls", "LS"})
SEARCH_TOOLS: frozenset[str] = frozenset({"web_search", "WebSearch"})
GREP_TOOLS: frozenset[str] = frozenset({"grep", "Grep", "grep_search", "search_files", "find", "file_search"})

# ── Tools the proxy can fake ──────────────────────────────────
# These must match tools the agent actually provides.

SAFE_READ_TOOLS_FOR_FAKING: frozenset[str] = frozenset(
    {
        "Read",
        "read_file",
        "Glob",
        "list_files",
        "list_directory",
        "Grep",
        "grep",
        "grep_search",
        "search_files",
    }
)

# ── Shared constants ────────────────────────────────────────

# Well-known project root config files — used for file path filtering
# (e.g. "pyproject.toml" is a real file, "Next.js" is not).
WELL_KNOWN_FILES: frozenset[str] = frozenset(
    {
        "requirements.txt",
        "pyproject.toml",
        "package.json",
        "tsconfig.json",
        "Makefile",
        "Dockerfile",
        "docker-compose.yml",
        "CLAUDE.md",
        "README.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "setup.py",
        "setup.cfg",
        ".gitignore",
        ".env",
        ".env.example",
        "Cargo.toml",
        "go.mod",
        "Gemfile",
        "pom.xml",
        "build.gradle",
    }
)

# File path extraction regex — centralized here as single source of truth.
# Handles standard files (path/file.ext) and dotfiles (.env, .gitignore).
FILE_REF_RE = re.compile(
    r'(?:^|[\s`"\'(])'
    r"([.\w/-]+\.[\w]{1,10}|\.[\w][\w.-]{1,30})"
    r'(?:[\s`"\'):,]|$)',
    re.MULTILINE,
)
