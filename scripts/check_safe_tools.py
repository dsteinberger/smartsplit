"""Check upstream agent repos for new read-only tools not yet in SAFE_TOOLS.

Fetches tool definitions from Claude Code (npm), Cline, Continue, and OpenCode
GitHub repos, compares with our tool_registry.py, and reports new/missing tools
with a read-only heuristic based on upstream classification.

Usage:
    python scripts/check_safe_tools.py          # print report
    python scripts/check_safe_tools.py --json   # JSON output (for cron/webhook)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.request
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Upstream sources ─────────────────────────────────────────

SOURCES: list[dict[str, str]] = [
    {
        "name": "Cline",
        "url": "https://raw.githubusercontent.com/cline/cline/main/src/shared/tools.ts",
        "parser": "cline",
    },
    {
        "name": "Continue",
        "url": "https://raw.githubusercontent.com/continuedev/continue/main/core/tools/builtIn.ts",
        "parser": "continue",
    },
    {
        "name": "OpenCode",
        "url": "https://raw.githubusercontent.com/opencode-ai/opencode/main/internal/llm/agent/tools.go",
        "parser": "opencode",
    },
]

# ── Heuristic keywords ──────────────────────────────────────

# ── Control-flow / meta tools ────────────────────────────────
# Tools we never want to anticipate regardless of read-only status.
# These are agent control flow, not data access.
IGNORED_TOOLS: frozenset[str] = frozenset(
    {
        # Cline control flow
        "ask_followup_question",
        "attempt_completion",
        "plan_mode_respond",
        "act_mode_respond",
        "new_task",
        "focus_chain",
        "condense",
        "summarize_task",
        "report_bug",
        "generate_explanation",
        "use_skill",
        "use_subagents",
        # Cline browser (can click/type — not purely read)
        "browser_action",
        # Continue control flow
        "request_rule",
        "create_rule_block",
        # OpenCode delegation
        "agent",
    }
)

WRITE_KEYWORDS = re.compile(
    r"\b(write|edit|create|new|delete|remove|execute|run|replace|patch|apply|send|push|"
    r"modify|update|insert|append|rename|move|install|deploy|kill|terminate)\b",
    re.IGNORECASE,
)
READ_KEYWORDS = re.compile(
    r"\b(read|search|list|get|find|view|glob|grep|fetch|browse|inspect|show|"
    r"describe|display|query|look|scan|check|diagnostic|diff|blame|log|status)\b",
    re.IGNORECASE,
)


@dataclass
class UpstreamTool:
    name: str
    source: str
    read_only: bool | None = None  # None = unknown
    description: str = ""


@dataclass
class Report:
    new_safe: list[UpstreamTool] = field(default_factory=list)
    new_write: list[UpstreamTool] = field(default_factory=list)
    new_ambiguous: list[UpstreamTool] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ── Fetcher ──────────────────────────────────────────────────


def _fetch(url: str, timeout: int = 15) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "smartsplit-checker/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode()
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


# ── Parsers ──────────────────────────────────────────────────


def _parse_cline(content: str) -> list[UpstreamTool]:
    """Parse Cline tools.ts — extract enum values and READ_ONLY_TOOLS."""
    tools: list[UpstreamTool] = []

    # Extract all enum values: KEY = "value"
    all_tools = re.findall(r'=\s*"([^"]+)"', content)

    # Extract READ_ONLY_TOOLS array
    ro_match = re.search(r"READ_ONLY_TOOLS\s*=\s*\[(.*?)\]", content, re.DOTALL)
    ro_names: set[str] = set()
    if ro_match:
        ro_names = set(re.findall(r"\.(\w+)", ro_match.group(1)))
        # Resolve enum key → value
        enum_map: dict[str, str] = {}
        for m in re.finditer(r"(\w+)\s*=\s*\"([^\"]+)\"", content):
            enum_map[m.group(1)] = m.group(2)
        ro_values = {enum_map.get(k, k) for k in ro_names}
    else:
        ro_values = set()

    for name in all_tools:
        read_only = name in ro_values if ro_values else None
        tools.append(UpstreamTool(name=name, source="Cline", read_only=read_only))
    return tools


def _parse_continue(content: str) -> list[UpstreamTool]:
    """Parse Continue builtIn.ts — extract tool names from BuiltInToolNames enum.

    The readonly field lives in separate definition files (core/tools/definitions/*.ts),
    not in builtIn.ts. We extract enum values and use heuristic classification.
    """
    tools: list[UpstreamTool] = []

    # Extract enum values: EnumKey = "tool_name"
    # Only match inside the enum block to avoid catching constants like BUILT_IN_GROUP_NAME
    enum_match = re.search(r"enum\s+BuiltInToolNames\s*\{(.*?)\}", content, re.DOTALL)
    if enum_match:
        for m in re.finditer(r'=\s*"([^"]+)"', enum_match.group(1)):
            tools.append(UpstreamTool(name=m.group(1), source="Continue", read_only=None))

    return tools


def _parse_opencode(content: str) -> list[UpstreamTool]:
    """Parse OpenCode tools.go — extract CoderAgentTools and TaskAgentTools."""
    tools: list[UpstreamTool] = []
    seen: set[str] = set()

    # TaskAgentTools = read-only subset
    task_match = re.search(r"func TaskAgentTools.*?\{(.*?)\n\}", content, re.DOTALL)
    task_tools: set[str] = set()
    if task_match:
        task_tools = set(re.findall(r"tools\.(\w+)Tool", task_match.group(1)))
        # Also try Name() patterns
        task_tools |= set(re.findall(r'"(\w+)"', task_match.group(1)))

    # CoderAgentTools = full set
    coder_match = re.search(r"func CoderAgentTools.*?\{(.*?)\n\}", content, re.DOTALL)
    if coder_match:
        coder_body = coder_match.group(1)
        # Extract tool names from New*Tool() calls or string literals
        for m in re.finditer(r'tools\.New(\w+)Tool|"(\w+)"', coder_body):
            raw = m.group(1) or m.group(2)
            name = raw[0].lower() + raw[1:] if m.group(1) else raw
            # Normalize: NewBashTool → bash, NewEditTool → edit
            if m.group(1):
                name = raw.replace("Tool", "").lower()
            if name not in seen:
                seen.add(name)
                # read_only if it's in TaskAgentTools
                is_ro = name in task_tools or any(name.lower() == t.lower() for t in task_tools)
                tools.append(UpstreamTool(name=name, source="OpenCode", read_only=is_ro))

    # Direct extraction fallback
    if not tools:
        for m in re.findall(r'"(\w+)"', content):
            if m not in seen and len(m) > 1:
                seen.add(m)
                tools.append(UpstreamTool(name=m, source="OpenCode", read_only=None))

    return tools


PARSERS = {
    "cline": _parse_cline,
    "continue": _parse_continue,
    "opencode": _parse_opencode,
}

# ── Heuristic classifier ────────────────────────────────────


def _heuristic_read_only(name: str) -> bool | None:
    """Guess read-only from tool name using keyword heuristics.

    Split on _ and camelCase boundaries first, since \\b treats _ as a word char.
    """
    # Normalize: split snake_case and camelCase into separate words
    words = " ".join(re.split(r"[_\-]", name))
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", words)

    write_score = len(WRITE_KEYWORDS.findall(words))
    read_score = len(READ_KEYWORDS.findall(words))
    if read_score > write_score:
        return True
    if write_score > read_score:
        return False
    return None


# ── Main check ───────────────────────────────────────────────


def check(our_safe_tools: frozenset[str] | None = None) -> Report:
    """Fetch upstream tools and compare with our SAFE_TOOLS."""
    if our_safe_tools is None:
        from smartsplit.tool_registry import SAFE_TOOLS

        our_safe_tools = SAFE_TOOLS

    report = Report()

    for source in SOURCES:
        content = _fetch(source["url"])
        if content is None:
            report.errors.append(f"Failed to fetch {source['name']} ({source['url']})")
            continue

        parser = PARSERS.get(source["parser"])
        if parser is None:
            report.errors.append(f"Unknown parser: {source['parser']}")
            continue

        upstream_tools = parser(content)
        logger.info("  %s: found %d tools", source["name"], len(upstream_tools))

        for tool in upstream_tools:
            if tool.name in our_safe_tools:
                continue  # Already known
            if tool.name in IGNORED_TOOLS:
                continue  # Control-flow / meta tool, never anticipate

            # Determine read-only status
            is_ro = tool.read_only
            if is_ro is None:
                is_ro = _heuristic_read_only(tool.name)

            if is_ro is True:
                report.new_safe.append(tool)
            elif is_ro is False:
                report.new_write.append(tool)
            else:
                report.new_ambiguous.append(tool)

    return report


def _print_report(report: Report) -> None:
    if report.errors:
        print("\n⚠ Errors:")
        for err in report.errors:
            print(f"  - {err}")

    if report.new_safe:
        print("\n✅ New read-only tools (consider adding to SAFE_TOOLS):")
        for t in report.new_safe:
            print(f"  + {t.name:<30}  [{t.source}]  upstream_ro={t.read_only}")

    if report.new_ambiguous:
        print("\n❓ Ambiguous tools (manual review needed):")
        for t in report.new_ambiguous:
            print(f"  ? {t.name:<30}  [{t.source}]")

    if report.new_write:
        print("\n❌ Write tools (do NOT add):")
        for t in report.new_write:
            print(f"  - {t.name:<30}  [{t.source}]  upstream_ro={t.read_only}")

    if not report.new_safe and not report.new_ambiguous and not report.new_write:
        print("\n✅ All upstream read-only tools are already in SAFE_TOOLS.")


def _markdown_report(report: Report) -> str:
    """Markdown section for inclusion in Provider Watch issue."""
    lines = ["## Upstream Tool Check", ""]
    if report.errors:
        lines.append("### :warning: Fetch Errors")
        for err in report.errors:
            lines.append(f"- {err}")
        lines.append("")

    if report.new_safe:
        lines.append("### :white_check_mark: New read-only tools (consider adding to SAFE_TOOLS)")
        lines.append("")
        lines.append("| Tool | Source | Upstream RO |")
        lines.append("|------|--------|-------------|")
        for t in report.new_safe:
            lines.append(f"| `{t.name}` | {t.source} | {t.read_only} |")
        lines.append("")

    if report.new_ambiguous:
        lines.append("### :question: Ambiguous tools (manual review needed)")
        lines.append("")
        lines.append("| Tool | Source |")
        lines.append("|------|--------|")
        for t in report.new_ambiguous:
            lines.append(f"| `{t.name}` | {t.source} |")
        lines.append("")

    if not report.new_safe and not report.new_ambiguous and not report.errors:
        lines.append(":white_check_mark: All upstream read-only tools are already in SAFE_TOOLS.")
        lines.append("")

    return "\n".join(lines)


def _json_report(report: Report) -> str:
    return json.dumps(
        {
            "new_safe": [{"name": t.name, "source": t.source} for t in report.new_safe],
            "new_write": [{"name": t.name, "source": t.source} for t in report.new_write],
            "new_ambiguous": [{"name": t.name, "source": t.source} for t in report.new_ambiguous],
            "errors": report.errors,
        },
        indent=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check upstream repos for new safe tools")
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="Output JSON instead of text")
    fmt.add_argument("--markdown", action="store_true", help="Output Markdown section")
    args = parser.parse_args()

    logger.info("Fetching upstream tool definitions...")
    report = check()

    if args.json:
        print(_json_report(report))
    elif args.markdown:
        print(_markdown_report(report))
    else:
        _print_report(report)

    # Exit code: 1 if new tools found (useful for CI/cron alerting)
    if report.new_safe or report.new_ambiguous:
        sys.exit(1)


if __name__ == "__main__":
    main()
