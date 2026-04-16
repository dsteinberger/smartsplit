#!/usr/bin/env python3
"""Decide whether the Provider Watch report contains real actionable items.

Prints "true" or "false" on stdout (for GITHUB_OUTPUT).

A report is considered actionable if ANY of:
- The "Action Items" section contains at least one bullet starting with a
  concrete verb (add, update, bump, switch, integrate, deprecate, remove,
  replace, fix, raise, lower, set, change, upgrade, downgrade).
  Bullets starting with vague verbs (monitor, consider, evaluate, review,
  watch, track, observe, keep) are ignored.
- The "Upstream Tool Check" section reports new read-only or ambiguous tools.
- A provider in the health-check table failed with a non-transient status
  (anything other than 429 rate-limited).

Usage:
    python scripts/report_has_actions.py report.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ACTIONABLE_VERBS = {
    "add",
    "update",
    "bump",
    "switch",
    "integrate",
    "deprecate",
    "remove",
    "replace",
    "fix",
    "raise",
    "lower",
    "set",
    "change",
    "upgrade",
    "downgrade",
    "enable",
    "disable",
    "drop",
}

VAGUE_VERBS = {
    "monitor",
    "consider",
    "evaluate",
    "review",
    "watch",
    "track",
    "observe",
    "keep",
    "investigate",
    "explore",
    "assess",
    "check",
    "look",
}

_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_SECTION_RE = re.compile(r"^##\s+(.*?)\s*$")


def _section(text: str, title: str) -> str:
    """Extract the body of a `## title` section until the next `## ` header."""
    lines = text.splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        m = _SECTION_RE.match(line)
        if m:
            if inside:
                break
            inside = m.group(1).strip().lower() == title.lower()
            continue
        if inside:
            out.append(line)
    return "\n".join(out)


def _first_word(bullet: str) -> str:
    word = bullet.strip().split(" ", 1)[0].strip("*_`.,:;")
    return word.lower()


def _action_items_actionable(report: str) -> bool:
    section = _section(report, "Action Items")
    if not section.strip():
        return False
    if "no action needed" in section.lower():
        return False
    for line in section.splitlines():
        m = _BULLET_RE.match(line)
        if not m:
            continue
        verb = _first_word(m.group(1))
        if verb in ACTIONABLE_VERBS:
            return True
        if verb in VAGUE_VERBS:
            continue
        # Unknown verb — be conservative, treat as non-actionable noise.
    return False


def _upstream_tools_actionable(report: str) -> bool:
    section = _section(report, "Upstream Tool Check")
    return "New read-only tools" in section or "Ambiguous tools" in section


def _health_check_actionable(report: str) -> bool:
    """Return True if any provider FAILed with something other than 429."""
    for line in report.splitlines():
        if "FAIL" not in line:
            continue
        # Transient rate-limit, ignore.
        if "429" in line or "rate-limit" in line.lower():
            continue
        return True
    return False


def has_actions(report: str) -> bool:
    return _action_items_actionable(report) or _upstream_tools_actionable(report) or _health_check_actionable(report)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: report_has_actions.py <report.md>", file=sys.stderr)
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8")
    result = has_actions(text)
    print("true" if result else "false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
