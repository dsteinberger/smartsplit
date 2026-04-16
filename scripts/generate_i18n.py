#!/usr/bin/env python3
"""Generate multilingual keyword translations for SmartSplit domain detection.

Usage:
    python scripts/generate_i18n.py              # Regenerate all languages
    python scripts/generate_i18n.py --add hi     # Add Hindi
    python scripts/generate_i18n.py --regen fr   # Regenerate French only

Requires: pip install deep-translator  (dev dependency only)
Output:   smartsplit/triage/i18n_keywords.py
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("Error: deep-translator is required. Install with: pip install deep-translator")
    sys.exit(1)

# ── Source keywords (English) ────────────────────────────────
# These are the canonical keywords to translate. Keep in sync with
# planner._DOMAIN_KEYWORDS and proxy._COMPARISON_KEYWORDS / _ANALYSIS_KEYWORDS.

DOMAIN_KEYWORDS_EN: dict[str, list[str]] = {
    "code": [
        "function",
        "class",
        "variable",
        "bug",
        "debug",
        "implement",
        "refactor",
        "compile",
        "runtime",
        "syntax",
        "endpoint",
        "database",
        "algorithm",
        "data structure",
        "deploy",
    ],
    "math": [
        "calculate",
        "compute",
        "equation",
        "formula",
        "integral",
        "derivative",
        "probability",
        "statistics",
        "matrix",
        "vector",
        "algebra",
        "geometry",
        "prove",
        "theorem",
        "logarithm",
        "factorial",
        "sum of",
        "product of",
        "solve",
        "optimize",
        "minimize",
        "maximize",
        "convergence",
    ],
    "creative": [
        "story",
        "poem",
        "novel",
        "chapter",
        "creative",
        "imagine",
        "campaign",
        "narrative",
        "character",
        "dialogue",
        "screenplay",
        "lyrics",
    ],
    "translation": [
        "translate",
        "translation",
    ],
    "extraction": [
        "extract",
        "parse",
        "scrape",
        "list all",
        "find all",
        "identify",
        "enumerate",
        "format as",
        "convert to csv",
        "convert to json",
        "structured data",
    ],
    "factual": [
        "what is",
        "who is",
        "when did",
        "where is",
        "how many",
        "define",
        "definition of",
        "meaning of",
        "capital of",
        "population of",
        "what year",
        "is it true",
        "fact check",
    ],
    "web_search": [
        "search",
        "look up",
        "find online",
        "latest news",
        "current",
        "recent",
        "today",
        "on the web",
        "on the internet",
        "online",
    ],
    "reasoning": [
        "analyze",
        "compare",
        "evaluate",
        "pros and cons",
        "advantages",
        "disadvantages",
        "strategy",
        "recommend",
        "which is better",
        "explain why",
        "root cause",
        "diagnose",
        "assess",
        "critique",
    ],
    "summarize": [
        "summarize",
        "summary",
        "brief overview",
        "key points",
        "main takeaways",
        "condense",
        "shorten",
    ],
    "writing": [
        "draft a report",
        "draft an email",
        "write a letter",
        "write a memo",
        "write a proposal",
        "report",
        "memo",
        "letter",
        "proposal",
        "presentation",
        "article",
        "blog post",
        "essay",
    ],
}

COMPARISON_KEYWORDS_EN = [
    "compare",
    "which is better",
    "pros and cons",
    "advantages and disadvantages",
]

ANALYSIS_KEYWORDS_EN = [
    "refactor",
    "review",
    "analyze",
    "audit",
    "explain",
    "what's wrong",
    "improve",
    "optimize",
    "diagnose",
]

# Default languages to generate
DEFAULT_LANGUAGES = ["fr", "es", "pt", "de", "zh-CN", "ja", "ko", "ru"]

# Map deep-translator codes to our short codes
LANG_CODE_MAP = {"zh-CN": "zh"}


def translate_keywords(keywords: list[str], target_lang: str) -> list[str]:
    """Translate a list of keywords to the target language."""
    translator = GoogleTranslator(source="en", target=target_lang)
    translated = []
    for kw in keywords:
        try:
            result = translator.translate(kw)
            if result and result.strip():
                translated.append(result.strip().lower())
        except Exception as e:
            print(f"  Warning: failed to translate '{kw}' → {target_lang}: {e}")
            continue
    return translated


def generate_for_language(lang_code: str) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Generate all keyword translations for a single language."""
    short_code = LANG_CODE_MAP.get(lang_code, lang_code)
    print(f"Generating translations for: {short_code} ({lang_code})")

    domains: dict[str, list[str]] = {}
    for domain, keywords in DOMAIN_KEYWORDS_EN.items():
        print(f"  Domain: {domain} ({len(keywords)} keywords)")
        domains[domain] = translate_keywords(keywords, lang_code)

    print(f"  Comparison keywords ({len(COMPARISON_KEYWORDS_EN)})")
    comparison = translate_keywords(COMPARISON_KEYWORDS_EN, lang_code)

    print(f"  Analysis keywords ({len(ANALYSIS_KEYWORDS_EN)})")
    analysis = translate_keywords(ANALYSIS_KEYWORDS_EN, lang_code)

    return domains, comparison, analysis


def format_string_list(items: list[str], indent: int = 12) -> str:
    """Format a list of strings as Python source code."""
    pad = " " * indent
    lines = [f'{pad}"{item}",' for item in items]
    return "\n".join(lines)


def format_output(
    domain_data: dict[str, dict[str, list[str]]],
    comparison_data: dict[str, list[str]],
    analysis_data: dict[str, list[str]],
) -> str:
    """Format all data as the i18n_keywords.py module source."""
    lines = [
        '"""Multilingual keyword translations for domain detection.',
        "",
        "Generated by scripts/generate_i18n.py — do not edit manually.",
        "To add a language: python scripts/generate_i18n.py --add <code>",
        f"Supported: {', '.join(sorted(domain_data.keys()))}",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "DOMAIN_KEYWORDS_I18N: dict[str, dict[str, list[str]]] = {",
    ]

    for lang in sorted(domain_data.keys()):
        lines.append(f'    "{lang}": {{')
        for domain, keywords in domain_data[lang].items():
            lines.append(f'        "{domain}": [')
            for kw in keywords:
                lines.append(f'            "{kw}",')
            lines.append("        ],")
        lines.append("    },")
    lines.append("}")
    lines.append("")

    lines.append("COMPARISON_KEYWORDS_I18N: dict[str, list[str]] = {")
    for lang in sorted(comparison_data.keys()):
        items = ", ".join(f'"{kw}"' for kw in comparison_data[lang])
        lines.append(f'    "{lang}": [{items}],')
    lines.append("}")
    lines.append("")

    lines.append("ANALYSIS_KEYWORDS_I18N: dict[str, list[str]] = {")
    for lang in sorted(analysis_data.keys()):
        items = ", ".join(f'"{kw}"' for kw in analysis_data[lang])
        lines.append(f'    "{lang}": [{items}],')
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _load_existing(
    output_path: Path,
) -> tuple[
    dict[str, dict[str, list[str]]],
    dict[str, list[str]],
    dict[str, list[str]],
]:
    """Load existing i18n data from the generated module to preserve other languages."""
    domain_data: dict[str, dict[str, list[str]]] = {}
    comparison_data: dict[str, list[str]] = {}
    analysis_data: dict[str, list[str]] = {}

    if not output_path.exists():
        return domain_data, comparison_data, analysis_data

    try:
        source = output_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            name = getattr(node.targets[0], "id", None)
            if name == "DOMAIN_KEYWORDS_I18N":
                domain_data = ast.literal_eval(source[node.value.col_offset :].split("\n}")[0] + "\n}")
            elif name == "COMPARISON_KEYWORDS_I18N":
                comparison_data = ast.literal_eval(source[node.value.col_offset :].split("\n}")[0] + "\n}")
            elif name == "ANALYSIS_KEYWORDS_I18N":
                analysis_data = ast.literal_eval(source[node.value.col_offset :].split("\n}")[0] + "\n}")
    except (SyntaxError, ValueError) as e:
        print(f"Warning: could not parse existing {output_path.name}, regenerating from scratch: {e}")
        return {}, {}, {}

    return domain_data, comparison_data, analysis_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate i18n keywords for SmartSplit")
    parser.add_argument("--add", metavar="LANG", help="Add a single language (e.g., hi for Hindi)")
    parser.add_argument("--regen", metavar="LANG", help="Regenerate a single language")
    args = parser.parse_args()

    output_path = Path(__file__).parent.parent / "smartsplit" / "triage" / "i18n_keywords.py"

    # For --add and --regen, load existing data first to preserve other languages
    if args.add or args.regen:
        domain_data, comparison_data, analysis_data = _load_existing(output_path)
        languages = [args.add or args.regen]
        action = "Adding" if args.add else "Regenerating"
        print(f"{action} language: {languages[0]} (preserving {len(domain_data)} existing languages)")
    else:
        domain_data = {}
        comparison_data = {}
        analysis_data = {}
        languages = DEFAULT_LANGUAGES

    for lang in languages:
        short = LANG_CODE_MAP.get(lang, lang)
        domains, comparison, analysis = generate_for_language(lang)
        domain_data[short] = domains
        comparison_data[short] = comparison
        analysis_data[short] = analysis

    source = format_output(domain_data, comparison_data, analysis_data)
    output_path.write_text(source, encoding="utf-8")
    print(f"\nWritten to: {output_path}")
    print("Review the output, then commit.")


if __name__ == "__main__":
    main()
