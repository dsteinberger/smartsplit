"""Tests for smartsplit.triage.enrichment_prompts — domain-aware template resolution."""

from __future__ import annotations

import pytest

from smartsplit.triage.enrichment_prompts import (
    MULTI_PERSPECTIVE_FALLBACK,
    MULTI_PERSPECTIVE_TEMPLATES,
    PRE_ANALYSIS_FALLBACK,
    PRE_ANALYSIS_TEMPLATES,
    resolve_multi_perspective_prompt,
    resolve_pre_analysis_prompt,
)


class TestPreAnalysisResolver:
    def test_picks_code_template_when_code_domain_top(self):
        filled, specialized = resolve_pre_analysis_prompt("Refactor auth.py", [("code", 0.6), ("reasoning", 0.3)])
        assert specialized is True
        assert "Invariants to preserve" in filled
        assert "Refactor auth.py" in filled

    def test_picks_reasoning_template(self):
        filled, specialized = resolve_pre_analysis_prompt("Should we X?", [("reasoning", 0.5)])
        assert specialized is True
        assert "Possible theses" in filled

    def test_picks_math_template(self):
        filled, specialized = resolve_pre_analysis_prompt("Prove this lemma", [("math", 0.4)])
        assert specialized is True
        assert "Precise formulation" in filled

    def test_picks_writing_template(self):
        filled, specialized = resolve_pre_analysis_prompt("Draft an essay", [("writing", 0.5)])
        assert specialized is True
        assert "Implicit audience" in filled

    def test_picks_creative_template(self):
        filled, specialized = resolve_pre_analysis_prompt("Write a short story", [("creative", 0.5)])
        assert specialized is True
        assert "Genre / reference works" in filled

    def test_picks_factual_template(self):
        filled, specialized = resolve_pre_analysis_prompt("When did X happen?", [("factual", 0.5)])
        assert specialized is True
        assert "Core fact being asked" in filled

    def test_falls_back_when_no_matching_domain(self):
        filled, specialized = resolve_pre_analysis_prompt("Do the thing", [("extraction", 0.5), ("translation", 0.4)])
        assert specialized is False
        assert "Key concepts" in filled
        assert "Do the thing" in filled

    def test_falls_back_when_domains_empty(self):
        filled, specialized = resolve_pre_analysis_prompt("neutral request", [])
        assert specialized is False
        assert filled == PRE_ANALYSIS_FALLBACK.replace("{prompt}", "neutral request")

    def test_first_matching_domain_wins(self):
        # Domains arrive sorted by score desc; resolver walks in order
        filled, specialized = resolve_pre_analysis_prompt("ambiguous", [("code", 0.6), ("writing", 0.5)])
        assert specialized is True
        assert "Invariants to preserve" in filled
        assert "Implicit audience" not in filled

    def test_prompt_is_injected_once(self):
        prompt = "unique-marker-xyz"
        filled, _ = resolve_pre_analysis_prompt(prompt, [("code", 0.5)])
        assert filled.count(prompt) == 1


class TestMultiPerspectiveResolver:
    def test_picks_code_template(self):
        filled, specialized = resolve_multi_perspective_prompt("Postgres vs MongoDB", [("code", 0.5)])
        assert specialized is True
        assert "**Claim**" in filled
        assert "**Evidence**" in filled
        assert "Postgres vs MongoDB" in filled

    def test_picks_reasoning_template(self):
        filled, specialized = resolve_multi_perspective_prompt("Kant vs Mill", [("reasoning", 0.5)])
        assert specialized is True
        assert "Steel-man" in filled

    def test_picks_creative_template(self):
        filled, specialized = resolve_multi_perspective_prompt(
            "First person vs third person narrator", [("creative", 0.5)]
        )
        assert specialized is True
        assert "Effect on the reader" in filled

    def test_picks_factual_template(self):
        filled, specialized = resolve_multi_perspective_prompt("Who discovered America", [("factual", 0.5)])
        assert specialized is True
        assert "competing answers" in filled.lower()

    def test_falls_back_when_no_matching_domain(self):
        filled, specialized = resolve_multi_perspective_prompt("Pick one", [("math", 0.5), ("writing", 0.3)])
        # math/writing have no multi_perspective template → fallback
        assert specialized is False
        assert filled == MULTI_PERSPECTIVE_FALLBACK.replace("{prompt}", "Pick one")

    def test_falls_back_when_domains_empty(self):
        filled, specialized = resolve_multi_perspective_prompt("Pick one", [])
        assert specialized is False
        assert "claim, evidence, cost or risk" in filled


class TestTemplateHygiene:
    """Guardrails to catch missing / extra placeholders when editing templates."""

    @pytest.mark.parametrize("name,template", list(PRE_ANALYSIS_TEMPLATES.items()))
    def test_pre_analysis_templates_have_prompt_placeholder(self, name, template):
        assert "{prompt}" in template, f"pre_analysis template '{name}' missing {{prompt}}"

    @pytest.mark.parametrize("name,template", list(MULTI_PERSPECTIVE_TEMPLATES.items()))
    def test_multi_perspective_templates_have_prompt_placeholder(self, name, template):
        assert "{prompt}" in template, f"multi_perspective template '{name}' missing {{prompt}}"

    def test_fallbacks_have_prompt_placeholder(self):
        assert "{prompt}" in PRE_ANALYSIS_FALLBACK
        assert "{prompt}" in MULTI_PERSPECTIVE_FALLBACK

    @pytest.mark.parametrize(
        "template",
        list(PRE_ANALYSIS_TEMPLATES.values())
        + list(MULTI_PERSPECTIVE_TEMPLATES.values())
        + [PRE_ANALYSIS_FALLBACK, MULTI_PERSPECTIVE_FALLBACK],
    )
    def test_no_unfilled_placeholders_other_than_prompt(self, template):
        # After removing {prompt}, no other { placeholder } should remain
        stripped = template.replace("{prompt}", "")
        # Simple check: no remaining `{foo}` pattern
        import re

        leftovers = re.findall(r"\{[a-zA-Z_]+\}", stripped)
        assert leftovers == [], f"Unfilled placeholders: {leftovers}"
