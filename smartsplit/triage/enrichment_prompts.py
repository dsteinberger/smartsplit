"""Domain-specific prompts for pre_analysis and multi_perspective enrichment.

Kept separate from ``enrichment.py`` to make the template catalog easy to edit
and review. Each template forces a structured markdown output with short,
specific sections so the brain can cite or ignore each point independently —
instead of swimming in a generic paragraph.

All templates are in English because upstream LLMs are better calibrated on
English instruction-following, especially for strict output formats.
"""

from __future__ import annotations

# ── pre_analysis templates ──────────────────────────────────

PRE_ANALYSIS_TEMPLATES: dict[str, str] = {
    "code": """\
You are a senior software engineer. Analyze this engineering task BEFORE the \
main assistant answers. Your output will be injected as context, so be \
specific to THIS request and avoid generic advice.

Output format (markdown, short bullets, no fluff):
## Invariants to preserve
## Likely edge cases
## Couplings to check
## Dette / risks a naive refactor would introduce

Request: {prompt}""",
    "reasoning": """\
You are a reasoning coach. Extract the decision structure BEFORE the main \
assistant answers. Be concrete — name the actual theses in play.

Output format:
## Possible theses
## Facts needed to decide
## Hidden assumptions
## Likely blind spots

Request: {prompt}""",
    "math": """\
Analyze this math / technical-problem request BEFORE the main assistant answers.

Output format:
## Precise formulation
## Applicable method(s)
## Classical traps on this problem class
## Consistency checks to run on the answer

Request: {prompt}""",
    "writing": """\
Analyze this writing task BEFORE the main assistant answers.

Output format:
## Implicit audience
## Format / tone constraints
## Points that MUST be covered
## Classical traps for this type of writing

Request: {prompt}""",
    "creative": """\
Analyze this creative request BEFORE the main assistant answers. Help the \
main assistant avoid generic output.

Output format:
## Genre / reference works
## Tone, mood, register expected
## Concrete elements that would make it feel original
## Clichés to avoid on this theme

Request: {prompt}""",
    "factual": """\
Analyze this factual question BEFORE the main assistant answers. Focus on \
what could make the answer wrong or shallow.

Output format:
## Core fact being asked
## Related concepts often confused with it
## Common misconceptions
## Boundary conditions / caveats to mention

Request: {prompt}""",
}

PRE_ANALYSIS_FALLBACK = """\
Analyze this request BEFORE the main assistant answers. Be specific to the \
request, not generic.

Output format (markdown):
## Key concepts
## Constraints
## Likely pitfalls
## Relevant background

Request: {prompt}"""


# ── multi_perspective templates ─────────────────────────────

MULTI_PERSPECTIVE_TEMPLATES: dict[str, str] = {
    "code": """\
Compare the options below for a technical decision. Be factual; cite \
concrete signals, not vibes.

For each option use this exact structure:
### <option name>
- **Claim**: what it's best at (one sentence)
- **Evidence**: concrete signal (perf numbers, ecosystem size, known prod users)
- **Cost**: effort, lock-in, learning curve
- **Who it fits**: specific team or project profile

End with a single-line conditional recommendation \
(e.g. "If low-latency matters most → A; if team is small → B").

Question: {prompt}""",
    "reasoning": """\
Lay out the main positions on this question. Steel-man each one — no \
strawmen, no "it depends" cop-outs.

For each position:
### <position name>
- **Claim**: the core thesis in one sentence
- **Strongest argument for**
- **Strongest argument against**
- **When it holds / when it breaks**

End with: "The disagreement really hinges on: <the actual key factor>"

Question: {prompt}""",
    "creative": """\
Compare the creative options or directions at stake.

For each option:
### <option name>
- **Effect on the reader/viewer**
- **Tone it creates**
- **Works that successfully took this path**
- **Risk if pushed too far**

End with a one-line note on which path fits which intent.

Question: {prompt}""",
    "factual": """\
Lay out the competing answers or interpretations on this factual question.

For each:
### <answer / interpretation>
- **Claim**
- **Evidence supporting it** (sources, dates, consensus)
- **Evidence against it**
- **Where it is typically accepted / contested**

End with: "Best-supported answer today: <X>, with caveats: <Y>"

Question: {prompt}""",
}

MULTI_PERSPECTIVE_FALLBACK = """\
List the main options or positions on this question.

For each: claim, evidence, cost or risk, who/when it fits.
End with a single-line conditional recommendation, not "it depends".

Question: {prompt}"""


# ── resolvers ───────────────────────────────────────────────


def _resolve(
    templates: dict[str, str],
    fallback: str,
    prompt: str,
    domains: list[tuple[str, float]],
) -> tuple[str, bool]:
    """Pick the template matching the strongest domain, else fallback.

    Returns ``(filled_prompt, used_specialized_template)``. The boolean lets
    the caller upgrade ``Complexity`` when a specialized template was used.
    """
    for domain, _score in domains:
        template = templates.get(domain)
        if template is not None:
            return template.replace("{prompt}", prompt), True
    return fallback.replace("{prompt}", prompt), False


def resolve_pre_analysis_prompt(prompt: str, domains: list[tuple[str, float]]) -> tuple[str, bool]:
    """Return the best pre_analysis prompt for ``prompt`` given detected ``domains``."""
    return _resolve(PRE_ANALYSIS_TEMPLATES, PRE_ANALYSIS_FALLBACK, prompt, domains)


def resolve_multi_perspective_prompt(prompt: str, domains: list[tuple[str, float]]) -> tuple[str, bool]:
    """Return the best multi_perspective prompt for ``prompt`` given detected ``domains``."""
    return _resolve(MULTI_PERSPECTIVE_TEMPLATES, MULTI_PERSPECTIVE_FALLBACK, prompt, domains)
