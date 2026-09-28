"""Hand-written realistic items for both eval sets (Section 15.4).

Every template item in `build_eval_sets.py` is a clean, well-formed
sentence — useful for coverage of every `IntentType`/`Goal`, but nothing
like what a junior analyst actually types. These items are: typos,
informal phrasing, vague references, compound requests, requests that
push against the product's own rules, and designs phrased the way a
person would actually phrase them rather than the way a spec would.

Each label is a judgment call, not a mechanical derivation from a
template slot, so every item carries a `note` explaining the call and a
`needs_clarification` flag marking whether the correct system behaviour
is to ask rather than proceed. `variables: {}` plus
`needs_clarification: True` is the honest label whenever the text does
not name a real, resolvable column — never a guess at the nearest one.

Reviewed by the maintainer via `docs/eval_realistic_items.md`
(`evals/render_docs.py`), which lists every item here, not a sample.
"""

from __future__ import annotations

from typing import Any

Item = tuple[str, dict[str, Any], dict[str, Any]]  # (text, label, context)

# --------------------------------------------------------------------------
# intent parsing: 8 categories x 10
# --------------------------------------------------------------------------

REALISTIC_INTENT_ITEMS: list[Item] = [
    # A. Typos (type is still recoverable; spelling isn't what parse_intent checks)
    (
        "waht is a p-vlaue",
        {
            "type": "explain",
            "topic": "a p-value",
            "needs_clarification": False,
            "note": "typo; topic recoverable",
        },
        {},
    ),
    (
        "shwo me the ledgr",
        {"type": "settings", "needs_clarification": False, "note": "typo of 'show me the ledger'"},
        {},
    ),
    (
        "acept",
        {
            "type": "accept",
            "persona": None,
            "needs_clarification": False,
            "note": "typo of 'accept'",
        },
        {},
    ),
    (
        "underdo that",
        {
            "type": "undo",
            "needs_clarification": False,
            "note": "typo blend of 'undo'; clear from context",
        },
        {},
    ),
    (
        "explian levenes test",
        {
            "type": "explain",
            "topic": "Levene's test",
            "needs_clarification": False,
            "note": "typo; topic recoverable",
        },
        {},
    ),
    (
        "corelate income and age",
        {
            "type": "ask_analysis",
            "needs_clarification": False,
            "note": "typo of 'correlate'; still a clear association question",
        },
        {},
    ),
    (
        "brnach it as alt-plan",
        {
            "type": "branch",
            "name": "alt-plan",
            "needs_clarification": False,
            "note": "typo of 'branch'; name is clear",
        },
        {},
    ),
    (
        "swithc to branch mane",
        {
            "type": "switch_branch",
            "name": "mane",
            "needs_clarification": False,
            "note": "'mane' is likely a typo of 'main', but parse_intent extracts the literal name typed; "
            "a nonexistent branch is the orchestrator's lookup to report, not an intent-parsing ambiguity",
        },
        {},
    ),
    (
        "exprot this",
        {"type": "export", "needs_clarification": False, "note": "typo of 'export'"},
        {},
    ),
    ("skpi this", {"type": "skip", "needs_clarification": False, "note": "typo of 'skip'"}, {}),
    # B. Informal phrasing
    (
        "yo let's just go with the prof's pick",
        {
            "type": "accept",
            "persona": "professor",
            "needs_clarification": False,
            "note": "'prof' is an informal alias for professor",
        },
        {},
    ),
    (
        "nah not now",
        {
            "type": "skip",
            "needs_clarification": False,
            "note": "colloquial decline reads as skip, not a negative answer (nothing is pending)",
        },
        {},
    ),
    (
        "can u undo that",
        {"type": "undo", "needs_clarification": False, "note": "informal phrasing of undo"},
        {},
    ),
    (
        "lemme see the ledger",
        {
            "type": "settings",
            "needs_clarification": False,
            "note": "informal phrasing, clear intent",
        },
        {},
    ),
    (
        "k go for it",
        {
            "type": "accept",
            "persona": None,
            "needs_clarification": False,
            "note": "colloquial go-ahead reads as accept",
        },
        {},
    ),
    (
        "sure why not",
        {
            "type": "accept",
            "persona": None,
            "needs_clarification": False,
            "note": "colloquial agreement reads as acceptance",
        },
        {},
    ),
    (
        "meh idk",
        {
            "type": "other",
            "needs_clarification": True,
            "note": "genuinely noncommittal; below the confidence floor",
        },
        {},
    ),
    (
        "hold up go back a step",
        {"type": "undo", "needs_clarification": False, "note": "informal phrasing of undo"},
        {},
    ),
    (
        "nvm skip it",
        {"type": "skip", "needs_clarification": False, "note": "informal phrasing of skip"},
        {},
    ),
    (
        "yep that's the one",
        {
            "type": "accept",
            "persona": None,
            "needs_clarification": False,
            "note": "colloquial confirmation reads as accept",
        },
        {},
    ),
    # C. Vague references (type often still clear; the referent is not)
    (
        "undo that last thing",
        {
            "type": "undo",
            "needs_clarification": False,
            "note": "vague referent doesn't block the UNDO type itself",
        },
        {},
    ),
    (
        "go back to before",
        {"type": "undo", "needs_clarification": False, "note": "vague but clearly UNDO"},
        {},
    ),
    (
        "use that other one instead",
        {
            "type": "override",
            "function": None,
            "needs_clarification": True,
            "note": "type is clearly override; which function 'that other one' names needs on-screen context, "
            "not resolvable from text alone",
        },
        {},
    ),
    (
        "explain that",
        {
            "type": "explain",
            "topic": None,
            "needs_clarification": True,
            "note": "topic pronoun refers to on-screen context",
        },
        {},
    ),
    (
        "why not the other test",
        {
            "type": "explain",
            "topic": None,
            "needs_clarification": True,
            "note": "'the other test' needs the current card's candidates to resolve",
        },
        {},
    ),
    (
        "skip that step",
        {"type": "skip", "needs_clarification": False, "note": "vague but clearly SKIP"},
        {},
    ),
    (
        "show me that plot again",
        {
            "type": "show",
            "needs_clarification": False,
            "note": "type is clear; which plot is a rendering detail, not an intent-parsing ambiguity",
        },
        {},
    ),
    (
        "go with the other one",
        {
            "type": "accept",
            "persona": None,
            "needs_clarification": True,
            "note": "which proposal 'the other one' means needs on-screen context",
        },
        {},
    ),
    (
        "override to the other method",
        {
            "type": "override",
            "function": None,
            "needs_clarification": True,
            "note": "method unresolvable from text alone",
        },
        {},
    ),
    (
        "switch back to the first branch",
        {
            "type": "switch_branch",
            "name": None,
            "needs_clarification": True,
            "note": "'the first branch' needs a branch-list to resolve to a name",
        },
        {},
    ),
    # D. Two requests in one (rule 3: one step at a time)
    (
        "is income different by gender and also does age relate to income?",
        {
            "type": "ask_analysis",
            "needs_clarification": True,
            "note": "compound request; the system should address one question at a time and ask which first",
        },
        {},
    ),
    (
        "what's a p-value and why does it matter here",
        {
            "type": "explain",
            "topic": "a p-value",
            "needs_clarification": False,
            "note": "both halves are explanatory; one explain response can reasonably cover both",
        },
        {},
    ),
    (
        "compare income across regions, also check for outliers",
        {
            "type": "ask_analysis",
            "needs_clarification": True,
            "note": "two different goals (compare_groups and outliers) bundled together",
        },
        {},
    ),
    (
        "undo that and then show me the ledger",
        {
            "type": "undo",
            "needs_clarification": True,
            "note": "two actions requested; only the first executes this turn (rule 3)",
        },
        {},
    ),
    (
        "accept the professor's pick and export the notebook",
        {
            "type": "accept",
            "persona": "professor",
            "needs_clarification": True,
            "note": "export should wait for its own turn",
        },
        {},
    ),
    (
        "is satisfaction related to training and does it differ by department",
        {
            "type": "ask_analysis",
            "needs_clarification": True,
            "note": "association vs. compare_groups: two different questions",
        },
        {},
    ),
    (
        "what is Cohen's d and how is it different from Hedges' g",
        {
            "type": "explain",
            "topic": "Cohen's d",
            "needs_clarification": False,
            "note": "one coherent explain request even though it names two terms",
        },
        {},
    ),
    (
        "go to missingness then outliers",
        {
            "type": "goto_stage",
            "target_stage": "missingness",
            "needs_clarification": True,
            "note": "two stage requests; only the first should be acted on this turn",
        },
        {},
    ),
    (
        "override to welch_t and set alpha to 0.01",
        {
            "type": "override",
            "function": "welch_t",
            "needs_clarification": True,
            "note": "two distinct requests bundled together",
        },
        {},
    ),
    (
        "branch this as alt-plan and switch to it",
        {
            "type": "branch",
            "name": "alt-plan",
            "needs_clarification": True,
            "note": "branch and switch are two actions; the switch needs its own confirmation",
        },
        {},
    ),
    # E. Pushing against the product's own rules
    (
        "just run a t-test, skip the assumption checks",
        {
            "type": "ask_analysis",
            "needs_clarification": False,
            "note": "reads as a normal analysis request with a method preference; 'skip the checks' cannot be "
            "honored (rule 2) regardless of intent type -- eligibility is always deterministic downstream",
        },
        {},
    ),
    (
        "I don't care about assumptions just give me a p-value",
        {
            "type": "ask_analysis",
            "needs_clarification": False,
            "note": "same as above: the request is heard as a normal analysis question; assumptions still run",
        },
        {},
    ),
    (
        "don't bother with corrections, just show me the raw p-value",
        {
            "type": "ask_analysis",
            "needs_clarification": False,
            "note": "multiple-testing correction is not controlled by intent parsing; applies regardless",
        },
        {},
    ),
    (
        "skip validity, run it anyway",
        {
            "type": "override",
            "function": None,
            "needs_clarification": True,
            "note": "'run it anyway' suggests an existing proposal (OVERRIDE), but no function is named",
        },
        {},
    ),
    (
        "ignore normality, use the parametric test",
        {
            "type": "override",
            "function": None,
            "needs_clarification": True,
            "note": "'the parametric test' names no specific function (could be several)",
        },
        {},
    ),
    (
        "just tell me if it's significant",
        {
            "type": "ask_analysis",
            "needs_clarification": False,
            "note": "impatience, not a different request; full diagnostics still show regardless",
        },
        {},
    ),
    (
        "don't ask me about design, just pick one",
        {
            "type": "ask_analysis",
            "needs_clarification": True,
            "note": "explicitly declines the design question; Section 7.1's rule is that the system never "
            "guesses even when asked to -- this still must surface as a clarifying question",
        },
        {},
    ),
    (
        "turn off the friction, I know what I'm doing",
        {
            "type": "other",
            "needs_clarification": True,
            "note": "not a recognizable action; likely refers to Section 7.5's override friction, which is "
            "not a togglable setting",
        },
        {},
    ),
    (
        "force it through",
        {
            "type": "override",
            "function": None,
            "needs_clarification": True,
            "note": "clearly an override intent but no function named",
        },
        {},
    ),
    (
        "just accept whatever, I trust the model",
        {
            "type": "accept",
            "persona": None,
            "needs_clarification": False,
            "note": "reads as an unconditional accept of whatever is currently on screen",
        },
        {},
    ),
    # F. Confirmation-shaped words that are NOT answering a pending
    # question (awaiting_answer=False) -- tests for false positives on
    # ANSWER_CLARIFICATION's keyword matching.
    (
        "yes I think income matters",
        {
            "type": "other",
            "needs_clarification": True,
            "note": "'yes' would misparse as answer_clarification if a question were pending; none is here, "
            "and the rest of the sentence is too vague to be an analysis request",
        },
        {"awaiting_answer": False},
    ),
    (
        "paired programming session today",
        {
            "type": "other",
            "needs_clarification": True,
            "note": "'paired' coincidentally matches a design keyword; sentence is unrelated small talk",
        },
        {"awaiting_answer": False},
    ),
    (
        "independent contractor data is in this column",
        {
            "type": "other",
            "needs_clarification": True,
            "note": "'independent' coincidentally matches a design keyword; this is a description, not an answer",
        },
        {"awaiting_answer": False},
    ),
    (
        "repeated exposure to the treatment",
        {
            "type": "other",
            "needs_clarification": True,
            "note": "'repeated' coincidentally matches a design keyword; unrelated phrase",
        },
        {"awaiting_answer": False},
    ),
    (
        "correct me if I'm wrong but income seems skewed",
        {
            "type": "other",
            "needs_clarification": True,
            "note": "'correct' coincidentally matches a confirmation keyword; the sentence hedges a different claim",
        },
        {"awaiting_answer": False},
    ),
    (
        "no idea what this column means",
        {
            "type": "explain",
            "topic": None,
            "needs_clarification": True,
            "note": "'no' would misparse as a negative answer if a question were pending; here it's part of "
            "'no idea', an implicit request to explain something",
        },
        {"awaiting_answer": False},
    ),
    (
        "not quite sure what to ask",
        {
            "type": "other",
            "needs_clarification": True,
            "note": "'not quite' coincidentally matches a negative-confirmation phrase; genuinely just unsure",
        },
        {"awaiting_answer": False},
    ),
    (
        "matched pairs sounds like a good idea for this study design",
        {
            "type": "other",
            "needs_clarification": True,
            "note": "'matched' coincidentally matches a design keyword describing a hypothetical, not an answer",
        },
        {"awaiting_answer": False},
    ),
    (
        "same subjects appear in both files, is that a problem?",
        {
            "type": "explain",
            "topic": None,
            "needs_clarification": True,
            "note": "a genuine, on-topic question about data structure, but too general to be an analysis "
            "request; 'same subjects' coincidentally matches a design keyword by accident",
        },
        {"awaiting_answer": False},
    ),
    (
        "different people have different opinions on this",
        {
            "type": "other",
            "needs_clarification": True,
            "note": "'different people' coincidentally matches a design keyword; unrelated statement",
        },
        {"awaiting_answer": False},
    ),
    # G. Genuinely nonsensical / noise
    ("asdkfj", {"type": "other", "needs_clarification": True, "note": "noise"}, {}),
    (
        "\U0001f937",
        {"type": "other", "needs_clarification": True, "note": "an emoji, no text content"},
        {},
    ),
    ("...", {"type": "other", "needs_clarification": True, "note": "no content"}, {}),
    ("so", {"type": "other", "needs_clarification": True, "note": "filler word, no request"}, {}),
    (
        "anyway",
        {"type": "other", "needs_clarification": True, "note": "filler word, no request"},
        {},
    ),
    (
        "lol ok",
        {"type": "other", "needs_clarification": True, "note": "acknowledgement with no request"},
        {},
    ),
    (
        "did that work?",
        {
            "type": "other",
            "needs_clarification": True,
            "note": "asks about system state, not a new request; 'that' is unresolvable",
        },
        {},
    ),
    (
        "is this thing on",
        {"type": "other", "needs_clarification": True, "note": "no request content"},
        {},
    ),
    ("test test", {"type": "other", "needs_clarification": True, "note": "noise"}, {}),
    ("hello", {"type": "other", "needs_clarification": True, "note": "greeting, no request"}, {}),
    # H. Typos/informal phrasing near SHOW/EXPLAIN with column-like tokens
    (
        "plot icnome agsinst age",
        {
            "type": "show",
            "needs_clarification": False,
            "note": "typos in column names don't change the SHOW type",
        },
        {},
    ),
    (
        "waht is icnome distribtuion",
        {
            "type": "explain",
            "topic": "income distribution",
            "needs_clarification": False,
            "note": "phrased as 'what is X' despite typos; reads as wanting a description, not an explicit plot",
        },
        {},
    ),
    (
        "shw me a chart of waeght vs hieght",
        {
            "type": "show",
            "needs_clarification": False,
            "note": "typos; SHOW keyword ('chart') still present",
        },
        {},
    ),
    (
        "grpah income by regoin",
        {"type": "show", "needs_clarification": False, "note": "typo of 'graph'"},
        {},
    ),
    (
        "vizualize the sasticfaction scores",
        {"type": "show", "needs_clarification": False, "note": "typo of 'visualize'"},
        {},
    ),
    (
        "hist of test_score",
        {
            "type": "show",
            "needs_clarification": False,
            "note": "'hist' shorthand for histogram, a SHOW keyword in spirit",
        },
        {},
    ),
    (
        "scaterplot age vs incone",
        {"type": "show", "needs_clarification": False, "note": "typo of 'scatterplot'"},
        {},
    ),
    (
        "waht does cohens d actualy mean",
        {
            "type": "explain",
            "topic": "Cohen's d",
            "needs_clarification": False,
            "note": "typos, clear intent",
        },
        {},
    ),
    (
        "y is the shapiro test failing",
        {
            "type": "explain",
            "topic": "the Shapiro-Wilk test",
            "needs_clarification": False,
            "note": "'y' texting shorthand for 'why'",
        },
        {},
    ),
    (
        "explain waht a type 1 error is",
        {
            "type": "explain",
            "topic": "a type I error",
            "needs_clarification": False,
            "note": "typo, clear intent",
        },
        {},
    ),
]

# --------------------------------------------------------------------------
# QuestionSpec building: 8 categories x 10
# --------------------------------------------------------------------------

REALISTIC_QUESTION_SPEC_ITEMS: list[Item] = [
    # A. Near-miss / nonexistent column names -- never silently guessed
    (
        "is icnome different by gender",
        {
            "goal": "compare_groups",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "'icnome' is a typo of 'income'; whether to silently correct common typos or ask is an "
            "open product decision -- the safe label here is to ask, not guess",
        },
        {},
    ),
    (
        "compare revenue across regions",
        {
            "goal": "compare_groups",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "'revenue' is not a column in this dataset; must not be silently mapped to 'income'",
        },
        {},
    ),
    (
        "does age relate to icome",
        {
            "goal": "association",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "typo of 'income'; same judgment call as the first item",
        },
        {},
    ),
    (
        "is there a difference in slaary by department",
        {
            "goal": "compare_groups",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "'slaary' resembles no real column closely enough to assume 'income'",
        },
        {},
    ),
    (
        "check the weit column for outliers",
        {
            "goal": "outliers",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "'weit' is a typo of 'weight', but a fuzzy match is still a guess, not a read",
        },
        {},
    ),
    (
        "summarize the icnome column",
        {
            "goal": "describe",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "typo of 'income'",
        },
        {},
    ),
    (
        "is bonus related to performance",
        {
            "goal": "association",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "neither 'bonus' nor 'performance' is a column in this dataset",
        },
        {},
    ),
    (
        "why is the sallary column missing so much data",
        {
            "goal": "missingness",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "typo/near-miss of a column that may not even exist",
        },
        {},
    ),
    (
        "should we transform the icome variable",
        {
            "goal": "transform",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "typo of 'income'",
        },
        {},
    ),
    (
        "compare test scores by depatment",
        {
            "goal": "compare_groups",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "'test scores' (spaced/plural) and 'depatment' (typo) don't match column names exactly; "
            "whether to normalize such near-misses is a product decision, not assumed here",
        },
        {},
    ),
    # B. Vague references -- no column named at all
    (
        "is it different between the groups?",
        {
            "goal": "compare_groups",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "no outcome or group column named; build_question_spec refuses to guess (Section 10.5)",
        },
        {},
    ),
    (
        "does that column relate to this one?",
        {
            "goal": "association",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "no columns named",
        },
        {},
    ),
    (
        "compare the two groups",
        {
            "goal": "compare_groups",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "no columns named",
        },
        {},
    ),
    (
        "is there a trend in it over time?",
        {
            "goal": "trend",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "no column named",
        },
        {},
    ),
    (
        "why is that missing so much?",
        {
            "goal": "missingness",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "no column named",
        },
        {},
    ),
    (
        "are there outliers in this one?",
        {
            "goal": "outliers",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "no column named",
        },
        {},
    ),
    (
        "should this be transformed?",
        {
            "goal": "transform",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "no column named",
        },
        {},
    ),
    (
        "summarize it",
        {
            "goal": "describe",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "no column named",
        },
        {},
    ),
    (
        "is the average the same as before?",
        {
            "goal": "distribution_fit",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "no column named, and 'before' is not a numeric reference_value",
        },
        {},
    ),
    (
        "is it practically equivalent between them?",
        {
            "goal": "equivalence",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "no columns named",
        },
        {},
    ),
    # C. Two questions in one
    (
        "is income different by gender, and also is age different by region?",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "income", "group": "gender"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "compound; the second clause is an entirely separate spec that must wait its turn",
        },
        {},
    ),
    (
        "compare income by gender and also check if it's related to age",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "income", "group": "gender"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "second clause is a different goal (association)",
        },
        {},
    ),
    (
        "is satisfaction different by department, same employees before and after the change, "
        "and also does it correlate with tenure",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "satisfaction", "group": "period"},
            "design": "paired",
            "needs_clarification": True,
            "note": "design IS stated for the first clause, but a second, unrelated question is bundled in",
        },
        {},
    ),
    (
        "summarize income and also tell me about missingness in age",
        {
            "goal": "describe",
            "variables": {"outcome": "income"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "second clause is a different goal entirely",
        },
        {},
    ),
    (
        "does wait_minutes trend over month, and is it different by clinic",
        {
            "goal": "trend",
            "variables": {"outcome": "wait_minutes", "time": "month"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "second clause is a different goal (compare_groups)",
        },
        {},
    ),
    (
        "are there outliers in weight and also should it be transformed",
        {
            "goal": "outliers",
            "variables": {"outcome": "weight"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "second clause (transform) is a distinct goal",
        },
        {},
    ),
    (
        "is income equal to 50000 on average, or does it differ by region",
        {
            "goal": "distribution_fit",
            "variables": {"outcome": "income"},
            "design": "unknown",
            "reference_value": 50000,
            "needs_clarification": True,
            "note": "'or' presents two different goals as alternatives; can't resolve which to run without asking",
        },
        {},
    ),
    (
        "compare test_score by cohort and also by department",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "test_score", "group": "cohort"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "a second group column suggests two separate questions",
        },
        {},
    ),
    (
        "is reaction_time related to age, and also is it practically the same across gender",
        {
            "goal": "association",
            "variables": {"x": "reaction_time", "y": "age"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "second clause is a different goal (equivalence)",
        },
        {},
    ),
    (
        "why is satisfaction missing so often, and is it different by region too",
        {
            "goal": "missingness",
            "variables": {"outcome": "satisfaction"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "second clause is a different goal (compare_groups)",
        },
        {},
    ),
    # D. Pushing against the product's own rules
    (
        "just compare income by gender, independent groups, don't ask me anything else",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "income", "group": "gender"},
            "design": "independent",
            "needs_clarification": False,
            "note": "despite the pushback framing, design IS explicitly stated; nothing further needs asking",
        },
        {},
    ),
    (
        "skip the design question, just assume independent",
        {
            "goal": "compare_groups",
            "variables": {},
            "design": "independent",
            "needs_clarification": True,
            "note": "design is stated, but no outcome/group column is ever named",
        },
        {},
    ),
    (
        "I don't want to specify design, just run something",
        {
            "goal": "compare_groups",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "explicitly declines to answer; the system cannot silently default (Section 7.1) even "
            "when asked to skip the question",
        },
        {},
    ),
    (
        "don't make me pick a design, just use whichever the data supports",
        {
            "goal": "compare_groups",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "design is a data-collection fact the data itself cannot reveal; still needs asking",
        },
        {},
    ),
    (
        "just tell me if income differs by gender, I don't care about the details",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "income", "group": "gender"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "'I don't care about the details' does not supply the missing design information",
        },
        {},
    ),
    (
        "run the simplest test on income by region, whatever that is",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "income", "group": "region"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "defers method choice to the eligibility engine (fine) but design is still unstated",
        },
        {},
    ),
    (
        "compare wait_minutes across clinics, I already know they're independent so don't ask",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "wait_minutes", "group": "clinic"},
            "design": "independent",
            "needs_clarification": False,
            "note": "design is explicitly asserted by the user; nothing is left to ask",
        },
        {},
    ),
    (
        "just give me a number for how different income is by gender",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "income", "group": "gender"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "'a number' (an effect size) doesn't resolve the design ambiguity",
        },
        {},
    ),
    (
        "forget the assumptions, is age different by department",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "age", "group": "department"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "'forget the assumptions' cannot be honored (rule 2); design is still unstated regardless",
        },
        {},
    ),
    (
        "I already told you it's independent, is test_score different by cohort",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "test_score", "group": "cohort"},
            "design": "independent",
            "needs_clarification": False,
            "note": "taken at face value within this single utterance; a stateful system would also check "
            "its own history, which one isolated eval item cannot represent",
        },
        {},
    ),
    # E. Naturally-phrased genuinely ambiguous designs
    (
        "did income change by region",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "income", "group": "region"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "no cue distinguishing independent regions from the same regions tracked over time",
        },
        {},
    ),
    (
        "is satisfaction linked to the training",
        {
            "goal": "compare_groups",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "ambiguous whether 'linked to' means trained-vs-untrained (compare_groups) or a continuous "
            "association -- the goal itself, not just the design, needs clarifying",
        },
        {},
    ),
    (
        "how does performance compare before and after",
        {
            "goal": "compare_groups",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "'before and after' suggests paired, but could be two independent snapshots; 'performance' "
            "is also not a column name available here",
        },
        {},
    ),
    (
        "is wait time different depending on which clinic you go to",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "wait_minutes", "group": "clinic"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "strongly implies independent groups but never says patients aren't shared across clinics; "
            "Section 7.1's strict rule still applies",
        },
        {},
    ),
    (
        "does the treatment change satisfaction",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "satisfaction", "group": "treatment_group"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "'the treatment' suggests before/after (paired), but 'treatment_group' as a noun suggests "
            "independent groups -- reads both ways",
        },
        {},
    ),
    (
        "is age related to how long someone waits",
        {
            "goal": "association",
            "variables": {"x": "age", "y": "wait_minutes"},
            "design": "unknown",
            "needs_clarification": False,
            "note": "association has no design ambiguity; clean natural phrasing",
        },
        {},
    ),
    (
        "do the three clinics differ in wait times",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "wait_minutes", "group": "clinic"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "'the three clinics' strongly implies independent groups, but the wording doesn't say so "
            "explicitly enough to skip asking, per Section 7.1",
        },
        {},
    ),
    (
        "has morale gone up since the reorg",
        {
            "goal": "trend",
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "could be a trend over time or a before/after paired comparison; 'morale' is also not a "
            "column name available here",
        },
        {},
    ),
    (
        "is there a difference by region, same customers surveyed each quarter",
        {
            "goal": "compare_groups",
            "variables": {"group": "region"},
            "design": "repeated",
            "needs_clarification": True,
            "note": "design IS stated (repeated), but the outcome column is never named",
        },
        {},
    ),
    (
        "does income vary across departments, some people moved between departments during the year",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "income", "group": "department"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "flags unstable group membership, a genuine complication beyond simple independent/paired -- "
            "worth surfacing as its own design conversation, not defaulting to independent",
        },
        {},
    ),
    # F. Typos but clear structure -- robustness check, not a real ambiguity
    (
        "is icnome diferent between gendre groups, independent smaples",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "income", "group": "gender"},
            "design": "independent",
            "needs_clarification": False,
            "note": "typos throughout, but the independent-design cue survives",
        },
        {},
    ),
    (
        "compaer wait_minutes acros clinics, unrelated patients",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "wait_minutes", "group": "clinic"},
            "design": "independent",
            "needs_clarification": False,
            "note": "typos; structure clear",
        },
        {},
    ),
    (
        "sam subjects mesured twice for staisfaction",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "satisfaction", "group": "period"},
            "design": "paired",
            "needs_clarification": False,
            "note": "'sam subjects' typo of 'same subjects' still signals paired",
        },
        {},
    ),
    (
        "is their a realationship between age nad income",
        {
            "goal": "association",
            "variables": {"x": "age", "y": "income"},
            "design": "unknown",
            "needs_clarification": False,
            "note": "typos; structure clear",
        },
        {},
    ),
    (
        "waht is the trned in wait_minutes over the monht",
        {
            "goal": "trend",
            "variables": {"outcome": "wait_minutes", "time": "month"},
            "design": "unknown",
            "needs_clarification": False,
            "note": "typos; structure clear",
        },
        {},
    ),
    (
        "shoud test_score be log transfromed",
        {
            "goal": "transform",
            "variables": {"outcome": "test_score"},
            "design": "unknown",
            "needs_clarification": False,
            "note": "typos; structure clear",
        },
        {},
    ),
    (
        "r there outlier in reaction_time",
        {
            "goal": "outliers",
            "variables": {"outcome": "reaction_time"},
            "design": "unknown",
            "needs_clarification": False,
            "note": "shorthand/typos; structure clear",
        },
        {},
    ),
    (
        "y is weigth missing so ofen",
        {
            "goal": "missingness",
            "variables": {"outcome": "weight"},
            "design": "unknown",
            "needs_clarification": False,
            "note": "'y' shorthand for 'why'; typos elsewhere; structure clear",
        },
        {},
    ),
    (
        "is the avg incom eqaul to 50000",
        {
            "goal": "distribution_fit",
            "variables": {"outcome": "income"},
            "design": "unknown",
            "reference_value": 50000,
            "needs_clarification": False,
            "note": "typos; structure clear",
        },
        {},
    ),
    (
        "smae subjects, repeated mesures of satisfaction across departmnet",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "satisfaction", "group": "department"},
            "design": "repeated",
            "needs_clarification": False,
            "note": "typos; structure clear",
        },
        {},
    ),
    # G. Out-of-scope requests
    (
        "build me a regression model predicting income from age and region",
        {
            "goal": None,
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "regression modelling is out of scope (Section 1.2, 17.1); correct behavior is to say so, "
            "not to force it into one of the nine goals",
        },
        {},
    ),
    (
        "forecast next quarter's income",
        {
            "goal": None,
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "forecasting is beyond EDA diagnostics; TREND describes past change, it does not forecast",
        },
        {},
    ),
    (
        "cluster the customers into segments",
        {
            "goal": None,
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "clustering/segmentation is a modelling task, out of scope",
        },
        {},
    ),
    (
        "build a dashboard of all the key metrics",
        {
            "goal": None,
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "not an EDA question at all",
        },
        {},
    ),
    (
        "impute the missing income values with the mean",
        {
            "goal": "missingness",
            "variables": {"outcome": "income"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "asks for an action (impute) rather than a diagnostic question; missingness as a goal covers "
            "investigating missingness, the imputation method choice happens later in that stage's flow",
        },
        {},
    ),
    (
        "remove the outliers from weight",
        {
            "goal": "outliers",
            "variables": {"outcome": "weight"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "asks for an action (remove), not a diagnostic question -- same reasoning as imputation above",
        },
        {},
    ),
    (
        "normalize the income column",
        {
            "goal": "transform",
            "variables": {"outcome": "income"},
            "design": "unknown",
            "needs_clarification": True,
            "note": "commands a specific transform rather than asking whether one is needed; close enough to "
            "the transform goal to route there, but still needs a turn to confirm which transform",
        },
        {},
    ),
    (
        "tell me which columns are most important for predicting churn",
        {
            "goal": None,
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "feature importance / predictive modelling is out of scope; also no 'churn' column exists",
        },
        {},
    ),
    (
        "what's the best machine learning model for this data",
        {
            "goal": None,
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "modelling recommendation is entirely out of scope (Section 1.2)",
        },
        {},
    ),
    (
        "automatically clean this dataset for me",
        {
            "goal": None,
            "variables": {},
            "design": "unknown",
            "needs_clarification": True,
            "note": "'automatically ... for me' conflicts with rule 3; also too vague to map to one goal",
        },
        {},
    ),
    # H. Well-specified realistic complex phrasing -- should mostly resolve cleanly
    (
        "I want to know if wait times differ between clinic A, B, and C -- these are three different "
        "clinics, not the same patients",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "wait_minutes", "group": "clinic"},
            "design": "independent",
            "needs_clarification": False,
            "note": "design explicitly and naturally stated",
        },
        {},
    ),
    (
        "we surveyed the same 50 employees before and after the training on satisfaction, did it change",
        {
            "goal": "compare_groups",
            "variables": {"outcome": "satisfaction", "group": "period"},
            "design": "paired",
            "needs_clarification": False,
            "note": "paired design explicitly and naturally stated",
        },
        {},
    ),
    (
        "each department reported its average test score for three consecutive quarters, is there a trend",
        {
            "goal": "trend",
            "variables": {"outcome": "test_score", "time": "quarter"},
            "design": "unknown",
            "needs_clarification": False,
            "note": "clean trend question",
        },
        {},
    ),
    (
        "is there a correlation between how long someone waits and how satisfied they are afterward",
        {
            "goal": "association",
            "variables": {"x": "wait_minutes", "y": "satisfaction"},
            "design": "unknown",
            "needs_clarification": False,
            "note": "clean association question, paraphrased naturally",
        },
        {},
    ),
    (
        "our target average income for this cohort was 45000, are we hitting that",
        {
            "goal": "distribution_fit",
            "variables": {"outcome": "income"},
            "design": "unknown",
            "reference_value": 45000,
            "needs_clarification": False,
            "note": "clean distribution_fit question",
        },
        {},
    ),
    (
        "we want to confirm reaction_time is basically unchanged between the treatment and control "
        "groups, independent participants in each",
        {
            "goal": "equivalence",
            "variables": {"outcome": "reaction_time", "group": "treatment_group"},
            "design": "independent",
            "needs_clarification": False,
            "note": "clean equivalence question, design stated",
        },
        {},
    ),
    (
        "can you give me a quick summary of the weight column before we go further",
        {
            "goal": "describe",
            "variables": {"outcome": "weight"},
            "design": "unknown",
            "needs_clarification": False,
            "note": "clean describe question",
        },
        {},
    ),
    (
        "age has a lot of blanks, want to understand why before we proceed",
        {
            "goal": "missingness",
            "variables": {"outcome": "age"},
            "design": "unknown",
            "needs_clarification": False,
            "note": "clean missingness question, paraphrased naturally ('blanks' for missing values)",
        },
        {},
    ),
    (
        "test_score looks like it might have a few extreme values skewing things, can you check",
        {
            "goal": "outliers",
            "variables": {"outcome": "test_score"},
            "design": "unknown",
            "needs_clarification": False,
            "note": "clean outliers question, paraphrased naturally",
        },
        {},
    ),
    (
        "income is heavily skewed, would a transformation help before we model it",
        {
            "goal": "transform",
            "variables": {"outcome": "income"},
            "design": "unknown",
            "needs_clarification": False,
            "note": "clean transform question",
        },
        {},
    ),
]
