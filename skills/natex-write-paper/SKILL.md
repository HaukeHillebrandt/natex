---
name: natex-write-paper
description: Render the AI-draft manuscript from a natex results bundle and walk the user through verifying it. Use when the user says things like "write up the discovery as a paper", "draft a paper from the natex results", "turn this discovery into a manuscript", or "render the natex paper". Covers the [report]/[plot] extras, natex paper --bundle in markdown and LaTeX (tectonic PDF compile), the mandatory AI-draft banner, checking every number against results.json, and the manual Google Docs route.
---

# Write up a natex discovery as a paper

## 1. Prerequisites

You need a finished results bundle directory — the `--out` dir of a completed
`natex discover` run (ideally produced via the discover-natural-experiments
skill). A plain `natex discover` run leaves `results.json` there; a plan-mode
run (`natex discover --plan ...`) leaves `discover_report.json` instead. The
bundle loader accepts either file.

The paper renderer needs the `report` extra (jinja2 templates); add `plot` too
if you want figures embedded:

```bash
uv add 'natex-discovery[report]'
uv add 'natex-discovery[plot]'   # optional: figures in the draft
```

(From a repo checkout: `uv sync --extra report --extra plot`.) Python >= 3.11.

## 2. Render the draft

Markdown always works:

```bash
uv run natex paper --bundle OUT --format md
```

This writes `OUT/paper/paper.md` (pass `--out` to choose another directory).

LaTeX:

```bash
uv run natex paper --bundle OUT --format latex
```

This writes `OUT/paper/paper.tex` and compiles it to `paper.pdf` **only when
`tectonic` is on PATH** (install it from
https://tectonic-typesetting.github.io, e.g. `brew install tectonic`). A
missing compiler is not an error: the command leaves the `.tex` file in place
and prints a message telling you tectonic was not found — hand the `.tex` to
any LaTeX toolchain, or fall back to `--format md`.

## 3. The AI-draft banner is non-negotiable

Every rendered draft opens with the banner, verbatim from the code:

> AI-generated draft — verify all claims before circulation

Do not remove it, and walk the user through earning it before the draft goes
anywhere:

- **Check every number in the draft against the bundle's report JSON** —
  `OUT/results.json` for a plain run, `OUT/discover_report.json` for a
  plan-mode run: the single source every rendered number comes from. Open the
  draft and the bundle side by side and confirm each estimate, standard error,
  confidence interval, p-value, and count matches. Never fabricate or "fix" a
  number that looks off; if the draft and the report JSON disagree, the render
  is stale — re-run `natex paper`, do not hand-edit statistics.
- **Read the validation section skeptically.** Only discoveries that passed
  the validation battery (randomization, placebo, density) belong in headline
  claims; surface `weak_instrument` flags and honest-split caveats rather than
  smoothing them over.
- **Missing values render as "—" and must stay that way.** A "—" (or
  `null`/`NaN` in `results.json`) means the computation failed or was
  underpowered — never fill one in with a number.

## 4. What counts as a finding

- A discovery is a finding only if it cleared the validation battery AND, for any
  design whose running variable is calendar time, its **placebo-calibrated p** (not
  the nominal HC1/CR1 p) is at or below 0.05 over at least 19 shifted placebo cutoffs.
  A 7-placebo grid has a floor of 0.125; "0 of 7 placebos reject" is not localization.
- Rejections at placebo cutoffs are a size diagnostic (the nominal test is oversized
  on that series), never a "power certificate" for a null at the event date.
- `inconclusive` verdicts (refused placebo test, mechanical rediscovery, too few
  placebo positions) are not findings and not nulls: they belong in one line of the
  limitations, not in the abstract.
- One abstract claim per surviving finding; a paper whose only surviving result is a
  null is a short note, not a paper.

## 5. Style contract

The draft is read by people who know the methods. Write for them:

- Sentences under 25 words, one idea each. No sentence may carry more than one
  statistic unless it is a table row.
- At most one em dash per page; prefer a full stop. No parentheses inside sentences
  for numbers — numbers go in tables.
- Titles are descriptive, not slogans: "Business AI adoption did not bend at
  DeepSeek-R1 in BTOS sector data", never "A Sharp Bend, No Verdict: ...".
- State the inference caveats ONCE, in a "Limitations" paragraph. Do not repeat
  "honest", "stated as run", "refused", "not patched over", "the dial reads zero",
  "rejections are results", "the battery is the analysis" or any such phrase; the
  automated pipeline's refusals and degenerate configurations go in one appendix
  table (family, status, reason), never in the results text.
- Report each number once: the table carries it, the prose interprets it. Do not
  list every specification cell in prose; give the range and point to the table.
- No self-citation chains: cite the natex software once; cite companion notes only
  when a number is taken from them.
- Say what was NOT established in the first paragraph of the discussion, in plain
  words ("these data cannot attribute the bend to X").
- Delete any sentence that praises the pipeline, the battery, or the authors'
  honesty. The reader judges that from the tables.

## 6. Google Docs (manual route)

natex **does not integrate** with the Google Docs API. To get the draft into
Google Docs, render markdown first, then either:

- open a new Google Doc and paste the contents of `OUT/paper/paper.md` in, or
- upload `paper.md` to Google Drive and use "Open with → Google Docs".

Either way, the banner line must survive the transfer at the top of the Doc.

## 7. Warnings

- **Never fabricate** statistics, citations, or results — every number comes
  from `results.json` and nowhere else.
- The draft may cover only discoveries that passed the **validation battery**;
  a candidate without it is not a finding and does not belong in a manuscript.
- The output is **AI-generated**: a human must verify every claim before the
  draft is shared, circulated, or submitted anywhere.
