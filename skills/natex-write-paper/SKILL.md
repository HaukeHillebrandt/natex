---
name: natex-write-paper
description: Render the AI-draft manuscript from a natex results bundle and walk the user through verifying it. Use when the user says things like "write up the discovery as a paper", "draft a paper from the natex results", "turn this discovery into a manuscript", or "render the natex paper". Covers the [report]/[plot] extras, natex paper --bundle in markdown and LaTeX (tectonic PDF compile), the mandatory AI-draft banner, checking every number against results.json, and the manual Google Docs route.
---

# Write up a natex discovery as a paper

## 1. Prerequisites

You need a finished results bundle directory — the `--out` dir of a completed
`natex discover` run (ideally produced via the discover-natural-experiments
skill), containing `results.json` and its companion files.

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

- **Check every number in the draft against `OUT/results.json`** — the single
  source every rendered number comes from. Open the draft and the bundle side
  by side and confirm each estimate, standard error, confidence interval,
  p-value, and count matches. Never fabricate or "fix" a number that looks off;
  if the draft and `results.json` disagree, the render is stale — re-run
  `natex paper`, do not hand-edit statistics.
- **Read the validation section skeptically.** Only discoveries that passed
  the validation battery (randomization, placebo, density) belong in headline
  claims; surface `weak_instrument` flags and honest-split caveats rather than
  smoothing them over.
- **Missing values render as "—" and must stay that way.** A "—" (or
  `null`/`NaN` in `results.json`) means the computation failed or was
  underpowered — never fill one in with a number.

## 4. Google Docs (manual route)

natex **does not integrate** with the Google Docs API. To get the draft into
Google Docs, render markdown first, then either:

- open a new Google Doc and paste the contents of `OUT/paper/paper.md` in, or
- upload `paper.md` to Google Drive and use "Open with → Google Docs".

Either way, the banner line must survive the transfer at the top of the Doc.

## 5. Warnings

- **Never fabricate** statistics, citations, or results — every number comes
  from `results.json` and nowhere else.
- The draft may cover only discoveries that passed the **validation battery**;
  a candidate without it is not a finding and does not belong in a manuscript.
- The output is **AI-generated**: a human must verify every claim before the
  draft is shared, circulated, or submitted anywhere.
