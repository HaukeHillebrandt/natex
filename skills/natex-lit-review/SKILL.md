---
name: natex-lit-review
description: Generate the deterministic natex research brief, hand it to the user's deep-research tooling, vet every returned citation, and merge the survivors into the paper draft's related-work section. Use when the user says things like "literature review for my discovery", "find related work for this natural experiment", "what papers relate to this RDD", or "deep research on my natex results". Covers natex brief --bundle and the research_brief Python API, the research-brief.md handoff (e.g. to a Gemini deep-research skill), citation verification, and the merge back into paper.md.
---

# Literature review for a natex discovery

## 1. Generate the brief

From a finished results bundle directory (the `--out` dir of a completed
`natex discover` run — a plain run leaves `results.json` there, a plan-mode
run leaves `discover_report.json`; the loader accepts either):

```bash
uv run natex brief --bundle OUT
```

This writes `OUT/research-brief.md` and prints its path. Pass
`--out some/dir` (or `--out some/path.md` to name the file exactly) to write
elsewhere. The brief is **deterministic**: pure text built from the bundle's
report JSON (`OUT/results.json` or `OUT/discover_report.json`) alone — no
network, no LLM, no timestamps — so reruns are byte-identical and safe to
repeat.

Python API alternative (core deps only, no extras needed):

```python
from natex.report import ResultsBundle, research_brief

path = research_brief(ResultsBundle.load("OUT"), "OUT")
print(path)  # OUT/research-brief.md
```

## 2. What the brief contains

`research-brief.md` is formatted to be pasted verbatim into a deep-research
agent. It carries:

- **Discovery context** — rows, source, treatment/outcome/forcing columns,
  user-provided context, and search coverage.
- **Discovered designs** — each scanned RDD/DiD candidate with its cutoff
  center or onset, top forcing influence, and Monte Carlo p-value.
- **Effect estimates** — tau, se, and 95% CI per estimator, with
  `weak_instrument` and other flags preserved; missing values stay "—".
- **Validation status** — randomization, placebo (Holm), and density results.
- **Numbered literature questions** — the concrete questions the research
  agent should answer (statutory thresholds, competing policies, prior
  quasi-experimental studies, published effect estimates).

## 3. Hand off to deep research

Give `research-brief.md` to the user's own deep-research tooling — e.g. a
Gemini deep-research skill / Deep Research query, or any comparable research
agent — pasting the brief verbatim as the query. natex performs **no research
calls itself**; the handoff is just this text file. Ask the research agent
for a source-linked literature review that answers the numbered questions and
flags any prior study of the same threshold, subset, or policy date.

## 4. Vet the returned citations, then merge into the paper

The returned review is itself AI-generated. Before any of it touches the
draft:

1. **Verify every citation actually exists.** Resolve each DOI or search the
   exact title/authors; open the source and confirm it says what the review
   claims. Drop anything unverifiable — **never fabricate** a reference, and
   never keep one you could not resolve.
2. **Select the genuinely related work** — prior studies of the same
   threshold, policy, or outcome; skip padding.
3. **Edit the related-work section** of `OUT/paper/paper.md` (or `.tex` if
   you rendered LaTeX) to weave the vetted citations in, with an accurate
   one-clause summary of each.
4. **Keep the AI-draft banner** ("AI-generated draft — verify all claims
   before circulation") at the top of the draft — merging literature does not
   earn its removal.
5. **Re-check that no statistic in the draft changed.** Every number still
   comes from `OUT/results.json` and nowhere else; the literature pass edits
   prose and references only.

## 5. Warnings

- Deep-research output is **AI-generated**: verify every claim and **every
  citation** by hand before merging anything.
- **Never fabricate** citations, quotes, or statistics — an unverifiable
  reference is deleted, not kept.
- The **validation battery** (randomization, placebo, density), not the
  literature, decides whether a discovery stands. Related work can contextualize
  a finding; it cannot rescue one that failed validation.
