# papers/ — the natex paper collection

Mini-papers written with the natex toolkit, published alongside the flagship
paper as one GitHub Pages site.

## Layout

```
paper/                  flagship paper (main.tex + Makefile + figures/)
papers/
  <slug>/main.tex       one mini-paper per directory
  capstone/main.tex     the capstone synthesis (optional; listed separately)
scripts/build_papers.sh shared build script (CI and local)
```

Any `papers/<slug>/` directory containing a `main.tex` is picked up
automatically — no registration step. A directory named `capstone` is treated
as the capstone and listed in its own section of the site index. Figures and
other inputs live inside the paper's own directory; datasets are never
committed (commit the rendered figures instead, as `paper/` does).

## Build

Both CI (`.github/workflows/paper.yml`) and local runs use the same script:

```sh
scripts/build_papers.sh          # builds everything into ./site
scripts/build_papers.sh /tmp/out # or into a directory of your choice
```

For each paper `<slug>` it produces, under the site directory:

- `<slug>.pdf` — via `tectonic main.tex`
- `<slug>/index.html` — via `latexmlc` (the arXiv HTML stack)
- `index.html` at the site root — a plain-HTML index of the collection, with
  titles parsed from each paper's `\title{}` (the flagship `paper.pdf` is also
  copied to `main.pdf` for old links)

Requirements: `tectonic` and `latexmlc` on `PATH` (macOS:
`brew install tectonic latexml`). The flagship `paper/Makefile` still works
for iterating on that paper alone (`make -C paper pdf`, `make -C paper html`,
`make -C paper figures`); per-paper Makefiles are optional — the script needs
only `main.tex`.

The generated `site/` tree is disposable output and git-ignored; the Pages
deploy is triggered by pushes to `main` touching `paper/**`, `papers/**`, the
build script, or the workflow.
