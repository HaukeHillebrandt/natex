# papers/ — the natex paper collection

**📕 Read everything in one file: [ALL_PAPERS.pdf](./ALL_PAPERS.pdf)** (80 pages, bookmarked) · **🌐 Always-current site: [haukehillebrandt.github.io/natex](https://haukehillebrandt.github.io/natex/)**

| # | Paper | Read on GitHub | Typeset |
|---|-------|----------------|---------|
| 1 | Systematic Survey (capstone) | [md](./capstone/) · [pdf](./capstone/main.pdf) | [html](https://haukehillebrandt.github.io/natex/capstone/) |
| 2 | Three Kinks and a Null (flagship) | [md](../paper/) · [pdf](../paper/main.pdf) | [html](https://haukehillebrandt.github.io/natex/paper/) |
| 3 | BTOS adoption at DeepSeek-R1 | [md](./btos-sector-did-r1/) · [pdf](./btos-sector-did-r1/main.pdf) | [html](https://haukehillebrandt.github.io/natex/btos-sector-did-r1/) |
| 4 | Export controls: three legs | [md](./export-controls-three-leg/) · [pdf](./export-controls-three-leg/main.pdf) | [html](https://haukehillebrandt.github.io/natex/export-controls-three-leg/) |
| 5 | BTOS rewording measurement RDD | [md](./btos-rewording-rdd/) · [pdf](./btos-rewording-rdd/main.pdf) | [html](https://haukehillebrandt.github.io/natex/btos-rewording-rdd/) |
| 6 | Capex at ChatGPT (DiK) | [md](./capex-dik-chatgpt/) · [pdf](./capex-dik-chatgpt/main.pdf) | [html](https://haukehillebrandt.github.io/natex/capex-dik-chatgpt/) |
| 7 | LMArena sycophancy ABA | [md](./lmarena-sycophancy-aba/) · [pdf](./lmarena-sycophancy-aba/main.pdf) | [html](https://haukehillebrandt.github.io/natex/lmarena-sycophancy-aba/) |
| 8 | EU AI Act bunching | [md](./euact-bunching-writeup/) · [pdf](./euact-bunching-writeup/main.pdf) | [html](https://haukehillebrandt.github.io/natex/euact-bunching-writeup/) |
| 9 | Still No Kink at o1 (ECI) | [md](./eci-fresh-kink-o1/) · [pdf](./eci-fresh-kink-o1/main.pdf) | [html](https://haukehillebrandt.github.io/natex/eci-fresh-kink-o1/) |
| 10 | Semiconductor event studies | [md](./semis-event-studies-writeup/) · [pdf](./semis-event-studies-writeup/main.pdf) | [html](https://haukehillebrandt.github.io/natex/semis-event-studies-writeup/) |
| 11 | Chinchilla: an honest miss | [md](./chinchilla-writeup/) · [pdf](./chinchilla-writeup/main.pdf) | [html](https://haukehillebrandt.github.io/natex/chinchilla-writeup/) |
| 12 | Prop 99 blind validation | [md](./prop99-validation-writeup/) · [pdf](./prop99-validation-writeup/main.pdf) | [html](https://haukehillebrandt.github.io/natex/prop99-validation-writeup/) |

Each paper directory carries a `README.md` (a pandoc Markdown render that GitHub
displays when you open the folder) and a committed `main.pdf` (GitHub's viewer
renders it in-browser). These are convenience snapshots refreshed by
`scripts/refresh_paper_artifacts.sh`; the LaTeX sources are canonical and the
Pages site is rebuilt from them on every push.

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
