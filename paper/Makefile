# Build the epoch-kinks paper: PDF via tectonic, HTML via LaTeXML (the arXiv
# HTML stack). Figures are committed (PDF for the PDF build, PNG for the HTML
# build); `make figures` regenerates them from a local extraction of the
# public Epoch AI datasets — see figures/make_figures.py.

DATA_ROOT ?= /Users/haukehillebrandt/dev/epoch-data/extracted

.PHONY: all pdf html figures clean

all: pdf html

pdf: main.pdf

main.pdf: main.tex
	tectonic main.tex

html: main.tex
	latexmlc main.tex --dest=html/index.html --timeout=600

# Requires the natex dev environment (uv sync --extra dev at the repo root)
# and a local copy of the public Epoch CSVs. Not run in CI — the data are
# never committed; the committed figures are the build inputs.
figures:
	cd .. && uv run python paper/figures/make_figures.py --data-root $(DATA_ROOT)

clean:
	rm -rf main.pdf html
