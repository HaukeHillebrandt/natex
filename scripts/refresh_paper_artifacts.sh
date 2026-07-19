#!/usr/bin/env bash
# Rebuild the committed paper artifacts: per-paper main.pdf, figure PNGs,
# Markdown renders (README.md), and papers/ALL_PAPERS.pdf.
# Requires: tectonic, pandoc, ghostscript (gs), python3 + pypdf.
set -euo pipefail
cd "$(dirname "$0")/.."
for d in paper papers/*/; do
  d=${d%/}
  [ -f "$d/main.tex" ] || continue
  (cd "$d" && tectonic main.tex >/dev/null)
  find "$d/figures" -name '*.pdf' 2>/dev/null | while read -r f; do
    gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r150 -o "${f%.pdf}.png" "$f"
  done
  pandoc "$d/main.tex" -f latex -t gfm --wrap=none -o "$d/README.md.tmp"
  python3 - "$d" <<'PY'
import re, sys, os
d = sys.argv[1]; slug = os.path.basename(d)
s = open(os.path.join(d, 'README.md.tmp')).read()
s = re.sub(r'(figures/[\w.-]+)\.pdf', r'\1.png', s)
s = re.sub(r'\.\./([\w-]+)/figures/([\w.-]+)\.pdf', r'../\1/figures/\2.png', s)
base = 'https://haukehillebrandt.github.io/natex/'
html = base + ('paper/' if slug == 'paper' else slug + '/')
hdr = f"> **Markdown render for GitHub browsing** — typeset versions: [HTML]({html}) · [PDF in this repo](./main.pdf)\n\n"
open(os.path.join(d, 'README.md'), 'w').write(hdr + s)
os.remove(os.path.join(d, 'README.md.tmp'))
PY
done
python3 scripts/merge_all_papers.py
echo "refreshed: per-paper PDFs, PNGs, READMEs, papers/ALL_PAPERS.pdf"
