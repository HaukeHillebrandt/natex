#!/usr/bin/env bash
# Build the natex paper collection into a static site tree.
#
# Every directory holding a main.tex is a paper: the flagship paper/ plus each
# mini-paper under papers/*/ (a papers/capstone/ dir, when present, is listed
# separately as the capstone). For each paper <slug> this builds
#
#   <site>/<slug>.pdf         via tectonic
#   <site>/<slug>/index.html  via latexmlc (the arXiv HTML stack)
#
# and finally writes <site>/index.html, a plain HTML index of the collection.
# The same script is run by CI (.github/workflows/paper.yml) and locally:
#
#   scripts/build_papers.sh [site-dir]    # default: ./site
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
site_arg="${1:-$repo/site}"
mkdir -p "$site_arg"
site="$(cd "$site_arg" && pwd)"

# ---- collect the papers -----------------------------------------------------

slugs=()   # site-tree name of each paper (basename of its directory)
dirs=()    # absolute source directory
kinds=()   # main | mini | capstone

if [[ -f "$repo/paper/main.tex" ]]; then
  slugs+=("paper"); dirs+=("$repo/paper"); kinds+=("main")
fi
for d in "$repo"/papers/*/; do
  [[ -f "${d}main.tex" ]] || continue
  slug="$(basename "$d")"
  kind="mini"
  [[ "$slug" == "capstone" ]] && kind="capstone"
  slugs+=("$slug"); dirs+=("${d%/}"); kinds+=("$kind")
done

if [[ ${#slugs[@]} -eq 0 ]]; then
  echo "error: no main.tex found under paper/ or papers/*/" >&2
  exit 1
fi

# ---- helpers ----------------------------------------------------------------

# Extract the \title{...} of a .tex file (multi-line, one nesting level of
# braces) and flatten it to plain text. Falls back to the slug if absent.
title_of() {
  local tex="$1" fallback="$2" t
  t="$(perl -0777 -ne '
    if (/\\title\{((?:[^{}]|\{[^{}]*\})*)\}/s) {
      my $t = $1;
      $t =~ s/\\\\/ /g;            # line breaks inside the title
      $t =~ s/\\[A-Za-z]+\s*/ /g;  # drop command names, keep their text
      $t =~ s/[{}~]/ /g;
      $t =~ s/\s+/ /g;
      $t =~ s/^\s+|\s+$//g;
      print $t;
    }' "$tex")"
  printf '%s' "${t:-$fallback}"
}

html_escape() {
  local s="$1"
  s="${s//&/&amp;}"; s="${s//</&lt;}"; s="${s//>/&gt;}"
  printf '%s' "$s"
}

# ---- build each paper -------------------------------------------------------

for i in "${!slugs[@]}"; do
  slug="${slugs[$i]}" dir="${dirs[$i]}"
  echo "==> building $slug (${dir#"$repo"/})"
  (cd "$dir" && tectonic main.tex)
  install -m 644 "$dir/main.pdf" "$site/$slug.pdf"
  mkdir -p "$site/$slug"
  (cd "$dir" && latexmlc main.tex --dest="$site/$slug/index.html" --timeout=600)
done

# Back-compat: the flagship PDF used to live at the site root as main.pdf.
[[ -f "$site/paper.pdf" ]] && cp "$site/paper.pdf" "$site/main.pdf"

# ---- site index -------------------------------------------------------------

entry() { # $1=slug  -> one <li> linking the HTML and PDF builds
  local slug="$1" dir="$2" title
  title="$(html_escape "$(title_of "$dir/main.tex" "$slug")")"
  printf '      <li><a href="./%s/">%s</a> <span class="fmt">[<a href="./%s.pdf">pdf</a>]</span></li>\n' \
    "$slug" "$title" "$slug"
}

{
  cat <<'HEAD'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>natex — papers</title>
<style>
  body { max-width: 44rem; margin: 3rem auto; padding: 0 1rem;
         font: 1rem/1.6 Georgia, 'Times New Roman', serif; color: #1a1a1a; }
  h1 { font-size: 1.6rem; margin-bottom: 0.2rem; }
  h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ccc;
       padding-bottom: 0.2rem; }
  ul { padding-left: 1.2rem; }
  li { margin: 0.5rem 0; }
  a { color: #1a5276; }
  .fmt { font-size: 0.85em; color: #666; }
  .sub { color: #555; margin-top: 0; }
  @media (prefers-color-scheme: dark) {
    body { background: #121212; color: #ddd; }
    a { color: #7fb3d5; } .fmt, .sub { color: #999; }
    h2 { border-color: #444; }
  }
</style>
</head>
<body>
<h1>natex — papers</h1>
<p class="sub">Formal natural-experiment analyses built with the
<a href="https://github.com/HaukeHillebrandt/natex">natex</a> toolkit.
Each entry links the HTML rendering; PDFs in brackets.</p>
HEAD

  for i in "${!slugs[@]}"; do
    [[ "${kinds[$i]}" == "main" ]] || continue
    echo '<h2>Main paper</h2>'
    echo '  <ul>'
    entry "${slugs[$i]}" "${dirs[$i]}"
    echo '  </ul>'
  done

  have_mini=0
  for k in "${kinds[@]}"; do [[ "$k" == "mini" ]] && have_mini=1; done
  if [[ $have_mini -eq 1 ]]; then
    echo '<h2>Mini-papers</h2>'
    echo '  <ul>'
    for i in "${!slugs[@]}"; do
      [[ "${kinds[$i]}" == "mini" ]] || continue
      entry "${slugs[$i]}" "${dirs[$i]}"
    done
    echo '  </ul>'
  fi

  for i in "${!slugs[@]}"; do
    [[ "${kinds[$i]}" == "capstone" ]] || continue
    echo '<h2>Capstone</h2>'
    echo '  <ul>'
    entry "${slugs[$i]}" "${dirs[$i]}"
    echo '  </ul>'
  done

  echo '</body>'
  echo '</html>'
} > "$site/index.html"

echo "==> site assembled at $site"
