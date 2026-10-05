#!/usr/bin/env python3
"""Merge the paper collection into papers/ALL_PAPERS.pdf with bookmarks.

Cover page is rebuilt from papers/_cover/cover.tex when present; the merge
order is the collection's reading order (capstone, flagship, minis by
suitability rank at authoring time).
"""
import os
import subprocess

from pypdf import PdfReader, PdfWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = [
    ("Cover & Contents", "papers/_cover/cover.pdf"),
    ("1. Systematic Survey (capstone)", "papers/capstone/main.pdf"),
    ("2. natex 0.3.0 methods paper", "papers/natex-methods/main.pdf"),
    ("3. Placebo-calibrated kink tests (flagship)", "paper/main.pdf"),
    ("4. BTOS adoption at DeepSeek-R1", "papers/btos-sector-did-r1/main.pdf"),
    ("5. Export controls: three legs", "papers/export-controls-three-leg/main.pdf"),
    ("6. BTOS rewording measurement RDD", "papers/btos-rewording-rdd/main.pdf"),
    ("7. Capex at ChatGPT (DiK)", "papers/capex-dik-chatgpt/main.pdf"),
    ("8. LMArena sycophancy ABA", "papers/lmarena-sycophancy-aba/main.pdf"),
    ("9. EU AI Act bunching", "papers/euact-bunching-writeup/main.pdf"),
    ("10. Still No Kink at o1 (ECI)", "papers/eci-fresh-kink-o1/main.pdf"),
    ("11. Semiconductor event studies", "papers/semis-event-studies-writeup/main.pdf"),
    ("12. Chinchilla: an honest miss", "papers/chinchilla-writeup/main.pdf"),
    ("13. Prop 99 blind validation", "papers/prop99-validation-writeup/main.pdf"),
    ("14. Chip-level export controls (SuDDDS)", "papers/chip-suddds-exportcontrols/main.pdf"),
    ("15. BTOS spliced panel: R1 kink out of sample", "papers/btos-spliced-r1-extension/main.pdf"),
    ("16. Benchmark contamination (public vs held-out)", "papers/benchmark-contamination-dee/main.pdf"),
    ("17. Datacenter sites at 2025Q1", "papers/datacenter-sites-2025q1/main.pdf"),
    ("18. State AI-exposure gradient at R1", "papers/aei-btos-state-gradient/main.pdf"),
    ("19. CVE publications: no kink at R1", "papers/cve-monthly-kink-r1/main.pdf"),
]


def main() -> None:
    cover_tex = os.path.join(ROOT, "papers/_cover/cover.tex")
    if os.path.exists(cover_tex):
        subprocess.run(["tectonic", "cover.tex"], cwd=os.path.dirname(cover_tex), check=True,
                       capture_output=True)
    writer = PdfWriter()
    page = 0
    for title, rel in ORDER:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"skip (missing): {rel}")
            continue
        reader = PdfReader(path)
        writer.append(reader)
        writer.add_outline_item(title, page)
        page += len(reader.pages)
    out = os.path.join(ROOT, "papers/ALL_PAPERS.pdf")
    with open(out, "wb") as fh:
        writer.write(fh)
    print(f"papers/ALL_PAPERS.pdf: {page} pages, {len(ORDER)} bookmarks")


if __name__ == "__main__":
    main()
