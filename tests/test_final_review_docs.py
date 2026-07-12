"""Final-review status docs (phase final-review, task 1).

Pins the skeleton contract of docs/status/final-review.md and
docs/status/future_work.md: later review tasks append rows under these exact
section headings, and the task-8/9 gate tests key on them. Pure text
assertions — no rendering, no CLI runs.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FINAL_REVIEW = ROOT / "docs" / "status" / "final-review.md"
FUTURE_WORK = ROOT / "docs" / "status" / "future_work.md"

REQUIRED_HEADINGS = (
    "## A. Math-audit conformance",
    "## B. API consistency sweep",
    "## C. Docs accuracy execution log",
    "## D. Spec completeness matrix",
    "## Findings register",
    "## Fixes applied",
    "## Run of record",
)

FUTURE_WORK_TABLE_HEADER = "| Item | Spec/audit ref | Rationale |"


def test_status_docs_exist_with_required_sections():
    assert FINAL_REVIEW.is_file(), f"missing {FINAL_REVIEW}"
    assert FUTURE_WORK.is_file(), f"missing {FUTURE_WORK}"
    review = FINAL_REVIEW.read_text(encoding="utf-8")
    for heading in REQUIRED_HEADINGS:
        assert heading in review, f"final-review.md missing heading: {heading}"
    future = FUTURE_WORK.read_text(encoding="utf-8")
    assert FUTURE_WORK_TABLE_HEADER in future, (
        "future_work.md missing its table header row"
    )


def _future_work_rows() -> list[list[str]]:
    """Data rows of the future_work.md table as stripped cell lists."""
    lines = FUTURE_WORK.read_text(encoding="utf-8").splitlines()
    start = lines.index(FUTURE_WORK_TABLE_HEADER)
    rows = []
    for line in lines[start + 2 :]:  # skip header + |---| separator
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


CARDS = ROOT / "docs" / "method_cards"


def test_method_cards_carry_all_audit_typos():
    """F-B1 (audit section-2 pure-typos block, task-3 protocol): every listed
    typo appears in the relevant method card. The six that were missing:
    Eq 5.21 (inverse-logit + log mu) in lord3.md; the thesis 6.26 -> 5.22
    cross-reference in suddds.md; Codex #31/#34/#35/#36 in dee.md."""
    lord3 = (CARDS / "lord3.md").read_text(encoding="utf-8")
    assert "5.21" in lord3, "Eq 5.21 typo missing from lord3.md"
    assert "inverse logit" in lord3.lower().replace("-", " ")
    assert "log μ" in lord3, "the missing log mu_i correction (Codex) absent"
    suddds = (CARDS / "suddds.md").read_text(encoding="utf-8")
    assert "6.26" in suddds and "5.22" in suddds, "thesis xref 6.26->5.22 missing"
    dee = " ".join((CARDS / "dee.md").read_text(encoding="utf-8").split())
    for token in ("#31", "#34", "#35", "#36", "conditional mean independence"):
        assert token in dee, f"DEE typo marker {token!r} missing from dee.md"


def test_future_work_rows_have_rationales():
    """Task 6: the future-work register is populated and every row is complete.

    The spec-completeness matrix (final-review.md section D) sends every
    deliberately-unimplemented promise here; at minimum the planner's
    pre-identified deferrals (rdrobust/rddensity bridges, PyPI, staggered
    adoption, Google Docs export, ...) mean the table has >= 6 rows, and each
    row must carry a non-empty Item, Spec/audit ref, and Rationale cell.
    """
    rows = _future_work_rows()
    assert len(rows) >= 6, f"expected >= 6 future-work rows, found {len(rows)}"
    for i, cells in enumerate(rows):
        assert len(cells) == 3, f"row {i} does not have exactly 3 cells: {cells}"
        item, ref, rationale = cells
        assert item, f"row {i}: empty Item cell"
        assert ref, f"row {i}: empty Spec/audit ref cell"
        assert rationale, f"row {i}: empty Rationale cell"
