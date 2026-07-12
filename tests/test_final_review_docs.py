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
