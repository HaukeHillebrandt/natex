"""Generalizability guard: the survey layer is dataset-agnostic and content-blind.

Three proofs, per the phase plan (task 10):

1. Token grep — no benchmark-dataset-specific token appears anywhere in the
   survey sources, the survey HTML module, or the survey report template.
2. Recording-stub run — the FULL :func:`heuristic_applicability` pipeline
   touches only allowed profile *metadata* attributes, never sees a pandas
   object, and no verdict callable even accepts a DataFrame parameter.
3. AST literal scan — verdict logic hard-codes no benchmark column names.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pandas as pd
import pytest

from natex.survey.applicability import heuristic_applicability
from natex.survey.registry import FAMILIES, FAMILY_ORDER, DeclaredInputs
from test_survey_registry import _ALLOWED_PROFILE_ATTRS, _RecordingProfile

ROOT = Path(__file__).resolve().parents[1]
_SURVEY_DIR = ROOT / "src" / "natex" / "survey"
_REPORT_FILES = (
    ROOT / "src" / "natex" / "report" / "survey_html.py",
    ROOT / "src" / "natex" / "report" / "templates" / "survey.html.j2",
)

# Benchmark-dataset tokens assembled from parts at runtime so this test file
# itself cannot trip the grep (nor any future scan that includes tests/).
_FORBIDDEN_TOKENS = (
    "fit" + "bit",
    "ep" + "och",
    "prop" + "99",
    "smok" + "ing",
    "chin" + "chilla",
    "gp" + "qa",
    "me" + "tr",
    "lm" + "arena",
    "take" + "out",
)

# Benchmark column/role names, also built from parts.
_FORBIDDEN_COLUMN_LITERALS = (
    "pre" + "test",
    "post" + "test",
    "months" + "_23",
    "cig" + "sale",
    "dist" + "_from_cut",
)


def _survey_source_files() -> list[Path]:
    files = [
        p
        for p in sorted(_SURVEY_DIR.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]
    return files + [p for p in _REPORT_FILES]


def _token_pattern(token: str) -> re.Pattern[str]:
    # Alphanumeric-boundary match: catches the token as a standalone word AND
    # underscore-joined identifiers, without tripping on English words that
    # merely contain it as a substring.
    return re.compile(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", re.IGNORECASE)


def test_no_dataset_specific_tokens():
    files = _survey_source_files()
    # Sanity: the scan must actually cover the survey modules and the report.
    names = {p.name for p in files}
    assert {"registry.py", "applicability.py", "runner.py", "survey_html.py"} <= names
    assert any(p.suffix == ".j2" for p in files)

    hits: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_TOKENS:
            if _token_pattern(token).search(text):
                hits.append(f"{path.relative_to(ROOT)}: {token}")
    assert not hits, f"dataset-specific tokens in survey sources: {hits}"


class _ValueRecordingProfile(_RecordingProfile):
    """Task-1 stub, extended to also record every value handed to a predicate."""

    def __init__(self) -> None:
        super().__init__()
        self.returned: list[tuple[str, object]] = []

    def __getattr__(self, name: str):
        value = super().__getattr__(name)
        self.returned.append((name, value))
        return value


def _leaves(value):
    """Yield every scalar reachable inside lists/tuples/dicts."""
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    else:
        yield value


def test_heuristic_applicability_is_content_blind():
    stub = _ValueRecordingProfile()
    verdicts = heuristic_applicability(stub, None, DeclaredInputs())

    # The full pipeline ran: every family got a verdict, in registry order.
    assert tuple(verdicts) == FAMILY_ORDER
    assert stub.accessed, "stub was never consulted"
    assert stub.accessed <= _ALLOWED_PROFILE_ATTRS, (
        f"heuristic_applicability read non-metadata attributes: "
        f"{stub.accessed - _ALLOWED_PROFILE_ATTRS}"
    )

    # No attribute access returned a pandas object — the stub serves only
    # ints/lists/dataclass-like column records, and everything the predicates
    # consumed stayed in that vocabulary.
    assert stub.returned, "no values were served to any predicate"
    for name, value in stub.returned:
        for leaf in _leaves(value):
            assert not isinstance(leaf, (pd.DataFrame, pd.Series, pd.Index)), (
                f"profile attribute {name!r} handed a pandas object to a predicate"
            )

    # Signature proof: neither heuristic_applicability nor any registry check
    # even ACCEPTS a DataFrame parameter.
    callables = [heuristic_applicability] + [
        req.check for fam in FAMILIES.values() for req in fam.requirements
    ]
    for fn in callables:
        for param in inspect.signature(fn).parameters.values():
            annotation = str(param.annotation)
            assert "DataFrame" not in annotation, (
                f"{getattr(fn, '__qualname__', fn)} parameter {param.name!r} "
                f"is annotated {annotation!r}"
            )
            assert "pandas" not in annotation and not re.search(r"\bpd\.", annotation), (
                f"{getattr(fn, '__qualname__', fn)} parameter {param.name!r} "
                f"is annotated {annotation!r}"
            )


def test_no_column_name_literals_in_verdict_logic():
    for module in ("registry.py", "applicability.py"):
        source = (_SURVEY_DIR / module).read_text(encoding="utf-8")
        strings = [
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        for name in _FORBIDDEN_COLUMN_LITERALS:
            offenders = [s for s in strings if name in s.lower()]
            assert not offenders, (
                f"{module} hard-codes benchmark column name {name!r}: {offenders[:3]}"
            )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
