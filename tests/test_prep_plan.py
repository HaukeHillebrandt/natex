"""PrepPlan: declarative, validated, deterministic data-prep executor (phase task 2).

No stochastic thresholds here: the only randomness is the plan-carried
subsample seed, and the expected row picks are hardcoded from
``np.sort(np.random.default_rng(0).choice(4, size=3, replace=False)) == [1, 2, 3]``,
which is bitwise-stable by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from natex.intake.prep import OPS, ROLES, PrepFilter, PrepPlan, Subsample


def golden_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "pretest": [100.0, 90.0, 120.0, np.nan, 130.0, 95.0, 110.0, 105.0],
            "group": ["a", "b", "a", "c", "b", "a", "c", "b"],
            "city": ["berlin", "oslo", "berlin", "paris", "amsterdam", "oslo", "paris", "oslo"],
            "score": [1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0, 7.0],
            "note": ["x", None, "y", None, "q", "w", "v", "u"],
        }
    )


def golden_plan() -> PrepPlan:
    return PrepPlan(
        column_roles={"pretest": "forcing", "score": "covariate"},
        encodings={"city": "ordinal", "group": "onehot"},
        discretize={"score": 2},
        drop_cols=["id"],
        subsample=Subsample(n=3, seed=0),
        filters=[
            PrepFilter(col="pretest", op=">=", value=100),
            PrepFilter(col="group", op="in", value=["a", "b"]),
            PrepFilter(col="note", op="notna"),
        ],
    )


def expected_golden() -> pd.DataFrame:
    # Filters keep labels {0, 2, 4, 7}; subsample seed=0 picks positions
    # [1, 2, 3] of those survivors -> labels [2, 4, 7].
    return pd.DataFrame(
        {
            "pretest": [120.0, 130.0, 105.0],
            "city": [1.0, 0.0, 2.0],  # factorize(sort=True): amsterdam=0, berlin=1, oslo=2
            "score": [0.0, 1.0, 1.0],  # quantile_bins([1,2,3,7], 2): edge 2.5
            "note": ["y", "q", "u"],
            "group_a": [1.0, 0.0, 0.0],
            "group_b": [0.0, 1.0, 1.0],
        },
        index=[2, 4, 7],
    )


def test_constants() -> None:
    assert ROLES == ("treatment", "outcome", "forcing", "covariate", "time", "unit", "ignore")
    assert OPS == ("==", "!=", ">", ">=", "<", "<=", "in", "notna")


def test_golden_apply_exact() -> None:
    df = golden_df()
    out, log = golden_plan().apply(df)
    pd.testing.assert_frame_equal(out, expected_golden(), check_exact=True)
    # One human-readable line per step, exact strings.
    assert "filter pretest >= 100: 8 -> 5 rows" in log
    assert "filter group in ['a', 'b']: 5 -> 4 rows" in log
    assert "filter note notna: 4 -> 4 rows" in log
    assert "drop columns: ['id']" in log
    assert "ordinal city: 3 categories" in log
    assert "onehot group -> ['group_a', 'group_b']" in log
    assert "discretize score: 2 bins requested, 2 effective" in log
    assert "subsample n=3 seed=0: 4 -> 3 rows" in log


def test_apply_bitwise_deterministic() -> None:
    df = golden_df()
    out1, log1 = golden_plan().apply(df)
    out2, log2 = golden_plan().apply(df)
    pd.testing.assert_frame_equal(out1, out2, check_exact=True)
    assert log1 == log2


def test_apply_never_mutates_input() -> None:
    df = golden_df()
    before = df.copy(deep=True)
    golden_plan().apply(df)
    pd.testing.assert_frame_equal(df, before, check_exact=True)


def test_json_round_trip_applies_identically() -> None:
    plan = golden_plan()
    plan2 = PrepPlan.model_validate_json(plan.model_dump_json())
    out1, log1 = plan.apply(golden_df())
    out2, log2 = plan2.apply(golden_df())
    pd.testing.assert_frame_equal(out1, out2, check_exact=True)
    assert log1 == log2


def test_empty_result_after_filters_logged() -> None:
    df = golden_df()
    plan = PrepPlan(filters=[PrepFilter(col="pretest", op=">", value=1e9)])
    out, log = plan.apply(df)
    assert len(out) == 0
    assert "WARNING: 0 rows remain" in log


def test_subsample_larger_than_df_keeps_all_rows() -> None:
    df = golden_df()
    plan = PrepPlan(subsample=Subsample(n=100, seed=3))
    out, log = plan.apply(df)
    assert len(out) == 8
    assert "subsample n=100 seed=3: 8 -> 8 rows" in log


# --- NaN policy -----------------------------------------------------------


def test_filter_comparison_drops_nan_rows() -> None:
    df = pd.DataFrame({"x": [1.0, 6.0, np.nan, 9.0]})
    plan = PrepPlan(filters=[PrepFilter(col="x", op=">", value=5)])
    out, _ = plan.apply(df)
    assert out["x"].tolist() == [6.0, 9.0]  # NaN comparison is False, never coerced


def test_ordinal_nan_stays_nan_never_zero() -> None:
    df = pd.DataFrame({"c": ["b", None, "a"]})
    plan = PrepPlan(encodings={"c": "ordinal"})
    out, _ = plan.apply(df)
    assert out["c"].iloc[0] == 1.0
    assert np.isnan(out["c"].iloc[1])
    assert out["c"].iloc[1] != 0.0
    assert out["c"].iloc[2] == 0.0


def test_discretize_nan_stays_nan() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 7.0, 3.0]})
    plan = PrepPlan(discretize={"x": 2})
    out, _ = plan.apply(df)
    assert np.isnan(out["x"].iloc[2])
    assert out["x"].iloc[2] != 0.0


# --- rejections (each names the offender) ---------------------------------


@pytest.mark.parametrize(
    "plan",
    [
        PrepPlan(column_roles={"ghost": "outcome"}),
        PrepPlan(encodings={"ghost": "onehot"}),
        PrepPlan(discretize={"ghost": 2}),
        PrepPlan(drop_cols=["ghost"]),
        PrepPlan(filters=[PrepFilter(col="ghost", op="notna")]),
    ],
    ids=["column_roles", "encodings", "discretize", "drop_cols", "filters"],
)
def test_rejects_unknown_column_by_name(plan: PrepPlan) -> None:
    with pytest.raises(ValueError, match="ghost"):
        plan.validate_against(golden_df())


def test_rejects_unknown_role_by_name() -> None:
    plan = PrepPlan(column_roles={"score": "confounder"})
    with pytest.raises(ValueError, match="confounder"):
        plan.validate_against(golden_df())


def test_rejects_two_outcomes() -> None:
    plan = PrepPlan(column_roles={"score": "outcome", "pretest": "outcome"})
    with pytest.raises(ValueError, match="outcome"):
        plan.validate_against(golden_df())


def test_rejects_bins_below_two() -> None:
    plan = PrepPlan(discretize={"score": 1})
    with pytest.raises(ValueError, match="score"):
        plan.validate_against(golden_df())


def test_rejects_discretize_on_string_column() -> None:
    plan = PrepPlan(discretize={"city": 2})
    with pytest.raises(ValueError, match="city"):
        plan.validate_against(golden_df())


def test_rejects_encoding_plus_discretize_same_column() -> None:
    plan = PrepPlan(encodings={"score": "ordinal"}, discretize={"score": 2})
    with pytest.raises(ValueError, match="score"):
        plan.validate_against(golden_df())


def test_rejects_in_filter_with_scalar_value() -> None:
    plan = PrepPlan(filters=[PrepFilter(col="group", op="in", value="a")])
    with pytest.raises(ValueError, match="group"):
        plan.validate_against(golden_df())


def test_rejects_comparison_filter_with_none_value() -> None:
    plan = PrepPlan(filters=[PrepFilter(col="score", op="==")])
    with pytest.raises(ValueError, match="score"):
        plan.validate_against(golden_df())


def test_rejects_drop_plus_role_conflict() -> None:
    plan = PrepPlan(drop_cols=["score"], column_roles={"score": "covariate"})
    with pytest.raises(ValueError, match="score"):
        plan.validate_against(golden_df())


def test_rejects_nonpositive_subsample_n() -> None:
    # field_validator (not gt=0: constraint keys break strict LLM schemas).
    with pytest.raises(ValueError, match="n"):
        Subsample(n=0)
