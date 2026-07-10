import json

import numpy as np
import pandas as pd

from natex.intake.profiler import profile


def test_profile_panel_dataset():
    rng = np.random.default_rng(0)
    units, years = 40, 10
    df = pd.DataFrame(
        {
            "state": np.repeat([f"s{i}" for i in range(units)], years),
            "year": np.tile(np.arange(1990, 1990 + years), units),
            "smoking": rng.normal(50, 5, units * years),
            "treated": np.zeros(units * years, dtype=int),
        }
    )
    prof = profile(df)
    assert prof.n_rows == 400
    assert ("state", "year") in prof.panel_candidates
    assert "smoking" in prof.forcing_candidates
    assert "treated" in prof.treatment_candidates
    parsed = json.loads(prof.to_json())
    assert parsed["n_rows"] == 400


def test_time_like_detection():
    df = pd.DataFrame({"year": [2001, 2002], "score": [1.5, 2.5], "when": ["2020-01-01", "2020-02-01"]})
    prof = profile(df)
    by_name = {c.name: c for c in prof.columns}
    assert by_name["year"].is_time_like
    assert by_name["when"].is_time_like
    assert not by_name["score"].is_time_like
