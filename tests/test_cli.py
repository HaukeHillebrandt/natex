import json

import numpy as np
from typer.testing import CliRunner

from natex.cli import app
from natex.data.synthetic import make_synthetic


def test_discover_end_to_end(tmp_path):
    ds, _ = make_synthetic(n=500, zeta=4.0, kind="real", rng=np.random.default_rng(0))
    csv = tmp_path / "d.csv"
    ds.df.to_csv(csv, index=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["discover", str(csv), "--treatment", "T", "--outcome", "y",
         "--k", "25", "--q", "9", "--seed", "0", "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "out" / "results.json").read_text())
    assert payload["scan"]["model"] == "normal"
    assert 0.0 < payload["scan"]["p_value"] <= 1.0
    assert len(payload["discoveries"]) > 0
    assert payload["effects"]["2sls"]["tau"] is not None
