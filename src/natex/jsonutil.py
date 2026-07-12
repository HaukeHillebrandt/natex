"""Shared JSON coercion for results payloads (house rule: NaN never 0.0).

Extracted from ``cli._clean`` verbatim: ndarray -> list, numpy scalars ->
python scalars, non-finite floats -> None (JSON null), recurse dict/list/tuple.
"""

from __future__ import annotations

import numpy as np


def jsonable(obj):
    """Coerce ``obj`` into something ``json.dumps`` accepts, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj
