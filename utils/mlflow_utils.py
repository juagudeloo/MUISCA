#!/usr/bin/env python3
"""Helpers for logging MUISCA runs to MLflow.

Deliberately small: the scripts call the MLflow API directly (`mlflow.start_run`,
`mlflow.log_metrics`, ...), and this module only covers the two things MLflow does not do
for us.

Backend: SQLite (see default_tracking_uri). MLflow 3.x deprecated the filesystem store, and
SQLite additionally enables the Model Registry. This is safe while the ablation variations
run sequentially in one job; parallelizing them means moving to a tracking server.

Note on what MLflow is and is not doing here: `output/<experiment>/<variation>/` remains the
functional source of truth. `finetune.py` resolves checkpoints by path and reads the
`experiment_config.json` beside them, and the analysis pipeline walks the same layout.
Anything logged to MLflow is an additional copy for browsing and comparison.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict

DEFAULT_TRACKING_DB = Path("/scratchsan/observatorio/juagudeloo/MUISCA/mlruns.db")

# MLflow rejects parameter values longer than this. Our logtau grid is 45 numbers, which
# blows past it -- and a single oversized value makes the whole log_params call fail, so it
# is summarized rather than left to explode.
_MAX_PARAM_CHARS = 480


def default_tracking_uri() -> str:
    """SQLite tracking URI of the shared run database (``MLFLOW_TRACKING_URI`` wins if set).

    MLflow 3.x puts the filesystem store in maintenance mode and refuses it unless
    ``MLFLOW_ALLOW_FILE_STORE=true``, so SQLite is the supported backend; it also enables the
    Model Registry. SQLite over NFS does not tolerate concurrent writers: the ablation
    flavors run one after another in one SLURM job, so there is one writer. If the flavors
    become a job array, run a tracking server (``mlflow server --backend-store-uri
    sqlite:///...``) and point ``MLFLOW_TRACKING_URI`` at it instead.

    Returns:
        The URI, e.g. ``sqlite:///.../mlruns.db``.

    Examples:
        >>> import os
        >>> os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///tmp/x.db"; default_tracking_uri()
        'sqlite:///tmp/x.db'
        >>> del os.environ["MLFLOW_TRACKING_URI"]
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    DEFAULT_TRACKING_DB.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_TRACKING_DB}"


def _summarize_sequence(value) -> str:
    """Short description of a sequence too long to log, keeping the step if the spacing is uniform.

    The 45-node grid becomes ``"[45 values: -3 ... 1.4, step 0.1]"``. Uniform means steps
    equal to 1e-4 relative (the float32 grid's steps differ by about 2.4e-6 relative from
    rounding); otherwise, or for non-numeric entries, only the endpoints are given.

    Examples:
        >>> _summarize_sequence([0.0, 0.5, 1.0, 1.5])
        '[4 values: 0 ... 1.5, step 0.5]'
    """
    if not value:
        return "[]"
    head, tail, n = value[0], value[-1], len(value)
    try:
        nums = [float(v) for v in value]
    except (TypeError, ValueError):
        return f"[{n} values: {head} ... {tail}]"

    if n > 2:
        diffs = [b - a for a, b in zip(nums[:-1], nums[1:])]
        spread = max(diffs) - min(diffs)
        scale = max(abs(d) for d in diffs) or 1.0
        # 1e-4 relative, not something tighter: the logtau grid is built in float32, so its
        # steps disagree by ~2.4e-6 relative purely from rounding. A stricter test rejects a
        # grid that is uniform by construction and silently drops the step from the summary.
        if spread <= 1e-4 * scale:
            step = sum(diffs) / len(diffs)
            return f"[{n} values: {head:g} ... {tail:g}, step {step:g}]"
    return f"[{n} values: {head} ... {tail}]"


def flatten_params(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested configuration dict into MLflow's flat parameter names.

    Nesting becomes dotted keys (``"data_config.min_step"``); lists and tuples become
    comma-separated text, summarized by :func:`_summarize_sequence` beyond 480 characters
    (MLflow's limit, which the 45-node grid exceeds).

    Args:
        config: Nested dict (e.g. ``experiment_config.json``).
        prefix: Prefix for the keys (used in recursion).

    Returns:
        Flat dict.

    Examples:
        >>> flatten_params({"data_config": {"min_step": 110, "steps": [110, 120]}})
        {'data_config.min_step': 110, 'data_config.steps': '110, 120'}
    """
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_params(value, prefix=name))
        elif isinstance(value, (list, tuple)):
            text = ", ".join(str(v) for v in value)
            if len(text) > _MAX_PARAM_CHARS:
                text = _summarize_sequence(value)
            flat[name] = text
        else:
            flat[name] = value
    return flat


def finite_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Only the finite numeric metrics (booleans, text and NaN are dropped).

    Physics losses are NaN while the gate is closed; a NaN in an MLflow series makes its
    charts unreadable, so the series simply starts when the gate opens.

    Args:
        metrics: Name to value.

    Returns:
        Name to float, finite values only.

    Examples:
        >>> finite_metrics({"mse": 0.1, "wfa": float("nan"), "flag": True, "mode": "x"})
        {'mse': 0.1}
    """
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if math.isfinite(value):
            out[key] = value
    return out
