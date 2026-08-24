"""Paper Eq. 14–17 normalisation and stability helpers."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .constants import METRIC_OPTIONS


def eq14_log_normalize_series(values: pd.Series) -> pd.Series:
    """Paper Eq. 14 log-normalisation → [0, 1], 0 = best."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(0.0, index=values.index, dtype=float)

    log_lo = math.log1p(float(valid.min()))
    log_hi = math.log1p(float(valid.max()))
    den = log_hi - log_lo
    if den < 1e-30:
        return pd.Series(0.0, index=values.index, dtype=float)

    return ((np.log1p(numeric) - log_lo) / den).fillna(0.0)


def eq14_normalize_metrics(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    suffix: str = "_norm",
) -> pd.DataFrame:
    """Add or overwrite per-metric Eq. 14 normalised columns."""
    metrics = metrics or METRIC_OPTIONS
    result = df.copy()
    for metric in metrics:
        target_col = metric if suffix == "" else f"{metric}{suffix}"
        result[target_col] = eq14_log_normalize_series(result[metric])
    return result


def eq14_add_composite_score(
    df: pd.DataFrame,
    weights: tuple[float, float, float] | None = None,
    score_col: str = "C_score",
    norm_suffix: str = "_norm",
) -> pd.DataFrame:
    """Combine Eq. 14 metric scores into one weighted composite score."""
    weights = weights or (1 / 3, 1 / 3, 1 / 3)
    result = df.copy()
    result[score_col] = 0.0
    for weight, metric in zip(weights, METRIC_OPTIONS):
        result[score_col] += weight * result[f"{metric}{norm_suffix}"]
    return result


def eq14_prepare_scored_frame(
    df: pd.DataFrame,
    weights: tuple[float, float, float] | None = None,
    score_col: str = "C_score",
) -> pd.DataFrame:
    """Apply Eq. 14 metric normalisation and composite score calculation."""
    return eq14_add_composite_score(eq14_normalize_metrics(df), weights=weights, score_col=score_col)


def eq17_pairwise_density(scores: pd.Series) -> float:
    """Paper Eq. 16 average pairwise distance across settings."""
    values = pd.to_numeric(scores, errors="coerce").dropna().tolist()
    n = len(values)
    if n <= 1:
        return 0.0
    total = sum(abs(values[i] - values[j]) for i in range(n) for j in range(i + 1, n))
    return total / (n * (n - 1) / 2)


def eq17_build_stability_scores(
    df: pd.DataFrame,
    score_col: str = "C_score",
    model_col: str = "xml_file",
    setting_cols: tuple[str, ...] = ("data_volume", "servers"),
    avg_col: str = "avg",
    count_col: str | None = None,
    delta_weight: float = 0.5,
    avg_weight: float = 0.5,
) -> pd.DataFrame:
    """Aggregate Eq. 16 and Eq. 17 scores per model across settings."""
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby([model_col, *setting_cols], as_index=False)[score_col]
        .mean()
    )

    rows: list[dict[str, Any]] = []
    for model, scores in grouped.groupby(model_col)[score_col]:
        row = {
            model_col: model,
            "delta": eq17_pairwise_density(scores),
            avg_col: float(scores.mean()),
        }
        if count_col is not None:
            row[count_col] = int(len(scores))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["d_tilde"] = delta_weight * result["delta"] + avg_weight * result[avg_col]
    return result.sort_values("d_tilde").reset_index(drop=True)
