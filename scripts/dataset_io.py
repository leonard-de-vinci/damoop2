"""Locate and read the assembled sweep dataset (Parquet preferred, CSV fallback).

The assembler writes Output/sweep_ml.parquet by default (Parquet is ~20x
smaller and far faster to load than the old multi-GB CSV), but every reader
below still accepts a .csv path so older assembled datasets keep working.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.generate_profile import PROJECT_ROOT

SWEEP_PARQUET = PROJECT_ROOT / "Output" / "sweep_ml.parquet"
SWEEP_CSV = PROJECT_ROOT / "Output" / "sweep_ml.csv"


def default_sweep_dataset() -> Path:
    """Prefer the Parquet dataset; fall back to a legacy CSV if that's all there is."""
    return SWEEP_PARQUET if SWEEP_PARQUET.exists() or not SWEEP_CSV.exists() else SWEEP_CSV


def read_sweep_dataset(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read the dataset by suffix. ``columns`` not present in the file are skipped."""
    if path.suffix == ".parquet":
        if columns is not None:
            import pyarrow.parquet as pq
            available = set(pq.ParquetFile(path).schema_arrow.names)
            columns = [c for c in columns if c in available]
        return pd.read_parquet(path, columns=columns)
    if columns is not None:
        wanted = set(columns)
        return pd.read_csv(path, usecols=lambda c: c in wanted)
    return pd.read_csv(path)
