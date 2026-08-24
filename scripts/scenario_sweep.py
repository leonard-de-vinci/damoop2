"""Generate the temporal (query-frequency) dataset by reweighting the structural CSV.

The parametric sweep (Output/sweep_ml.csv) already holds every model's per-query *unit*
costs. This script samples M query-frequency mixes (LHS, log-uniform around the
DEFAULT_QUERY_FREQUENCIES baseline), reweights the 3 total costs for each mix via
lib.scenarios.reweight_costs (pure Python — no JAR runs), appends the mix as feature
columns, and streams the result to a single Parquet file.

Output row = 18 structural features + 20 frequency features + freq_total + mix_id
             + ids + 3 reweighted cost targets.

CLI:
    python -m scripts.scenario_sweep --m 50 --seed 42 \
        --csv Output/sweep_ml.csv --out Output/scenario_ml.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.lhs import lhs
from scripts.generate_profile import NON_KEY_COLUMNS, PROJECT_ROOT
from lib.scenarios import reweight_costs
from lib.constants import DEFAULT_QUERY_FREQUENCIES, QUERY_COST_STEMS

# Frequency sampling bounds: log-uniform over [LO x baseline, HI x baseline] per query.
FREQ_LO, FREQ_HI = 0.2, 5.0

ID_COLS = ["profile_id", "xml_file", "data_volume", "servers"]
STRUCT_COLS = (
    ["customer_count", "order_count"]
    + [f"sel_{c}" for c in NON_KEY_COLUMNS]
    + [f"idx_{c}" for c in NON_KEY_COLUMNS]
)
TARGETS = ["total_time_cost", "total_environmental_cost", "total_financial_cost"]


def _sample_mixes(m: int, seed: int) -> list[dict[str, int]]:
    """Return m frequency mixes. Mix 0 = exact baseline; 1..m-1 = LHS log-uniform."""
    stems = list(QUERY_COST_STEMS)
    base = np.array([DEFAULT_QUERY_FREQUENCIES[s] for s in stems], dtype=float)
    mixes: list[dict[str, int]] = [dict(DEFAULT_QUERY_FREQUENCIES)]  # anchor
    if m > 1:
        u = lhs(m - 1, len(stems), seed=seed)  # (m-1, 20) in [0,1)
        lo, hi = np.log(base * FREQ_LO), np.log(base * FREQ_HI)
        for row in u:
            vals = np.exp(lo + row * (hi - lo))
            vals = np.maximum(1, np.round(vals)).astype(int)
            mixes.append({s: int(v) for s, v in zip(stems, vals)})
    return mixes


def _arrow_schema() -> pa.schema:
    fields = [
        ("mix_id", pa.int32()),
        ("profile_id", pa.string()),
        ("xml_file", pa.string()),
        ("data_volume", pa.int64()),
        ("servers", pa.int32()),
        ("customer_count", pa.int64()),
        ("order_count", pa.int64()),
    ]
    for c in NON_KEY_COLUMNS:
        fields.append((f"sel_{c}", pa.int32()))
    for c in NON_KEY_COLUMNS:
        fields.append((f"idx_{c}", pa.int8()))
    for s in QUERY_COST_STEMS:
        fields.append((f"freq_{s}", pa.int32()))
    fields.append(("freq_total", pa.int64()))
    for t in TARGETS:
        fields.append((t, pa.float32()))
    return pa.schema(fields)


def main() -> None:
    p = argparse.ArgumentParser(description="Reweight the structural CSV across M frequency mixes.")
    p.add_argument("--csv", type=Path, default=None,
                   help="Sweep dataset; defaults to Output/sweep_ml.parquet (or legacy .csv).")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "Output" / "scenario_ml.parquet")
    p.add_argument("--m", type=int, default=50, help="Number of frequency mixes (incl. baseline).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from scripts.dataset_io import default_sweep_dataset, read_sweep_dataset
    dataset = args.csv or default_sweep_dataset()
    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")

    # Columns we need: ids, structural features, per-query unit costs, totals.
    unit_cols = [f"{s}{suf}" for s in QUERY_COST_STEMS
                 for suf in ("_time_cost", "_env_cost", "_fin_cost")]
    usecols = ID_COLS + STRUCT_COLS + TARGETS + unit_cols
    print(f"Loading {dataset} ({len(usecols)} columns)...")
    df = read_sweep_dataset(dataset, usecols)
    n = len(df)
    print(f"  {n:,} structural rows loaded.")

    mixes = _sample_mixes(args.m, args.seed)
    print(f"Generating {len(mixes)} mixes -> {n * len(mixes):,} total rows -> {args.out.name}")

    # Static columns (constant across mixes) prepared once.
    static = {c: df[c].values for c in ID_COLS + STRUCT_COLS}
    schema = _arrow_schema()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out, schema, compression="zstd")

    try:
        for mid, mix in enumerate(mixes):
            rw = reweight_costs(df, mix, DEFAULT_QUERY_FREQUENCIES)
            chunk = {
                "mix_id": np.full(n, mid, dtype=np.int32),
                "profile_id": static["profile_id"],
                "xml_file": static["xml_file"],
                "data_volume": static["data_volume"].astype(np.int64),
                "servers": static["servers"].astype(np.int32),
                "customer_count": static["customer_count"].astype(np.int64),
                "order_count": static["order_count"].astype(np.int64),
            }
            for c in NON_KEY_COLUMNS:
                chunk[f"sel_{c}"] = static[f"sel_{c}"].astype(np.int32)
                chunk[f"idx_{c}"] = static[f"idx_{c}"].astype(np.int8)
            ftot = np.zeros(n, dtype=np.int64)
            for s in QUERY_COST_STEMS:
                fv = np.full(n, mix[s], dtype=np.int32)
                chunk[f"freq_{s}"] = fv
                ftot += mix[s]
            chunk["freq_total"] = ftot
            for t in TARGETS:
                chunk[t] = rw[t].values.astype(np.float32)
            writer.write_table(pa.table(chunk, schema=schema))
            if mid % 10 == 0 or mid == len(mixes) - 1:
                print(f"  mix {mid + 1}/{len(mixes)} written")
    finally:
        writer.close()

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"\nWrote {n * len(mixes):,} rows to {args.out} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
