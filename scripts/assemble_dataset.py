"""Assemble the ML training dataset from the archived sweep outputs.

Walks Output/sweep/profile_S*/GeneratedDMs/, parses every costs.txt via
lib.parse_costs.build_costs_dataframe, joins each row with its profile's
params.json (selectivities, index flags, customer/order counts), and writes
one master dataset — Parquet by default (~20x smaller and much faster to
load than CSV); pass an --out ending in .csv for the legacy format.

Profiles are parsed in parallel (one worker per CPU core) and appended to the
output as they complete, so memory stays flat regardless of sweep size.

CLI:
    python -m scripts.assemble_dataset --archive-dir Output/sweep \\
        --out Output/sweep_ml.parquet --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.generate_profile import NON_KEY_COLUMNS, PROJECT_ROOT
from lib.parse_costs import build_costs_dataframe


def _flatten_params(params: dict) -> dict:
    """Convert nested spec dict into flat columns suitable for a DataFrame row."""
    flat: dict = {
        "profile_id": params.get("profile_id"),
        "seed_idx": params.get("seed_idx"),
        "customer_count": params.get("customer_count"),
        "order_count": params.get("order_count"),
    }
    for col in NON_KEY_COLUMNS:
        flat[f"sel_{col}"] = params.get("selectivities", {}).get(col)
        flat[f"idx_{col}"] = int(bool(params.get("index_flags", {}).get(col, False)))
    return flat


def _assemble_profile(pdir_str: str) -> pd.DataFrame | None:
    """Parse one archived profile into a DataFrame (runs in a worker process)."""
    pdir = Path(pdir_str)
    params_file = pdir / "params.json"
    gdm = pdir / "GeneratedDMs"
    if not params_file.exists() or not gdm.is_dir():
        return None

    params = json.loads(params_file.read_text(encoding="utf-8"))
    df = build_costs_dataframe(gdm)
    if df.empty:
        return None

    for k, v in _flatten_params(params).items():
        df[k] = v
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Assemble the ML training dataset from sweep outputs.")
    p.add_argument("--archive-dir", type=Path,
                   default=PROJECT_ROOT / "Output" / "sweep")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "Output" / "sweep_ml.parquet",
                   help="Output file; .parquet (default) or .csv by suffix.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="Parallel parse workers (default: CPU count - 1; 1 disables multiprocessing).")
    args = p.parse_args()

    if not args.archive_dir.is_dir():
        raise SystemExit(f"Archive dir not found: {args.archive_dir}")

    profile_dirs = sorted(
        d for d in args.archive_dir.iterdir()
        if d.is_dir() and d.name.startswith("profile_")
    )
    if not profile_dirs:
        raise SystemExit(f"No profile_* dirs in {args.archive_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # Stream results to disk as they arrive: the first completed profile fixes
    # the column schema, every later frame is aligned to it. Nothing but the
    # in-flight frames is ever held in memory, so this scales to any sweep size.
    to_parquet = args.out.suffix == ".parquet"
    pq_writer: pq.ParquetWriter | None = None
    pq_schema: pa.Schema | None = None
    columns: list[str] | None = None
    total_rows = 0
    per_profile_rows: list[int] = []
    n_profiles = 0

    def _write(pdir: Path, df: pd.DataFrame | None) -> None:
        nonlocal pq_writer, pq_schema, columns, total_rows, n_profiles
        if df is None:
            print(f"  skipping {pdir.name}: missing params.json/GeneratedDMs or no cost rows")
            return
        if columns is None:
            columns = list(df.columns)
            if to_parquet:
                table = pa.Table.from_pandas(df, preserve_index=False)
                pq_schema = table.schema
                pq_writer = pq.ParquetWriter(args.out, pq_schema, compression="zstd")
                pq_writer.write_table(table)
            else:
                df.to_csv(args.out, index=False)
        else:
            extra = set(df.columns) - set(columns)
            if extra:
                print(f"  WARNING {pdir.name}: dropping unexpected columns {sorted(extra)}")
            df = df.reindex(columns=columns)
            if to_parquet:
                pq_writer.write_table(pa.Table.from_pandas(df, schema=pq_schema, preserve_index=False))
            else:
                df.to_csv(args.out, mode="a", header=False, index=False)
        total_rows += len(df)
        per_profile_rows.append(len(df))
        n_profiles += 1
        print(f"  {pdir.name}: {len(df):,} rows", flush=True)

    try:
        if args.workers <= 1:
            for pdir in profile_dirs:
                _write(pdir, _assemble_profile(str(pdir)))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                # map preserves submission (sorted) order while parsing in parallel
                for pdir, df in zip(profile_dirs,
                                    pool.map(_assemble_profile, (str(d) for d in profile_dirs))):
                    _write(pdir, df)
    finally:
        if pq_writer is not None:
            pq_writer.close()

    if total_rows == 0:
        raise SystemExit("No data assembled.")

    counts = pd.Series(per_profile_rows)
    print(f"\nWrote {total_rows:,} rows x {len(columns or [])} cols to {args.out} "
          f"in {time.perf_counter() - t0:.1f}s ({args.workers} workers)")
    print(f"Profiles: {n_profiles}; per-profile counts: {counts.describe().to_dict()}")


if __name__ == "__main__":
    main()
