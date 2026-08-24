"""Validate the temporal scenario dataset (Output/scenario_ml.parquet).

Memory-bounded: streams the Parquet in batches (the dataset can be hundreds of millions
of rows). Checks structure, target sanity, baseline-mix correctness (mix 0 reproduces the
original sweep dataset totals), frequency bounds, and exact time-cost reconstruction.

CLI:
    python -m scripts.validate_scenario_dataset \
        --parquet Output/scenario_ml.parquet --csv Output/sweep_ml.parquet
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.generate_profile import PROJECT_ROOT
from lib.constants import DEFAULT_QUERY_FREQUENCIES, QUERY_COST_STEMS

TARGETS = ["total_time_cost", "total_environmental_cost", "total_financial_cost"]
FREQ_LO, FREQ_HI = 0.2, 5.0
KEYS = ["profile_id", "xml_file", "data_volume", "servers"]
FREQ_COLS = [f"freq_{s}" for s in QUERY_COST_STEMS]


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  —  {detail}" if detail else ""))
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Validate the scenario (temporal) dataset.")
    p.add_argument("--parquet", type=Path, default=PROJECT_ROOT / "Output" / "scenario_ml.parquet")
    p.add_argument("--csv", type=Path, default=None,
                   help="Sweep dataset; defaults to Output/sweep_ml.parquet (or legacy .csv).")
    args = p.parse_args()
    if not args.parquet.exists():
        raise SystemExit(f"Parquet not found: {args.parquet}")

    from scripts.dataset_io import default_sweep_dataset, read_sweep_dataset
    sweep_dataset = args.csv or default_sweep_dataset()
    if not sweep_dataset.exists():
        raise SystemExit(f"Sweep dataset not found: {sweep_dataset}")

    pf = pq.ParquetFile(args.parquet)
    total = pf.metadata.num_rows
    print(f"Streaming {args.parquet} ({total:,} rows)...\n")

    # Structural row count comes from the source sweep dataset — each frequency
    # mix in the scenario dataset must reproduce exactly this many rows.
    if sweep_dataset.suffix == ".parquet":
        n_struct = pq.ParquetFile(sweep_dataset).metadata.num_rows
    else:
        with sweep_dataset.open(encoding="utf-8") as fh:
            n_struct = sum(1 for _ in fh) - 1

    need = KEYS + ["mix_id"] + TARGETS + FREQ_COLS
    mix_counts: dict[int, int] = {}
    tstat = {t: {"min": math.inf, "max": -math.inf, "nan": False, "nonpos": False} for t in TARGETS}
    fminmax = {fc: [math.inf, -math.inf] for fc in FREQ_COLS}
    mix0_parts: list[pd.DataFrame] = []
    sample_key: dict | None = None
    cell_parts: list[pd.DataFrame] = []

    for batch in pf.iter_batches(batch_size=500_000, columns=need):
        d = batch.to_pandas()
        for k, v in d["mix_id"].value_counts().items():
            mix_counts[int(k)] = mix_counts.get(int(k), 0) + int(v)
        for t in TARGETS:
            col = d[t].astype("float64")
            tstat[t]["min"] = min(tstat[t]["min"], float(col.min()))
            tstat[t]["max"] = max(tstat[t]["max"], float(col.max()))
            tstat[t]["nan"] = tstat[t]["nan"] or not bool(np.isfinite(col).all())
            tstat[t]["nonpos"] = tstat[t]["nonpos"] or bool((col <= 0).any())
        nb = d[d["mix_id"] != 0]
        if len(nb):
            for fc in FREQ_COLS:
                fminmax[fc][0] = min(fminmax[fc][0], int(nb[fc].min()))
                fminmax[fc][1] = max(fminmax[fc][1], int(nb[fc].max()))
        m0 = d[d["mix_id"] == 0]
        if len(m0):
            mix0_parts.append(m0[KEYS + TARGETS + FREQ_COLS].copy())
        if sample_key is None:
            r = d.iloc[0]
            sample_key = {k: r[k] for k in KEYS}
        mask = np.ones(len(d), bool)
        for k, v in sample_key.items():
            mask &= (d[k] == v).values
        if mask.any():
            cell_parts.append(d[mask][KEYS + FREQ_COLS + ["total_time_cost", "mix_id"]].copy())

    n_mix = len(mix_counts)
    results: list[bool] = []

    # 1. Structure
    print("1. STRUCTURE")
    results.append(_check("row count == n_struct x n_mixes", total == n_struct * n_mix,
                          f"{total:,} = {n_struct:,} x {n_mix}"))
    results.append(_check("each mix has n_struct rows", all(c == n_struct for c in mix_counts.values()),
                          f"{n_mix} mixes, counts {min(mix_counts.values())}–{max(mix_counts.values())}"))
    results.append(_check("mix_id contiguous 0..n-1", sorted(mix_counts) == list(range(n_mix)),
                          f"{n_mix} mixes"))

    # 2. Targets
    print("\n2. TARGET SANITY")
    for t in TARGETS:
        s = tstat[t]
        results.append(_check(f"'{t}' positive & finite", not s["nan"] and not s["nonpos"],
                              f"min={s['min']:.3g} max={s['max']:.3g}"))

    # 3. Baseline mix correctness
    print("\n3. BASELINE MIX CORRECTNESS (mix 0 == original sweep dataset totals)")
    m0 = pd.concat(mix0_parts, ignore_index=True)
    orig = read_sweep_dataset(sweep_dataset, KEYS + TARGETS)
    mg = m0.merge(orig, on=KEYS, suffixes=("_sc", "_orig"))
    for t in TARGETS:
        a = mg[f"{t}_sc"].astype("float64").values
        b = mg[f"{t}_orig"].astype("float64").values
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-9)
        results.append(_check(f"'{t}' relative err < 1e-4", float(rel.max()) < 1e-4,
                              f"max rel err {rel.max():.2e} (float32 storage)"))

    # 4. Frequency bounds
    print("\n4. FREQUENCY BOUNDS (mix 0 = baseline; mixes 1+ within [0.2x, 5x])")
    base_ok = all(int(m0[f"freq_{s}"].iloc[0]) == DEFAULT_QUERY_FREQUENCIES[s] for s in QUERY_COST_STEMS)
    results.append(_check("mix 0 freqs == DEFAULT_QUERY_FREQUENCIES", base_ok))
    bad = []
    for s in QUERY_COST_STEMS:
        base = DEFAULT_QUERY_FREQUENCIES[s]
        lo, hi = max(1, int(np.floor(base * FREQ_LO))), int(np.ceil(base * FREQ_HI))
        mn, mx = fminmax[f"freq_{s}"]
        if mn < lo or mx > hi:
            bad.append(f"{s}[{mn},{mx}]∉[{lo},{hi}]")
    results.append(_check("all sampled freqs within per-query bounds", not bad,
                          "ok" if not bad else ", ".join(bad[:4])))

    # 5. Exact reconstruction (time cost == sum(unit*freq) for a sample cell across all mixes)
    print("\n5. EXACT RECONSTRUCTION (time cost == sum(unit*freq), sample cell, all mixes)")
    sk = sample_key
    unit_cols = [f"{s}_time_cost" for s in QUERY_COST_STEMS]
    src = read_sweep_dataset(sweep_dataset, KEYS + unit_cols)
    srow = src[(src["profile_id"] == sk["profile_id"]) & (src["xml_file"] == sk["xml_file"]) &
               (src["data_volume"] == sk["data_volume"]) & (src["servers"] == sk["servers"])]
    if srow.empty:
        results.append(_check("sample cell present in source CSV", False,
                              f"no row matching {sk} — cannot verify reconstruction"))
        _summary(results)
        return
    units = srow[unit_cols].iloc[0].astype("float64").values
    cell = pd.concat(cell_parts, ignore_index=True)
    fmat = cell[FREQ_COLS].astype("float64").values
    expected = fmat @ units
    stored = cell["total_time_cost"].astype("float64").values
    rel = np.abs(expected - stored) / np.maximum(np.abs(stored), 1e-9)
    results.append(_check("time cost == sum(unit*freq) across all mixes (rel err < 1e-4)",
                          float(rel.max()) < 1e-4, f"max rel err {rel.max():.2e} over {len(cell)} mixes"))

    _summary(results)


def _summary(results: list[bool]) -> None:
    n_pass = sum(results)
    print("\n" + "=" * 60)
    print(f"SUMMARY: {n_pass}/{len(results)} checks passed")
    print("ALL CHECKS PASSED — scenario dataset valid." if n_pass == len(results)
          else f"{len(results) - n_pass} CHECK(S) FAILED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
