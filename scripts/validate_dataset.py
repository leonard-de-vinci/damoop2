"""Validate the assembled sweep ML dataset (Output/sweep_ml.parquet or legacy .csv).

Runs a battery of structural + semantic checks and prints a PASS/FAIL report.

CLI:
    python -m scripts.validate_dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.generate_profile import NON_KEY_COLUMNS, PROJECT_ROOT
from scripts.sweep_profiles import (
    CUSTOMER_RANGE,
    ORDER_RANGE,
    SELECTIVITY_RANGES,
)

TARGETS = ["total_time_cost", "total_environmental_cost", "total_financial_cost"]
EXPECTED_MODELS = 135
EXPECTED_VOLUMES = 6      # 1K, 10K, 100K, 1M, 10M, 100M
EXPECTED_SERVERS = 7      # 10, 50, 100, 500, 1000, 5000, 10000


def _check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f"  —  {detail}"
    print(line)
    return ok


def main() -> None:
    from scripts.dataset_io import default_sweep_dataset, read_sweep_dataset

    p = argparse.ArgumentParser(description="Validate the sweep ML dataset (.parquet or .csv).")
    p.add_argument("--csv", type=Path, default=None,
                   help="Dataset path; defaults to Output/sweep_ml.parquet (or legacy .csv).")
    args = p.parse_args()
    dataset = args.csv or default_sweep_dataset()

    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")

    print(f"Loading {dataset} ...")
    df = read_sweep_dataset(dataset)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns\n")

    results: list[bool] = []

    # ── 1. Shape ────────────────────────────────────────────────────────
    print("1. STRUCTURE")
    n_profiles = df["profile_id"].nunique()
    expected_rows = n_profiles * EXPECTED_MODELS * EXPECTED_VOLUMES * EXPECTED_SERVERS
    results.append(_check(
        "row count matches profiles x models x volumes x servers",
        len(df) == expected_rows,
        f"{len(df):,} rows; expected {n_profiles}x{EXPECTED_MODELS}x"
        f"{EXPECTED_VOLUMES}x{EXPECTED_SERVERS} = {expected_rows:,}",
    ))
    results.append(_check(
        "distinct profiles", n_profiles > 0, f"{n_profiles} profiles",
    ))
    results.append(_check(
        "distinct models per profile == 135",
        (df.groupby("profile_id")["xml_file"].nunique() == EXPECTED_MODELS).all(),
        f"range {df.groupby('profile_id')['xml_file'].nunique().min()}–"
        f"{df.groupby('profile_id')['xml_file'].nunique().max()}",
    ))
    results.append(_check(
        "distinct data_volume levels == 6",
        df["data_volume"].nunique() == EXPECTED_VOLUMES,
        f"{sorted(df['data_volume'].dropna().unique().astype(int).tolist())}",
    ))
    results.append(_check(
        "distinct server levels == 7",
        df["servers"].nunique() == EXPECTED_SERVERS,
        f"{sorted(df['servers'].dropna().unique().astype(int).tolist())}",
    ))
    per_profile = df.groupby("profile_id").size()
    results.append(_check(
        "equal row count per profile",
        per_profile.nunique() == 1,
        f"each profile has {per_profile.iloc[0]:,} rows" if per_profile.nunique() == 1
        else f"UNEQUAL: {per_profile.min()}–{per_profile.max()}",
    ))

    # ── 2. Missing / invalid values ─────────────────────────────────────
    print("\n2. COMPLETENESS")
    key_cols = ["profile_id", "xml_file", "data_volume", "servers"] + TARGETS
    for col in key_cols:
        n_missing = df[col].isna().sum()
        results.append(_check(
            f"no missing values in '{col}'", n_missing == 0,
            f"{n_missing:,} missing" if n_missing else "complete",
        ))

    # ── 3. Target sanity ────────────────────────────────────────────────
    print("\n3. TARGET SANITY")
    for col in TARGETS:
        vals = pd.to_numeric(df[col], errors="coerce")
        all_positive = (vals > 0).all()
        no_inf = np.isfinite(vals).all()
        results.append(_check(
            f"'{col}' all positive & finite",
            all_positive and no_inf,
            f"min={vals.min():.3g}, max={vals.max():.3g}",
        ))

    # ── 4. Profile params correctly broadcast (constant within profile) ──
    print("\n4. PARAM BROADCAST (constant within each profile)")
    param_cols = (
        ["customer_count", "order_count"]
        + [f"sel_{c}" for c in NON_KEY_COLUMNS]
        + [f"idx_{c}" for c in NON_KEY_COLUMNS]
    )
    nonconstant = []
    for col in param_cols:
        if col not in df.columns:
            nonconstant.append(f"{col} (missing)")
            continue
        if (df.groupby("profile_id")[col].nunique() != 1).any():
            nonconstant.append(col)
    results.append(_check(
        "all 18 profile params constant within each profile",
        not nonconstant,
        "all constant" if not nonconstant else f"VARYING: {nonconstant}",
    ))

    # ── 5. Selectivity bounds ───────────────────────────────────────────
    print("\n5. SELECTIVITY BOUNDS (per-profile values within configured ranges)")
    prof = df.drop_duplicates("profile_id")
    for col in NON_KEY_COLUMNS:
        lo, hi = SELECTIVITY_RANGES[col]
        vals = prof[f"sel_{col}"]
        within = ((vals >= lo) & (vals <= hi)).all()
        results.append(_check(
            f"sel_{col} in [{lo}, {hi}]", within,
            f"observed {int(vals.min())}–{int(vals.max())}",
        ))

    # ── 6. Index flags binary ───────────────────────────────────────────
    print("\n6. INDEX FLAGS (0/1 only)")
    bad_idx = []
    for col in NON_KEY_COLUMNS:
        uniq = set(prof[f"idx_{col}"].unique().tolist())
        if not uniq.issubset({0, 1}):
            bad_idx.append(f"{col}={uniq}")
    results.append(_check(
        "all 8 index flags are 0/1", not bad_idx,
        "binary" if not bad_idx else f"NON-BINARY: {bad_idx}",
    ))

    # ── 7. Count bounds ─────────────────────────────────────────────────
    print("\n7. COUNT BOUNDS")
    cust = prof["customer_count"]
    ordr = prof["order_count"]
    results.append(_check(
        f"customer_count in [{CUSTOMER_RANGE[0]:,}, {CUSTOMER_RANGE[1]:,}]",
        ((cust >= CUSTOMER_RANGE[0]) & (cust <= CUSTOMER_RANGE[1])).all(),
        f"observed {int(cust.min()):,}–{int(cust.max()):,}",
    ))
    results.append(_check(
        f"order_count in [{ORDER_RANGE[0]:,}, {ORDER_RANGE[1]:,}]",
        ((ordr >= ORDER_RANGE[0]) & (ordr <= ORDER_RANGE[1])).all(),
        f"observed {int(ordr.min()):,}–{int(ordr.max()):,}",
    ))

    # ── 8. Variation check (each sampled dim actually varies) ───────────
    print("\n8. VARIATION (each sampled dimension spans >1 value across profiles)")
    flat = []
    for col in NON_KEY_COLUMNS:
        if prof[f"sel_{col}"].nunique() <= 1:
            flat.append(f"sel_{col}")
        if prof[f"idx_{col}"].nunique() <= 1:
            flat.append(f"idx_{col}")
    if prof["customer_count"].nunique() <= 1:
        flat.append("customer_count")
    if prof["order_count"].nunique() <= 1:
        flat.append("order_count")
    results.append(_check(
        "all 18 dimensions vary across profiles", not flat,
        "all vary" if not flat else f"FLAT: {flat}",
    ))

    # ── 9. Duplicate rows ───────────────────────────────────────────────
    print("\n9. DUPLICATES")
    key = ["profile_id", "xml_file", "data_volume", "servers"]
    n_dupes = df.duplicated(subset=key).sum()
    results.append(_check(
        "no duplicate (profile, model, volume, servers) rows",
        n_dupes == 0, f"{n_dupes:,} duplicates",
    ))

    # ── 10. Cross-profile cost variance (catches the conf/ bug) ─────────
    # For a fixed (model, volume, servers) cell, cost MUST vary across profiles.
    # If it doesn't, the sweep never applied per-profile inputs (conf/ frozen).
    print("\n10. CROSS-PROFILE COST VARIANCE (the conf/ bug check)")
    cell_keys = ["xml_file", "data_volume", "servers"]
    for tgt in TARGETS:
        nun = df.groupby(cell_keys)[tgt].nunique()
        n_varying = (nun > 1).sum()
        frac = n_varying / len(nun) if len(nun) else 0.0
        results.append(_check(
            f"'{tgt}' varies across profiles in most cells",
            frac > 0.5,
            f"{n_varying:,}/{len(nun):,} cells vary ({frac:.0%}) — "
            f"if ~0%, conf/ was not synced before Phase 2",
        ))

    # ── Summary ─────────────────────────────────────────────────────────
    n_pass = sum(results)
    n_total = len(results)
    print("\n" + "=" * 60)
    print(f"SUMMARY: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("ALL CHECKS PASSED — dataset is valid for ML training.")
    else:
        print(f"{n_total - n_pass} CHECK(S) FAILED — review above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
