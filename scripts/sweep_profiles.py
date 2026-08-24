"""Generate N parametric profiles via Latin Hypercube sampling.

CLI:
    python -m scripts.sweep_profiles --n 25 --seed 42 \\
        --out Input/profiles/sweep
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.generate_profile import (
    NON_KEY_COLUMNS,
    PROJECT_ROOT,
    generate_profile,
)
from scripts.lhs import lhs


# Per-column selectivity ranges, log-uniform. Bounded by each column's
# semantic meaning to avoid training ML on physically impossible scenarios.
# Baselines (profile_A): w_name=1, w_city=2, w_tax=10, c_last=10,
# balance=3, c_credit=2, o_carrier_id=100, o_ol_cnt=15.
SELECTIVITY_RANGES: dict[str, tuple[int, int]] = {
    "w_name":       (1, 5),       # Mostly unique per warehouse
    "w_city":       (1, 20),      # Cities cluster a few warehouses
    "w_tax":        (1, 50),      # Limited tax brackets
    "c_last":       (1, 30),      # Last names: 3 baseline -> 30 degraded
    "balance":      (1, 20),      # Balance value clusters
    "c_credit":     (1, 10),      # Few credit categories
    "o_carrier_id": (10, 1000),   # 10 carriers, scales with order volume
    "o_ol_cnt":     (5, 100),     # Order line counts (5-15 standard)
}

# Absolute row counts (log-uniform). Baseline A: Customer=30M, Order=60M.
CUSTOMER_RANGE: tuple[int, int] = (100_000, 100_000_000)
ORDER_RANGE:    tuple[int, int] = (200_000, 200_000_000)


def _log_unmap(u: float, low: float, high: float) -> float:
    """Map u in [0, 1) to a log-uniform sample in [low, high]."""
    return math.exp(math.log(low) + u * (math.log(high) - math.log(low)))


def _decode_spec(row: np.ndarray, profile_id: str, seed_idx: int) -> dict:
    cols = list(NON_KEY_COLUMNS)
    sels: dict[str, int] = {}
    for i, col in enumerate(cols):
        lo, hi = SELECTIVITY_RANGES[col]
        sels[col] = max(1, int(round(_log_unmap(float(row[i]), lo, hi))))
    flags: dict[str, bool] = {
        col: bool(row[8 + i] >= 0.5) for i, col in enumerate(cols)
    }
    cust = int(round(_log_unmap(float(row[16]), *CUSTOMER_RANGE)))
    ordr = int(round(_log_unmap(float(row[17]), *ORDER_RANGE)))
    return {
        "profile_id": profile_id,
        "seed_idx": seed_idx,
        "selectivities": sels,
        "index_flags": flags,
        "customer_count": cust,
        "order_count": ordr,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate N parametric profiles via LHS.")
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "Input" / "profiles" / "sweep",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sample = lhs(args.n, 18, seed=args.seed)
    width = max(2, len(str(args.n - 1)))

    summary: list[dict] = []
    for i in range(args.n):
        pid = f"S{i:0{width}d}"
        dest = args.out / f"profile_{pid}"
        spec = _decode_spec(sample[i], profile_id=pid, seed_idx=i)
        generate_profile(spec, dest)
        try:
            rel = str(dest.relative_to(PROJECT_ROOT))
        except ValueError:
            rel = str(dest)
        summary.append({
            "profile_id": pid,
            "dest": rel,
            "customer_count": spec["customer_count"],
            "order_count": spec["order_count"],
        })

    (args.out / "sweep_index.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Generated {args.n} profiles under {args.out}")


if __name__ == "__main__":
    main()
