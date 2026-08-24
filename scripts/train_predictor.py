"""Train a cost-evolution model and save it to the model registry for the Predict page.

Trains 3 XGBoost regressors (time / env / financial) on log1p targets over a subsample of
Output/scenario_ml.parquet. Records held-out (by-profile) R2/MAE per target, then saves the
models + feature order + model-id encoding + input ranges + metadata to
Output/models/<name>.pkl. The Predict page lists every *.pkl in that folder.

CLI:
    python -m scripts.train_predictor --name cost_evolution --rows 1500000
"""
from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

from scripts.generate_profile import NON_KEY_COLUMNS, PROJECT_ROOT
from scripts.sweep_profiles import SELECTIVITY_RANGES, CUSTOMER_RANGE, ORDER_RANGE
from lib.constants import DEFAULT_QUERY_FREQUENCIES, QUERY_COST_STEMS

TARGETS = ["total_time_cost", "total_environmental_cost", "total_financial_cost"]
STRUCT = (["customer_count", "order_count"]
          + [f"sel_{c}" for c in NON_KEY_COLUMNS]
          + [f"idx_{c}" for c in NON_KEY_COLUMNS])
FREQ = [f"freq_q{i}" for i in range(1, 21)]
FEATURES = STRUCT + FREQ + ["data_volume", "servers", "model_code"]
REGISTRY = PROJECT_ROOT / "Output" / "models"


def main() -> None:
    p = argparse.ArgumentParser(description="Train + register the cost-evolution predictor.")
    p.add_argument("--parquet", type=Path, default=PROJECT_ROOT / "Output" / "scenario_ml.parquet")
    p.add_argument("--name", default="cost_evolution", help="Registry name -> Output/models/<name>.pkl")
    p.add_argument("--rows", type=int, default=1_500_000, help="Approx training rows (subsample).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pf = pq.ParquetFile(args.parquet)
    total = pf.metadata.num_rows
    frac = min(1.0, args.rows / total)
    print(f"Sampling ~{frac:.2%} of {total:,} rows...")
    parts = []
    for batch in pf.iter_batches(batch_size=500_000):
        d = batch.to_pandas()
        parts.append(d.sample(frac=frac, random_state=args.seed))
    df = pd.concat(parts, ignore_index=True)
    del parts
    print(f"  {len(df):,} rows.")

    cat = df["xml_file"].astype("category")
    df["model_code"] = cat.cat.codes
    model_code_map = {name: i for i, name in enumerate(cat.cat.categories)}

    X = df[FEATURES]
    groups = df["profile_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr, te = next(gss.split(X, np.zeros(len(X)), groups=groups))

    # Volume normalization (Option A): the target is the log-cost RESIDUAL after removing
    # the (data_volume, servers, model) group mean. This stops the 6-orders-of-magnitude
    # volume axis from dominating, forcing the model to learn the selectivity/index/count/
    # workload signal — and it anchors out-of-distribution configs (base + bounded residual).
    # At inference: cost = expm1(model.predict(X) + base[(volume, servers, model)]).
    KEYCOLS = ["data_volume", "servers", "xml_file"]
    models, metrics, base_tables = {}, {}, {}
    for t in TARGETS:
        logy = np.log1p(df[t].values)
        tmp = df[KEYCOLS].copy()
        tmp["_logy"] = logy
        g = tmp.groupby(KEYCOLS)["_logy"]
        base_row = g.transform("mean").values          # per-row group mean
        base_tables[t] = {(int(k[0]), int(k[1]), str(k[2])): float(v)
                          for k, v in g.mean().items()}   # (vol, srv, model) -> mean log1p
        resid = logy - base_row

        m = XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.1,
                         tree_method="hist", n_jobs=-1, random_state=args.seed)
        m.fit(X.iloc[tr], resid[tr])
        # Evaluate on RECONSTRUCTED log-cost (base + predicted residual) vs true log-cost.
        recon = m.predict(X.iloc[te]) + base_row[te]
        metrics[t] = {"r2": round(float(r2_score(logy[te], recon)), 4),
                      "mae": round(float(mean_absolute_error(logy[te], recon)), 4)}
        models[t] = m
        print(f"  {t}: R2={metrics[t]['r2']}  MAE={metrics[t]['mae']}  (reconstructed)")

    bundle = {
        "models": models,
        "normalized": True,            # targets are residuals; reconstruct with base_tables
        "base_tables": base_tables,    # target -> {(data_volume, servers, xml_file): mean_log1p}
        "base_key": ["data_volume", "servers", "model_id"],
        "features": FEATURES,
        "struct_cols": STRUCT,
        "freq_cols": FREQ,
        "model_code_map": model_code_map,
        "default_freqs": dict(DEFAULT_QUERY_FREQUENCIES),
        "selectivity_ranges": dict(SELECTIVITY_RANGES),
        "customer_range": CUSTOMER_RANGE,
        "order_range": ORDER_RANGE,
        "query_stems": list(QUERY_COST_STEMS),
        "meta": {
            "name": args.name,
            "algorithm": "XGBoost (400 trees, depth 8)",
            "created": datetime.now().isoformat(timespec="seconds"),
            "dataset": args.parquet.name,
            "dataset_rows": int(total),
            "train_rows": int(len(df)),
            "held_out_metrics": metrics,   # by-profile split (20% held out)
        },
        "note": "predict log1p(cost); apply np.expm1 to recover real cost.",
    }
    REGISTRY.mkdir(parents=True, exist_ok=True)
    out = REGISTRY / f"{args.name}.pkl"
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved {out} ({out.stat().st_size/1024/1024:.1f} MB) — registry name '{args.name}'.")


if __name__ == "__main__":
    main()
