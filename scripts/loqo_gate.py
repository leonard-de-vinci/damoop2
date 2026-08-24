"""Phase 4 gate — can the ranker generalize to UNSEEN queries? (leave-queries-out)

The Phase-3 ranker has one output head per query, so a new query has no head.
Query generalization requires the query to be a *featurized input*. Before
building a query sweep or a new trainer, this experiment answers the cheap
feasibility question using data we already have:

    Train a unit-time-cost model on 16 of the 20 queries (queries described by
    FEATURES, not identity), then rank the 135 data models for the 4 held-out
    queries it has never seen. 5-fold CV covers all 20 queries.

Query features (parsed from Input/queries.txt): #joins, tables touched, which
joins (Customer-Order / Warehouse-Customer), #projected columns, result size,
max_time, and — config-dependent — the where-column's selectivity and index
flag under each profile. Model features: code + structure parsed from the
transformation path (depth, merges materializing each join, splits per table),
plus query x model crosses ("does this model pre-join what this query joins").

Gate metrics per unseen query, vs a strong query-agnostic baseline (rank models
by their TRUE mean cost over the 16 training queries):
  - Spearman(predicted model ranking, true model ranking)
  - time-cost regret of picking the predicted-best model

CLI:
    python -m scripts.loqo_gate --cells 500 --trees 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor

from scripts.dataset_io import default_sweep_dataset, read_sweep_dataset
from scripts.generate_profile import NON_KEY_COLUMNS, PROJECT_ROOT

NQ = 20
STEMS = [f"q{i}" for i in range(1, NQ + 1)]
UNIT_T = [f"{s}_time_cost" for s in STEMS]
SEL = [f"sel_{c}" for c in NON_KEY_COLUMNS]
IDX = [f"idx_{c}" for c in NON_KEY_COLUMNS]
KEY_COLS = {"w_ID", "c_ID", "o_ID"}
TABLES = ("Warehouse", "Customer", "Order")


# ---------------------------------------------------------------------------
# Query featurization (from Input/queries.txt)
# ---------------------------------------------------------------------------
def parse_query_specs(qfile: Path) -> list[dict]:
    specs = []
    for line in qfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("query/"):
            continue
        body, size, _freq, max_t = line.rsplit("/", 3)
        tok = body.split(",")
        i_from, i_where = tok.index("from"), tok.index("where")
        mid = tok[i_from + 1:i_where]
        where_cols = tok[i_where + 1:]
        specs.append({
            "q_n_joins": mid.count("join"),
            "q_n_proj": i_from - 1,
            "q_size": float(size),
            "q_max_time": float(max_t),
            "q_tbl_wh": int("Warehouse" in mid),
            "q_tbl_cust": int("Customer" in mid),
            "q_tbl_ord": int("Order" in mid),
            "q_join_co": int("c_o_ID" in mid),
            "q_join_wc": int("w_c_ID" in mid),
            "q_n_where": len(where_cols),
            "q_where_is_key": int(all(w in KEY_COLS for w in where_cols)),
            "q_where_wh": int(any(_COL_TABLE.get(w) == "wh" for w in where_cols)),
            "q_where_cust": int(any(_COL_TABLE.get(w) == "cust" for w in where_cols)),
            "q_where_ord": int(any(_COL_TABLE.get(w) == "ord" for w in where_cols)),
            "_where_cols": [w for w in where_cols if w in NON_KEY_COLUMNS],
        })
    return specs


QFEATS = ["q_n_joins", "q_n_proj", "q_lg_size", "q_lg_max_time",
          "q_tbl_wh", "q_tbl_cust", "q_tbl_ord", "q_join_co", "q_join_wc",
          "q_n_where", "q_where_is_key", "q_lg_sel_where", "q_idx_where",
          "q_where_wh", "q_where_cust", "q_where_ord"]

# where-column -> table it lives in (keys included)
_COL_TABLE = {"w_name": "wh", "w_city": "wh", "w_tax": "wh", "w_ID": "wh",
              "c_last": "cust", "balance": "cust", "c_credit": "cust",
              "c_ID": "cust", "w_c_ID": "cust",
              "o_carrier_id": "ord", "o_ol_cnt": "ord", "o_ID": "ord", "c_o_ID": "ord"}


# ---------------------------------------------------------------------------
# Model-structure featurization (from the xml_file transformation path)
# ---------------------------------------------------------------------------
def model_struct_features(xml_ids: list[str]) -> pd.DataFrame:
    rows = []
    for xf in xml_ids:
        ops = [] if xf.endswith(".xml") and "/" not in xf else xf.split("/")
        joined = "|".join(ops)
        rows.append({
            "xml_file": xf,
            "m_depth": len(ops),
            "m_merge_co": int("c_o_ID" in joined),
            "m_merge_wc": int("w_c_ID" in joined),
            "m_split_wh": joined.count("Split_r_Warehouse"),
            "m_split_cust": joined.count("Split_r_Customer"),
        })
    return pd.DataFrame(rows)


MFEATS = ["model_code", "m_depth", "m_merge_co", "m_merge_wc", "m_split_wh", "m_split_cust"]
CFEATS = [f"lg_{c}" for c in SEL] + IDX + ["lg_customer_count", "lg_order_count",
                                           "lg_data_volume", "lg_servers"]
XFEATS = ["x_mat_co", "x_mat_wc", "x_split_wh", "x_split_cust"]
FEATURES = CFEATS + MFEATS + QFEATS + XFEATS


def build_long(df: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    """One row per (config-cell row) x query, with config-dependent query features."""
    parts = []
    for qi, spec in enumerate(specs, start=1):
        p = df[["_cell", "xml_file", "model_code"] + CFEATS + MFEATS[1:]].copy()
        p["y"] = df[f"q{qi}_time_cost"].to_numpy(float)
        for k, v in spec.items():
            if k.startswith("q_"):
                p[k] = v
        p["q_lg_size"] = np.log10(spec["q_size"])
        p["q_lg_max_time"] = np.log10(spec["q_max_time"])
        wc = spec["_where_cols"]
        if wc:
            sel = np.ones(len(df))
            for w in wc:
                sel = sel * df[f"sel_{w}"].to_numpy(float)
            p["q_lg_sel_where"] = np.log10(sel) / len(wc)     # geometric-mean log-selectivity
            p["q_idx_where"] = df[[f"idx_{w}" for w in wc]].mean(axis=1).to_numpy(float)
        else:  # key lookup
            p["q_lg_sel_where"] = 0.0
            p["q_idx_where"] = 1.0
        p["x_mat_co"] = spec["q_join_co"] * p["m_merge_co"]
        p["x_mat_wc"] = spec["q_join_wc"] * p["m_merge_wc"]
        # does the model split a table this query reads?
        p["x_split_wh"] = spec["q_tbl_wh"] * p["m_split_wh"]
        p["x_split_cust"] = spec["q_tbl_cust"] * p["m_split_cust"]
        p["query"] = qi
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Leave-queries-out gate for query generalization.")
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--cells", type=int, default=500,
                    help="Config cells (profile,volume,servers) sampled; each has 135 models.")
    ap.add_argument("--eval-cells", type=int, default=120, help="Cells used for ranking eval.")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    specs = parse_query_specs(PROJECT_ROOT / "Input" / "queries.txt")
    assert len(specs) == NQ, f"expected {NQ} queries, parsed {len(specs)}"

    dataset = args.dataset or default_sweep_dataset()
    print(f"Loading {dataset.name} ...")
    cols = (["profile_id", "xml_file", "data_volume", "servers", "customer_count", "order_count"]
            + SEL + IDX + UNIT_T)
    df = read_sweep_dataset(dataset, cols)
    for c in df.columns:
        if c not in ("profile_id", "xml_file"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sample whole config cells so every sampled cell has all 135 models
    df["_cell"] = (df["profile_id"].astype(str) + "|" + df["data_volume"].astype(int).astype(str)
                   + "|" + df["servers"].astype(int).astype(str))
    rng = np.random.default_rng(args.seed)
    cells = rng.choice(df["_cell"].unique(), size=args.cells, replace=False)
    df = df[df["_cell"].isin(set(cells))].copy()
    print(f"  {args.cells} cells x {df['xml_file'].nunique()} models = {len(df):,} base rows")

    cat = df["xml_file"].astype("category")
    df["model_code"] = cat.cat.codes.astype("int32")
    df = df.merge(model_struct_features(sorted(cat.cat.categories)), on="xml_file", how="left")
    for c in SEL + ["customer_count", "order_count", "data_volume", "servers"]:
        df[f"lg_{c}"] = np.log10(df[c].astype(float))

    long = build_long(df, specs)
    print(f"  long frame: {len(long):,} rows x {len(FEATURES)} features")

    # Fold assignment: deal queries round-robin after sorting by join count, so
    # every fold holds out a mix of lookup / 1-join / 2-join queries.
    order = sorted(range(1, NQ + 1), key=lambda q: (specs[q - 1]["q_n_joins"], q))
    fold_of = {q: i % args.folds for i, q in enumerate(order)}

    eval_cells = set(rng.choice(cells, size=min(args.eval_cells, len(cells)), replace=False))
    per_query = []
    for fold in range(args.folds):
        test_qs = [q for q in range(1, NQ + 1) if fold_of[q] == fold]
        tr = long[~long["query"].isin(test_qs)]
        m = XGBRegressor(objective="reg:squarederror", n_estimators=args.trees, max_depth=8,
                         learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                         tree_method="hist", n_jobs=-1, random_state=args.seed)
        m.fit(tr[FEATURES], np.log1p(np.maximum(tr["y"].to_numpy(float), 0.0)))

        # Query-agnostic baseline: within each cell, a model's TRUE mean cost
        # over the TRAINING queries (the best you can do while ignoring the new query)
        base_rank_src = (tr[tr["_cell"].isin(eval_cells)]
                         .groupby(["_cell", "xml_file"], observed=True)["y"].mean())

        te = long[long["query"].isin(test_qs) & long["_cell"].isin(eval_cells)].copy()
        te["pred"] = np.expm1(m.predict(te[FEATURES]))
        te["rel"] = np.abs(te["pred"] - te["y"]) / np.maximum(te["y"].abs(), 1e-9)

        for q in test_qs:
            tq = te[te["query"] == q]
            rhos, regs, hits, regs_b, hits_b = [], [], [], [], []
            for cell, g in tq.groupby("_cell", observed=True):
                yt, yp = g["y"].to_numpy(float), g["pred"].to_numpy(float)
                if np.ptp(yt) <= 0:
                    continue
                rhos.append(spearmanr(yt, yp).statistic if np.ptp(yp) > 0 else 0.0)
                ybest = float(yt.min())
                pick = int(np.argmin(yp))
                regs.append((yt[pick] - ybest) / max(ybest, 1e-9))
                # tie-robust top-1: did we land in the cheapest tie-group?
                hits.append(yt[pick] <= ybest * (1 + 1e-9))
                yb = base_rank_src.loc[cell].reindex(g["xml_file"]).to_numpy(float)
                pick_b = int(np.nanargmin(yb))
                regs_b.append((yt[pick_b] - ybest) / max(ybest, 1e-9))
                hits_b.append(yt[pick_b] <= ybest * (1 + 1e-9))
            per_query.append({
                "query": f"q{q}", "n_joins": specs[q - 1]["q_n_joins"],
                "rel_err_med": float(tq["rel"].median()) if len(tq) else np.nan,
                "spearman": float(np.mean(rhos)),
                "regret_med": float(np.median(regs)),
                "regret_p90": float(np.quantile(regs, .9)),
                "top1_tie": float(np.mean(hits)),
                "base_regret_med": float(np.median(regs_b)),
                "base_top1_tie": float(np.mean(hits_b)),
            })
            r = per_query[-1]
            print(f"  fold {fold} q{q:<3} joins={r['n_joins']}  "
                  f"regret med {r['regret_med']:.2%} (base {r['base_regret_med']:.2%})  "
                  f"top1 {r['top1_tie']:.0%} (base {r['base_top1_tie']:.0%})  "
                  f"spearman {r['spearman']:.3f}", flush=True)

    res = pd.DataFrame(per_query).sort_values("query", key=lambda s: s.str[1:].astype(int))
    agg = {
        "unseen_regret_median": float(res["regret_med"].median()),
        "unseen_regret_worst_query": float(res["regret_med"].max()),
        "unseen_regret_p90_median": float(res["regret_p90"].median()),
        "unseen_top1_tie_mean": float(res["top1_tie"].mean()),
        "unseen_spearman_mean": float(res["spearman"].mean()),
        "baseline_regret_median": float(res["base_regret_med"].median()),
        "baseline_top1_tie_mean": float(res["base_top1_tie"].mean()),
    }
    # Regret-based gate (ties make rank-correlation unreliable here): the median
    # unseen query must have small regret AND clearly beat the query-agnostic baseline.
    gate = agg["unseen_regret_median"] <= 0.10 and \
        agg["unseen_regret_median"] <= 0.5 * max(agg["baseline_regret_median"], 1e-9)
    agg["verdict"] = "GO" if gate else "NO-GO"

    print("\n" + "=" * 78)
    print(res.to_string(index=False,
                        float_format=lambda v: f"{v:.3f}"))
    print("=" * 78)
    for k, v in agg.items():
        print(f"  {k}: {v if isinstance(v, str) else round(v, 4)}")
    out = PROJECT_ROOT / "Output" / "models" / "loqo_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"aggregate": agg, "per_query": res.to_dict("records")}, indent=2),
                   encoding="utf-8")
    print(f"\nsaved -> {out}")
    print(f"VERDICT: {agg['verdict']}")


if __name__ == "__main__":
    main()
