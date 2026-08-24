"""Phase 3 — train the model-recommender ranker.

Predicts the 60 per-query UNIT costs (20 queries x time/env/fin) plus the two
frequency-independent overheads (env storage, fin servers) from STRUCTURAL
features only: 18 schema dims + data_volume + servers + model_code.
No frequency features — the workload is applied analytically afterwards:

    time = U_t @ f          env = U_e @ f + env_oh          fin = U_f @ f + fin_oh
    (closed form verified against the real Cost JAR by scripts/verify_closed_form.py)

Trains BOTH candidate objectives and picks by RAW relative error (the EDA gate
showed log-space metrics flatter reality: sigma=0.3 log error -> 51% raw p90):
  - gamma : XGBoost reg:gamma on raw units (EDA mean-variance slope p=1.99)
  - log   : XGBoost reg:squarederror on log1p(units), expm1 at predict time

Evaluation = RANKING quality on held-out profiles (never seen in training),
across sampled workloads, per (volume, servers) config over the 135 models:
  - regret%   : true composite-relevant raw costs of the recommended model vs
                the true best (headline; target < 5-10% median)
  - top-1 / top-3 : recommended == true best / true best in predicted top 3
  - spearman  : predicted vs true composite ordering

CLI:
    python -m scripts.train_ranker --rows 1000000 --test-profiles 20 --trees 300
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor

from scripts.dataset_io import default_sweep_dataset, read_sweep_dataset
from scripts.generate_profile import NON_KEY_COLUMNS, PROJECT_ROOT
from scripts.verify_closed_form import parse_freqs

NQ = 20
STEMS = [f"q{i}" for i in range(1, NQ + 1)]
UNIT_T = [f"{s}_time_cost" for s in STEMS]
UNIT_E = [f"{s}_env_cost" for s in STEMS]
UNIT_F = [f"{s}_fin_cost" for s in STEMS]
UNITS = UNIT_T + UNIT_E + UNIT_F
OVERHEADS = ["env_overhead", "fin_overhead"]
TARGETS = UNITS + OVERHEADS
TOTALS = ["total_time_cost", "total_environmental_cost", "total_financial_cost"]

SEL = [f"sel_{c}" for c in NON_KEY_COLUMNS]
IDX = [f"idx_{c}" for c in NON_KEY_COLUMNS]
LOG_FEATS = SEL + ["customer_count", "order_count", "data_volume", "servers"]
FEATURES = [f"lg_{c}" for c in LOG_FEATS] + IDX + ["model_code"]

REGISTRY = PROJECT_ROOT / "Output" / "models"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_frame(dataset: Path, f0: np.ndarray) -> tuple[pd.DataFrame, dict[str, int]]:
    cols = ["profile_id", "xml_file", "data_volume", "servers"] + SEL + IDX + \
           ["customer_count", "order_count"] + TOTALS + UNITS
    df = read_sweep_dataset(dataset, cols)
    for c in df.columns:
        if c not in ("profile_id", "xml_file"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Frequency-independent overheads recovered at the default workload f0
    df["env_overhead"] = (df["total_environmental_cost"].to_numpy(float)
                          - df[UNIT_E].to_numpy(float) @ f0).clip(min=0.0)
    df["fin_overhead"] = (df["total_financial_cost"].to_numpy(float)
                          - df[UNIT_F].to_numpy(float) @ f0).clip(min=0.0)

    cat = df["xml_file"].astype("category")
    code_map = {name: i for i, name in enumerate(cat.cat.categories)}
    df["model_code"] = cat.cat.codes.astype("int32")
    for c in LOG_FEATS:
        df[f"lg_{c}"] = np.log10(df[c].astype(float))
    return df, code_map


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def fit_models(objective: str, df_tr: pd.DataFrame, trees: int, seed: int) -> tuple[dict, dict]:
    """Returns (models, scales). Gamma targets are median-normalized before the
    fit: reg:gamma's exp link overflows/degenerates on raw targets spanning up
    to ~1e12, but is scale-equivariant, so fitting y/median and multiplying the
    prediction back is exact in expectation."""
    X = df_tr[FEATURES]
    models: dict[str, XGBRegressor] = {}
    scales: dict[str, float] = {}
    # Gamma uses a log link, so n_estimators x learning_rate bounds how many
    # log-units the prediction can travel from base_score. Unit costs span ~14
    # log-units within one target, so gamma needs a much higher rate than the
    # log-target objective (which regresses log values directly).
    lr = 0.3 if objective == "gamma" else 0.1
    params = dict(n_estimators=trees, max_depth=8, learning_rate=lr,
                  subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", n_jobs=-1, random_state=seed)
    for i, t in enumerate(TARGETS, 1):
        y = df_tr[t].to_numpy(float)
        if objective == "gamma":
            pos = y[y > 0]
            med = float(np.median(pos)) if pos.size else 1.0
            scales[t] = med or 1.0
            m = XGBRegressor(objective="reg:gamma", **params)
            m.fit(X, np.maximum(y, 1e-9 * scales[t]) / scales[t])
        elif objective == "msle":
            scales[t] = 1.0
            m = XGBRegressor(objective="reg:squaredlogerror", **params)
            m.fit(X, np.maximum(y, 0.0))
        else:
            scales[t] = 1.0
            m = XGBRegressor(objective="reg:squarederror", **params)
            m.fit(X, np.log1p(np.maximum(y, 0.0)))
        models[t] = m
        if i % 10 == 0 or i == len(TARGETS):
            print(f"    [{objective}] {i}/{len(TARGETS)} targets fitted", flush=True)
    return models, scales


def predict_targets(models: dict, scales: dict, objective: str, X: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    for t, m in models.items():
        p = m.predict(X)
        if objective == "gamma":
            p = p * scales[t]
        elif objective == "log":
            p = np.expm1(p)
        # msle predicts on the raw scale directly
        out[t] = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    return out.clip(lower=0.0)


# ---------------------------------------------------------------------------
# Evaluation — raw error on units + ranking quality via the closed form
# ---------------------------------------------------------------------------
def _norm01_log(v: np.ndarray) -> np.ndarray:
    v = np.log1p(np.maximum(v, 0.0))
    span = v.max() - v.min()
    return (v - v.min()) / span if span > 0 else np.zeros_like(v)


def composite(tt: np.ndarray, ee: np.ndarray, ff: np.ndarray) -> np.ndarray:
    return (_norm01_log(tt) + _norm01_log(ee) + _norm01_log(ff)) / 3.0


def evaluate(df_te: pd.DataFrame, pred: pd.DataFrame, f0: np.ndarray,
             n_workloads: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    workloads = [f0] + [f0 * np.exp(rng.uniform(np.log(0.2), np.log(5.0), NQ))
                        for _ in range(n_workloads)]

    # Raw relative error on the 60 unit costs (log-space metrics flatter reality)
    true_u = df_te[UNITS].to_numpy(float)
    pred_u = pred[UNITS].to_numpy(float)
    rel = np.abs(pred_u - true_u) / np.maximum(np.abs(true_u), 1e-9)
    unit_err = {"median": float(np.median(rel)), "p90": float(np.quantile(rel, .9)),
                "p99": float(np.quantile(rel, .99))}

    regrets, top1, top3, rhos = [], [], [], []
    grp_keys = ["profile_id", "data_volume", "servers"]
    for _, g in df_te.groupby(grp_keys, observed=True):
        gi = g.index
        Ut, Ue, Uf = (g[UNIT_T].to_numpy(float), g[UNIT_E].to_numpy(float),
                      g[UNIT_F].to_numpy(float))
        eo, fo = g["env_overhead"].to_numpy(float), g["fin_overhead"].to_numpy(float)
        Pt, Pe, Pf = (pred.loc[gi, UNIT_T].to_numpy(float),
                      pred.loc[gi, UNIT_E].to_numpy(float),
                      pred.loc[gi, UNIT_F].to_numpy(float))
        peo, pfo = (pred.loc[gi, "env_overhead"].to_numpy(float),
                    pred.loc[gi, "fin_overhead"].to_numpy(float))
        for f in workloads:
            tt, ee, ff = Ut @ f, Ue @ f + eo, Uf @ f + fo            # truth
            pt, pe, pf = Pt @ f, Pe @ f + peo, Pf @ f + pfo          # predicted
            c_true, c_pred = composite(tt, ee, ff), composite(pt, pe, pf)
            best, pick = int(np.argmin(c_true)), int(np.argmin(c_pred))
            order = np.argsort(c_pred)
            top1.append(pick == best)
            top3.append(best in order[:3])
            # A degenerate (constant) prediction has no defined rank correlation;
            # count it as 0 rather than poisoning the mean with NaN.
            rho = spearmanr(c_true, c_pred).statistic if np.ptp(c_pred) > 0 else np.nan
            rhos.append(0.0 if np.isnan(rho) else float(rho))
            # Regret in RAW cost space: mean % extra across the three metrics
            reg = np.mean([(tt[pick] - tt[best]) / max(tt[best], 1e-9),
                           (ee[pick] - ee[best]) / max(ee[best], 1e-9),
                           (ff[pick] - ff[best]) / max(ff[best], 1e-9)])
            regrets.append(reg)

    regrets = np.array(regrets)
    return {
        "unit_raw_rel_err": unit_err,
        "regret_median": float(np.median(regrets)),
        "regret_p90": float(np.quantile(regrets, .9)),
        "regret_mean": float(regrets.mean()),
        "top1_acc": float(np.mean(top1)),
        "top3_recall": float(np.mean(top3)),
        "spearman_mean": float(np.mean(rhos)),
        "n_rank_cases": int(len(regrets)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Train + evaluate the model-recommender ranker.")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--rows", type=int, default=1_000_000, help="Training rows (subsampled).")
    p.add_argument("--test-profiles", type=int, default=20, help="Whole profiles held out.")
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--workloads", type=int, default=15, help="Random workloads per config in eval.")
    p.add_argument("--objectives", default="gamma,log")
    p.add_argument("--name", default="model_ranker")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dataset = args.dataset or default_sweep_dataset()
    f0 = np.array(parse_freqs(PROJECT_ROOT / "Input" / "queries.txt"), float)
    print(f"Loading {dataset} ...")
    df, code_map = load_frame(dataset, f0)
    print(f"  {len(df):,} rows, {df['profile_id'].nunique()} profiles")

    rng = np.random.default_rng(args.seed)
    profiles = np.array(sorted(df["profile_id"].unique()))
    test_prof = set(rng.choice(profiles, size=args.test_profiles, replace=False))
    df_te = df[df["profile_id"].isin(test_prof)].copy()
    df_tr = df[~df["profile_id"].isin(test_prof)]
    if len(df_tr) > args.rows:
        df_tr = df_tr.sample(args.rows, random_state=args.seed)
    print(f"  train rows {len(df_tr):,} ({len(profiles) - args.test_profiles} profiles), "
          f"held-out rows {len(df_te):,} ({args.test_profiles} profiles)")

    results = {}
    for objective in [o.strip() for o in args.objectives.split(",") if o.strip()]:
        print(f"\n=== objective: {objective} ===")
        t0 = datetime.now()
        models, scales = fit_models(objective, df_tr, args.trees, args.seed)
        print(f"  fitted {len(TARGETS)} targets in {(datetime.now()-t0).total_seconds():.0f}s")
        pred = predict_targets(models, scales, objective, df_te[FEATURES])
        metrics = evaluate(df_te, pred, f0, args.workloads, args.seed)
        results[objective] = {"models": models, "scales": scales, "metrics": metrics}
        print("  unit raw rel err:", {k: round(v, 3) for k, v in metrics["unit_raw_rel_err"].items()})
        print(f"  regret: median {metrics['regret_median']:.2%}  p90 {metrics['regret_p90']:.2%}  "
              f"mean {metrics['regret_mean']:.2%}")
        print(f"  top-1 {metrics['top1_acc']:.1%}  top-3 {metrics['top3_recall']:.1%}  "
              f"spearman {metrics['spearman_mean']:.3f}  ({metrics['n_rank_cases']:,} cases)")

    # Winner by raw relative error (tie-break: median regret)
    winner = min(results, key=lambda o: (results[o]["metrics"]["unit_raw_rel_err"]["median"],
                                         results[o]["metrics"]["regret_median"]))
    print(f"\nWINNER by raw relative error: {winner}")

    REGISTRY.mkdir(parents=True, exist_ok=True)
    for objective, res in results.items():
        bundle = {
            "kind": "unit_cost_ranker",
            "objective": objective,
            "models": res["models"],
            "target_scales": res["scales"],
            "targets": TARGETS,
            "features": FEATURES,
            "log_feats": LOG_FEATS,
            "idx_feats": IDX,
            "model_code_map": code_map,
            "default_freqs": {s: int(v) for s, v in zip(STEMS, f0)},
            "closed_form": "time=Ut@f; env=Ue@f+env_oh; fin=Uf@f+fin_oh",
            "meta": {
                "name": f"{args.name}_{objective}",
                "created": datetime.now().isoformat(timespec="seconds"),
                "dataset": str(dataset.name),
                "train_rows": int(len(df_tr)),
                "test_profiles": sorted(str(x) for x in test_prof),
                "trees": args.trees,
                "metrics": res["metrics"],
                "is_winner": objective == winner,
            },
        }
        out = REGISTRY / f"{args.name}_{objective}.pkl"
        with open(out, "wb") as fh:
            pickle.dump(bundle, fh)
        print(f"saved {out} ({out.stat().st_size/1e6:.1f} MB)")

    (REGISTRY / f"{args.name}_metrics.json").write_text(
        json.dumps({o: r["metrics"] for o, r in results.items()} | {"winner": winner},
                   indent=2), encoding="utf-8")
    print(f"metrics -> {REGISTRY / (args.name + '_metrics.json')}")


if __name__ == "__main__":
    main()
