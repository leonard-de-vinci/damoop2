"""Export figure-ready CSV/JSON for the IEEE Big Data paper.

Loads the shipped ranker bundle ONCE and emits:
  C2  report/feature_importance.csv   — mean gain importance across the 62 target models
  C3  report/pred_vs_actual.csv       — sampled (true, pred) unit costs per dimension (log-plottable)
  C4  report/winner_grid.csv          — composite cost of a few models across the 6x7 grid, one profile
      + report/winner_stats.json      — re-confirmed distinct-winner statistics

Run: .venv/Scripts/python -m report.export_figure_data
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_ranker import (
    FEATURES, LOG_FEATS, IDX, UNIT_T, UNIT_E, UNIT_F, UNITS, SEL,
    TOTALS, PROJECT_ROOT,
)
from scripts.dataset_io import default_sweep_dataset, read_sweep_dataset
from lib.model_io import load_bundle

REPORT = PROJECT_ROOT / "report"
BUNDLE = PROJECT_ROOT / "Output" / "models" / "model_ranker_log.pkl"
RNG = np.random.default_rng(42)


def n01_log(v: np.ndarray) -> np.ndarray:
    v = np.log1p(np.maximum(v, 0.0))
    s = v.max() - v.min()
    return (v - v.min()) / s if s > 0 else np.zeros_like(v)


def main() -> None:
    print(f"loading {BUNDLE.name} ...", flush=True)
    bundle = load_bundle(BUNDLE)
    models = bundle["models"]
    code_map = bundle["model_code_map"]
    test_profiles = set(bundle["meta"]["test_profiles"])
    print(f"  {len(models)} target models; {len(test_profiles)} held-out profiles")

    # ---- C2: feature importance (gain), averaged over the 62 target models ----
    imp = np.zeros(len(FEATURES))
    for t, m in models.items():
        b = m.get_booster()
        score = b.get_score(importance_type="gain")  # keyed by feature NAME
        tot = sum(score.values()) or 1.0
        for j, feat in enumerate(FEATURES):
            imp[j] += score.get(feat, 0.0) / tot   # per-model normalized, then averaged
    imp /= len(models)
    imp_df = (pd.DataFrame({"feature": FEATURES, "mean_gain_norm": imp})
              .sort_values("mean_gain_norm", ascending=False).reset_index(drop=True))
    imp_df.to_csv(REPORT / "feature_importance.csv", index=False)
    print("  C2 -> feature_importance.csv (top 5):")
    print(imp_df.head(5).to_string(index=False))

    # ---- Load held-out rows for C3 ----
    f0 = np.array([bundle["default_freqs"][s] for s in
                   [f"q{i}" for i in range(1, 21)]], float)
    cols = (["profile_id", "xml_file", "data_volume", "servers"] + SEL + IDX +
            ["customer_count", "order_count"] + TOTALS + UNITS)
    df = read_sweep_dataset(default_sweep_dataset(), cols)
    df = df[df["profile_id"].isin(test_profiles)].copy()
    for c in df.columns:
        if c not in ("profile_id", "xml_file"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["model_code"] = df["xml_file"].map(code_map).astype("float").fillna(-1).astype("int32")
    for c in LOG_FEATS:
        df[f"lg_{c}"] = np.log10(df[c].astype(float))
    print(f"  held-out rows: {len(df):,}")

    X = df[FEATURES]
    pred = {}
    for t, m in models.items():
        pred[t] = np.expm1(m.predict(X))  # log objective
    pred = pd.DataFrame(pred, index=df.index).clip(lower=0.0)

    # ---- C3: sampled predicted-vs-actual per dimension (~5000 pts each) ----
    rows = []
    for dim, ucols in [("time", UNIT_T), ("env", UNIT_E), ("fin", UNIT_F)]:
        true = df[ucols].to_numpy(float).ravel()
        prd = pred[ucols].to_numpy(float).ravel()
        keep = (true > 0) & (prd > 0)          # log-plottable only
        idx = np.where(keep)[0]
        take = RNG.choice(idx, size=min(5000, idx.size), replace=False)
        rows.append(pd.DataFrame({"dim": dim, "true": true[take], "pred": prd[take]}))
    pva = pd.concat(rows, ignore_index=True)
    pva.to_csv(REPORT / "pred_vs_actual.csv", index=False)
    print(f"  C3 -> pred_vs_actual.csv ({len(pva):,} points; "
          f"{(pva['dim']=='fin').sum()} fin / {(pva['dim']=='time').sum()} time / "
          f"{(pva['dim']=='env').sum()} env)")

    # ---- C4: winner grid for one held-out profile + winner stats ----
    prof = sorted(test_profiles)[0]
    g = df[df["profile_id"] == prof]
    comp_rows = []
    for (vol, srv), cell in g.groupby(["data_volume", "servers"]):
        comp = (n01_log(cell["total_time_cost"].to_numpy(float))
                + n01_log(cell["total_environmental_cost"].to_numpy(float))
                + n01_log(cell["total_financial_cost"].to_numpy(float))) / 3.0
        for xf, c in zip(cell["xml_file"], comp):
            comp_rows.append({"profile": prof, "data_volume": vol, "servers": srv,
                              "xml_file": xf, "composite": c})
    grid = pd.DataFrame(comp_rows)
    grid.to_csv(REPORT / "winner_grid.csv", index=False)
    winners = grid.loc[grid.groupby(["data_volume", "servers"])["composite"].idxmin()]
    print(f"  C4 -> winner_grid.csv (profile {prof}); "
          f"{winners['xml_file'].nunique()} distinct winners across its 42 cells")

    # Re-confirm the cross-profile winner stats on the full parquet (default workload)
    full = read_sweep_dataset(default_sweep_dataset(),
                              ["profile_id", "xml_file", "data_volume", "servers"] + TOTALS)
    for c in TOTALS:
        full[c] = pd.to_numeric(full[c], errors="coerce")
    win_ids = []
    for _, cell in full.groupby(["profile_id", "data_volume", "servers"]):
        comp = (n01_log(cell["total_time_cost"].to_numpy(float))
                + n01_log(cell["total_environmental_cost"].to_numpy(float))
                + n01_log(cell["total_financial_cost"].to_numpy(float))) / 3.0
        win_ids.append(cell["xml_file"].to_numpy()[int(np.argmin(comp))])
    win_ids = pd.Series(win_ids)
    stats = {
        "n_configs": int(len(win_ids)),
        "distinct_winners": int(win_ids.nunique()),
        "top_winner_share": float(win_ids.value_counts(normalize=True).iloc[0]),
    }
    (REPORT / "winner_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  winner_stats.json: {stats}")


if __name__ == "__main__":
    main()
