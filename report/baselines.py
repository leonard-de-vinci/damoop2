"""Naive/oracle baselines vs DaMoOp* on the SAME 13,440 held-out cases.

Same composite objective and regret definition as the DaMoOp* row (regret = mean of
the three raw-dimension relative regrets of the picked model; top-1 = tie-tolerant
composite-best; zero-regret share = regret == 0). All baselines use TRUE costs (no ML):

  (a) random      : uniform random model per case (reported as the exact expectation
                    over all case x model pairs — no sampling noise).
  (b) oracle-static: the single fixed model that minimizes mean regret over all cases
                    (best hindsight commitment).
  (c) default-cheapest: the model cheapest at the default configuration (baseline profile,
                    default volume/servers, default workload), applied unchanged everywhere.
  (d) per-profile-default: per profile, the model cheapest at that profile's default
                    setting, applied across that profile's 42 settings.

Run: .venv/Scripts/python -m report.baselines
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.train_ranker import (
    SEL, IDX, UNIT_T, UNIT_E, UNIT_F, UNITS, TOTALS, PROJECT_ROOT, composite,
)
from scripts.dataset_io import default_sweep_dataset, read_sweep_dataset
from lib.model_io import load_bundle

BUNDLE = PROJECT_ROOT / "Output" / "models" / "model_ranker_log.pkl"
NQ = 20
DEFAULT_VOL, DEFAULT_SRV = 1_000_000.0, 100.0
# (c) reference model: cheapest at the baseline/default config, validated #1/135 in
# PAPER_QA Q19 (Predict-page defaults, real JAR run).
DEFAULT_CONFIG_MODEL = "Merge_c_o_ID_c_ID/Merge_w_c_ID_w_ID"
EPS = 1e-9


def stats(regret: np.ndarray, tied_best: np.ndarray) -> dict:
    return {
        "regret_median": float(np.median(regret)),
        "regret_mean": float(regret.mean()),
        "regret_p90": float(np.quantile(regret, 0.9)),
        "top1_acc": float(tied_best.mean()),
        "zero_regret_share": float((regret <= EPS).mean()),
        "n_cases": int(len(regret)),
    }


def main() -> None:
    bundle = load_bundle(BUNDLE)
    test_profiles = set(bundle["meta"]["test_profiles"])
    f0 = np.array([bundle["default_freqs"][f"q{i}"] for i in range(1, NQ + 1)], float)

    cols = (["profile_id", "xml_file", "data_volume", "servers"] + SEL + IDX +
            ["customer_count", "order_count"] + TOTALS + UNITS)
    df = read_sweep_dataset(default_sweep_dataset(), cols)
    df = df[df["profile_id"].isin(test_profiles)].copy()
    for c in df.columns:
        if c not in ("profile_id", "xml_file"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["env_oh"] = (df["total_environmental_cost"].to_numpy(float)
                    - df[UNIT_E].to_numpy(float) @ f0).clip(min=0.0)
    df["fin_oh"] = (df["total_financial_cost"].to_numpy(float)
                    - df[UNIT_F].to_numpy(float) @ f0).clip(min=0.0)

    models = sorted(df["xml_file"].unique())
    midx = {m: i for i, m in enumerate(models)}
    M = len(models)
    assert M == 135, M
    assert DEFAULT_CONFIG_MODEL in midx, "default-config model not found"
    c_model = midx[DEFAULT_CONFIG_MODEL]

    rng = np.random.default_rng(42)
    workloads = [f0] + [f0 * np.exp(rng.uniform(np.log(0.2), np.log(5.0), NQ))
                        for _ in range(15)]

    # Per-case regret matrix R[case, model] and tie-tolerant best indicator.
    R_rows, TB_rows = [], []
    case_profile = []
    default_winner: dict[str, int] = {}   # (d): per-profile default-cell winner

    for (prof, vol, srv), g in df.groupby(["profile_id", "data_volume", "servers"],
                                          observed=True):
        g = g.sort_values("xml_file")
        assert list(g["xml_file"]) == models, "model ordering mismatch"
        Ut, Ue, Uf = g[UNIT_T].to_numpy(float), g[UNIT_E].to_numpy(float), g[UNIT_F].to_numpy(float)
        eo, fo = g["env_oh"].to_numpy(float), g["fin_oh"].to_numpy(float)
        for wk, f in enumerate(workloads):
            tt, ee, ff = Ut @ f, Ue @ f + eo, Uf @ f + fo
            comp = composite(tt, ee, ff)
            best = int(np.argmin(comp))
            # regret of EVERY model (mean of 3 raw-dim relative regrets vs best)
            rt = (tt - tt[best]) / max(tt[best], 1e-9)
            re = (ee - ee[best]) / max(ee[best], 1e-9)
            rf = (ff - ff[best]) / max(ff[best], 1e-9)
            R_rows.append((rt + re + rf) / 3.0)
            TB_rows.append(comp <= comp[best] + EPS)     # tie-tolerant best
            case_profile.append(prof)
            if vol == DEFAULT_VOL and srv == DEFAULT_SRV and wk == 0:
                default_winner[prof] = best

    R = np.array(R_rows, dtype=float)              # (13440, 135)
    TB = np.array(TB_rows, dtype=bool)
    case_profile = np.array(case_profile)
    n_cases = R.shape[0]
    print(f"cases {n_cases}, models {M}")

    results = {}

    # (a) uniform random — exact expectation over all case x model pairs
    results["a_random"] = stats(R.reshape(-1), TB.reshape(-1))

    # (b) oracle static — fixed model minimizing mean regret
    per_model_mean = R.mean(axis=0)
    b_star = int(np.argmin(per_model_mean))
    results["b_oracle_static"] = stats(R[:, b_star], TB[:, b_star])
    results["b_oracle_static"]["model"] = models[b_star]
    # most-frequent true-best model (max top-1) — the analogue of the recorded 28.8%
    win_star = int(np.argmax(TB.sum(axis=0)))
    most_freq_win_share = float(TB[:, win_star].mean())
    results["b2_most_frequent_winner"] = stats(R[:, win_star], TB[:, win_star])
    results["b2_most_frequent_winner"]["model"] = models[win_star]

    # (c) cheapest at default config, applied everywhere
    results["c_default_cheapest"] = stats(R[:, c_model], TB[:, c_model])
    results["c_default_cheapest"]["model"] = DEFAULT_CONFIG_MODEL

    # (d) per-profile default winner, applied across that profile's settings
    pick_d = np.array([default_winner[p] for p in case_profile])
    reg_d = R[np.arange(n_cases), pick_d]
    tb_d = TB[np.arange(n_cases), pick_d]
    results["d_per_profile_default"] = stats(reg_d, tb_d)

    # DaMoOp* headline for side-by-side (from recorded metrics)
    results["damoop_star"] = {
        "regret_median": 0.0, "regret_mean": 0.13617800921348813,
        "regret_p90": 0.3409639937844188, "top1_acc": 0.6112351190476191,
        "zero_regret_share": 0.6113839285714286, "n_cases": 13440,
    }

    order = ["a_random", "b_oracle_static", "b2_most_frequent_winner",
             "c_default_cheapest", "d_per_profile_default", "damoop_star"]
    tbl = pd.DataFrame([{"method": k, **{kk: results[k][kk] for kk in
              ["regret_median", "regret_mean", "regret_p90", "top1_acc",
               "zero_regret_share"]}} for k in order])
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(tbl.to_string(index=False))
    print(f"\n(b) oracle-static model = {models[b_star]}")
    print(f"most-frequent true-best model wins {most_freq_win_share:.4f} "
          f"of the 13,440 held-out cases  (recorded default-workload top-winner share = 0.288)")
    print(f"(d) uses {len(default_winner)} per-profile default winners; "
          f"{len(set(default_winner.values()))} distinct")

    (PROJECT_ROOT / "report" / "baselines.json").write_text(
        json.dumps({"results": results,
                    "most_freq_win_share": most_freq_win_share,
                    "oracle_static_model": models[b_star]}, indent=2))
    tbl.to_csv(PROJECT_ROOT / "report" / "baselines.csv", index=False)
    print("\n-> report/baselines.csv , report/baselines.json")


if __name__ == "__main__":
    main()
