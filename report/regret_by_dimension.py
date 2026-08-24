"""D3 — per-objective regret breakdown on the SAME 13,440 held-out cases.

For each ranking objective (balanced composite, time-only, env-only, fin-only),
report median/mean/p90 regret and top-1 accuracy. Reuses the shipped ranker
bundle and the exact eval workloads (seed 42, 16 workloads) so numbers line up
with model_ranker_metrics.json.

For the single-dimension objectives BOTH selection and penalty are that dimension's
raw cost (regret >= 0); only the composite row uses composite-based selection with
the mean-of-three raw penalty (matching train_ranker's headline).

Run: .venv/Scripts/python -m report.regret_by_dimension
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.train_ranker import (
    FEATURES, LOG_FEATS, IDX, SEL, UNIT_T, UNIT_E, UNIT_F, UNITS, TOTALS,
    PROJECT_ROOT, composite,
)
from scripts.dataset_io import default_sweep_dataset, read_sweep_dataset
from lib.model_io import load_bundle

BUNDLE = PROJECT_ROOT / "Output" / "models" / "model_ranker_log.pkl"
NQ = 20


def main() -> None:
    bundle = load_bundle(BUNDLE)
    models = bundle["models"]
    code_map = bundle["model_code_map"]
    test_profiles = set(bundle["meta"]["test_profiles"])
    f0 = np.array([bundle["default_freqs"][f"q{i}"] for i in range(1, NQ + 1)], float)

    cols = (["profile_id", "xml_file", "data_volume", "servers"] + SEL + IDX +
            ["customer_count", "order_count"] + TOTALS + UNITS)
    df = read_sweep_dataset(default_sweep_dataset(), cols)
    df = df[df["profile_id"].isin(test_profiles)].copy()
    for c in df.columns:
        if c not in ("profile_id", "xml_file"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["env_overhead"] = (df["total_environmental_cost"].to_numpy(float)
                          - df[UNIT_E].to_numpy(float) @ f0).clip(min=0.0)
    df["fin_overhead"] = (df["total_financial_cost"].to_numpy(float)
                          - df[UNIT_F].to_numpy(float) @ f0).clip(min=0.0)
    df["model_code"] = df["xml_file"].map(code_map).astype("float").fillna(-1).astype("int32")
    for c in LOG_FEATS:
        df[f"lg_{c}"] = np.log10(df[c].astype(float))

    X = df[FEATURES]
    pred = pd.DataFrame({t: np.expm1(m.predict(X)) for t, m in models.items()},
                        index=df.index).clip(lower=0.0)

    rng = np.random.default_rng(42)
    workloads = [f0] + [f0 * np.exp(rng.uniform(np.log(0.2), np.log(5.0), NQ))
                        for _ in range(15)]

    objectives = ["composite", "time", "env", "fin"]
    acc = {o: {"regret": [], "top1": []} for o in objectives}

    for _, g in df.groupby(["profile_id", "data_volume", "servers"], observed=True):
        gi = g.index
        Ut, Ue, Uf = g[UNIT_T].to_numpy(float), g[UNIT_E].to_numpy(float), g[UNIT_F].to_numpy(float)
        eo, fo = g["env_overhead"].to_numpy(float), g["fin_overhead"].to_numpy(float)
        Pt, Pe, Pf = (pred.loc[gi, UNIT_T].to_numpy(float), pred.loc[gi, UNIT_E].to_numpy(float),
                      pred.loc[gi, UNIT_F].to_numpy(float))
        peo, pfo = pred.loc[gi, "env_overhead"].to_numpy(float), pred.loc[gi, "fin_overhead"].to_numpy(float)
        for f in workloads:
            tt, ee, ff = Ut @ f, Ue @ f + eo, Uf @ f + fo
            pt, pe, pf = Pt @ f, Pe @ f + peo, Pf @ f + pfo
            # For single-dimension objectives, regret is measured in that
            # dimension's own RAW cost. For the balanced composite, the pick is
            # by composite score but regret is the mean of the three raw-dim
            # relative regrets (identical to train_ranker's headline metric).
            scores = {"composite": (composite(tt, ee, ff), composite(pt, pe, pf)),
                      "time": (tt, pt), "env": (ee, pe), "fin": (ff, pf)}
            raw = {"time": tt, "env": ee, "fin": ff}
            for o, (ct, cp) in scores.items():
                best, pick = int(np.argmin(ct)), int(np.argmin(cp))
                acc[o]["top1"].append(pick == best)
                if o == "composite":
                    reg = np.mean([(raw[d][pick] - raw[d][best]) / max(raw[d][best], 1e-9)
                                   for d in ("time", "env", "fin")])
                else:
                    reg = (ct[pick] - ct[best]) / max(ct[best], 1e-9)
                acc[o]["regret"].append(float(reg))

    rows = []
    for o in objectives:
        r = np.array(acc[o]["regret"])
        rows.append({
            "objective": o,
            "regret_median": float(np.median(r)),
            "regret_mean": float(r.mean()),
            "regret_p90": float(np.quantile(r, 0.9)),
            "top1_acc": float(np.mean(acc[o]["top1"])),
            "zero_regret_share": float((r == 0).mean()),
            "n_cases": int(len(r)),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    (PROJECT_ROOT / "report" / "regret_by_dimension.json").write_text(
        json.dumps(rows, indent=2))
    # The composite per-case regrets ARE the 13,440 DaMoOp* values used by the
    # regret-CDF figure — dump them so make_figures.py has its input.
    pd.DataFrame({"regret": acc["composite"]["regret"]}).to_csv(
        PROJECT_ROOT / "report" / "regret_cases.csv", index=False)
    print("-> report/regret_by_dimension.json , report/regret_cases.csv")


if __name__ == "__main__":
    main()
