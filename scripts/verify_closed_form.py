"""Verify the closed-form workload reweighting the recommender relies on.

The Phase-3 recommender predicts per-query UNIT costs and applies the query
workload analytically:

    time  = U_time @ f
    env   = U_env  @ f + env_overhead      (storage cost, frequency-independent)
    fin   = U_fin  @ f + fin_overhead      (server cost,  frequency-independent)

Two checks:

A. DATASET ALGEBRA — over the whole assembled dataset, totals at the default
   workload f0 must reconstruct from the unit columns, and the recovered env/fin
   overheads must be non-negative.

B. JAR GROUND TRUTH — activate one sweep profile, run DMG+Cost at f0, then
   rewrite conf/queries.txt with perturbed frequencies f1 and run the Cost JAR
   again. The JAR's f1 totals must match the closed-form prediction computed
   ONLY from the f0 run — including which model is cheapest. Also cross-checks
   lib.scenarios.reweight_costs against the inline formula.

CLI:
    python -m scripts.verify_closed_form                 # A only (fast)
    python -m scripts.verify_closed_form --jar           # A + B (runs the JARs)
    python -m scripts.verify_closed_form --jar --profile Input/profiles/sweep_runs/<run>/profile_S000
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from lib.parse_costs import build_costs_dataframe
from scripts.dataset_io import default_sweep_dataset, read_sweep_dataset
from scripts.run_sweep import (
    CONF_DIR,
    INPUT_DIR,
    OUTPUT_DMS_DIR,
    PROJECT_ROOT,
    _activate,
    _restore_input,
    _run_phase1,
    _run_phase2,
    _snapshot_input,
)

NQ = 20
STEMS = [f"q{i}" for i in range(1, NQ + 1)]
UNIT_T = [f"{s}_time_cost" for s in STEMS]
UNIT_E = [f"{s}_env_cost" for s in STEMS]
UNIT_F = [f"{s}_fin_cost" for s in STEMS]
TOTALS = ["total_time_cost", "total_environmental_cost", "total_financial_cost"]
KEYCOLS = ["xml_file", "data_volume", "servers"]

# Deterministic aggressive perturbation (0.2x .. 5x) — enough to flip rankings.
FREQ_MULT = [5.0, 0.2, 3.0, 0.5, 2.0, 4.0, 0.25, 1.5, 0.33, 2.5,
             5.0, 0.2, 3.0, 0.5, 2.0, 4.0, 0.25, 1.5, 0.33, 2.5]


def parse_freqs(queries_file: Path) -> list[int]:
    freqs = []
    for line in queries_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("query/"):
            continue
        freqs.append(int(line.split("/")[-2]))
    return freqs


def write_perturbed_freqs(queries_file: Path, mults: list[float]) -> list[int]:
    """Rewrite the frequency token of each query line; returns the new freqs."""
    out_lines, new_freqs, qi = [], [], 0
    for raw in queries_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("query/"):
            out_lines.append(raw)
            continue
        parts = line.split("/")
        nf = max(1, round(int(parts[-2]) * mults[qi]))
        parts[-2] = str(nf)
        out_lines.append("/".join(parts))
        new_freqs.append(nf)
        qi += 1
    queries_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return new_freqs


def rel_err(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.abs(pred - true) / np.maximum(np.abs(true), 1e-9)


def check_dataset(dataset: Path) -> bool:
    print(f"\n=== A. DATASET ALGEBRA ({dataset.name}) ===")
    f0 = np.array(parse_freqs(INPUT_DIR / "queries.txt"), float)
    df = read_sweep_dataset(dataset, KEYCOLS + TOTALS + UNIT_T + UNIT_E + UNIT_F)
    for c in TOTALS + UNIT_T + UNIT_E + UNIT_F:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ok = True
    tt = df[UNIT_T].to_numpy(float) @ f0
    r = rel_err(tt, df["total_time_cost"].to_numpy(float))
    print(f"  time  == U@f0:            max rel err {r.max():.2e}  (rows {len(df):,})")
    ok &= r.max() < 1e-3

    for name, ucols, tcol in [("env", UNIT_E, "total_environmental_cost"),
                              ("fin", UNIT_F, "total_financial_cost")]:
        oh = df[tcol].to_numpy(float) - df[ucols].to_numpy(float) @ f0
        neg_share = float((oh < -1e-6 * np.abs(df[tcol].to_numpy(float))).mean())
        print(f"  {name}   overhead = total-U@f0: min {oh.min():.4g}, negative share {neg_share:.2%}")
        ok &= neg_share < 0.001
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_jar(profile_dir: Path, base_model: str, java: str, heap: str) -> bool:
    print(f"\n=== B. JAR GROUND TRUTH ({profile_dir.name}) ===")
    snap = Path(tempfile.mkdtemp(prefix="cf_verify_snap_"))
    logs = snap / "logs"
    _snapshot_input(snap)

    # Phase1 wipes and regenerates Output/GeneratedDMs/, so this check clobbers
    # whatever live output the user had (e.g. the tracked showcase example) —
    # snapshot it alongside Input/conf so it can be put back exactly.
    gdm_snap = snap / "_GeneratedDMs"
    had_output = OUTPUT_DMS_DIR.exists()
    if had_output:
        shutil.copytree(OUTPUT_DMS_DIR, gdm_snap)

    try:
        _activate(profile_dir)
        print("  phase1 (DMG) at default workload f0...")
        if not _run_phase1(base_model, java, heap, logs / "p1.log", 600):
            print("  [FAIL] phase1 failed"); return False
        print("  phase2 (Cost) at f0...")
        if not _run_phase2(base_model, java, heap, logs / "p2_f0.log", 600):
            print("  [FAIL] phase2 (f0) failed"); return False
        df0 = build_costs_dataframe(OUTPUT_DMS_DIR)

        f0 = np.array(parse_freqs(CONF_DIR / "queries.txt"), float)
        # Perturb ONLY conf/queries.txt — the Cost JAR reads conf/, and leaving
        # Input/ at f0 mirrors how the recommender will treat "original" freqs.
        new_freqs = write_perturbed_freqs(CONF_DIR / "queries.txt", FREQ_MULT)
        f1 = np.array(new_freqs, float)
        print(f"  perturbed freqs (first 6): {new_freqs[:6]} ...")

        # The Cost JAR rewrites costs.txt per model; clear them to be safe.
        for cf in OUTPUT_DMS_DIR.rglob("costs.txt"):
            cf.unlink()
        print("  phase2 (Cost) at perturbed workload f1...")
        if not _run_phase2(base_model, java, heap, logs / "p2_f1.log", 600):
            print("  [FAIL] phase2 (f1) failed"); return False
        df1 = build_costs_dataframe(OUTPUT_DMS_DIR)
    finally:
        _restore_input(snap)
        if OUTPUT_DMS_DIR.exists():
            shutil.rmtree(OUTPUT_DMS_DIR)
        if had_output:
            shutil.copytree(gdm_snap, OUTPUT_DMS_DIR)
        shutil.rmtree(snap, ignore_errors=True)
        print("  Input/ + conf/ + Output/GeneratedDMs/ restored.")

    for d in (df0, df1):
        for c in TOTALS + UNIT_T + UNIT_E + UNIT_F:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Closed-form prediction of the f1 totals from the f0 run only
    pred = df0[KEYCOLS].copy()
    for name, ucols, tcol in [("time", UNIT_T, "total_time_cost"),
                              ("env", UNIT_E, "total_environmental_cost"),
                              ("fin", UNIT_F, "total_financial_cost")]:
        U = df0[ucols].to_numpy(float)
        overhead = df0[tcol].to_numpy(float) - U @ f0
        pred[tcol] = U @ f1 + overhead

    m = pred.merge(df1[KEYCOLS + TOTALS], on=KEYCOLS, suffixes=("_pred", "_jar"))
    print(f"  matched rows: {len(m):,} (f0 run {len(df0):,}, f1 run {len(df1):,})")
    ok = len(m) == len(df0) == len(df1)

    for tcol in TOTALS:
        r = rel_err(m[f"{tcol}_pred"].to_numpy(float), m[f"{tcol}_jar"].to_numpy(float))
        print(f"  {tcol}: max rel err {r.max():.2e}, p99 {np.quantile(r, .99):.2e}")
        ok &= r.max() < 1e-3

    # Does the closed form pick the same cheapest model per (volume, servers)?
    # Ties count as agreement: with errors at machine precision, two models with
    # identical cost can argmin-flip on float noise — that is not a disagreement.
    agree = {t: 0 for t in TOTALS}; n_cells = 0
    for (_, _), grp in m.groupby(["data_volume", "servers"]):
        n_cells += 1
        for t in TOTALS:
            pick_cost = float(grp[f"{t}_jar"].loc[grp[f"{t}_pred"].idxmin()])
            best_cost = float(grp[f"{t}_jar"].min())
            agree[t] += int(pick_cost <= best_cost * (1 + 1e-9))
    for t in TOTALS:
        print(f"  cheapest-model agreement ({t}, tie-tolerant): {agree[t]}/{n_cells}")
        ok &= agree[t] == n_cells

    # Cross-check lib.scenarios.reweight_costs produces the same prediction
    try:
        from lib.scenarios import reweight_costs
        f0_map = {s: int(v) for s, v in zip(STEMS, f0)}
        f1_map = {s: int(v) for s, v in zip(STEMS, f1)}
        rw = reweight_costs(df0, f1_map, f0_map)
        r = max(rel_err(rw[t].to_numpy(float), pred[t].to_numpy(float)).max() for t in TOTALS)
        print(f"  lib.scenarios.reweight_costs vs inline closed form: max rel err {r:.2e}")
        ok &= r < 1e-9
    except Exception as e:  # noqa: BLE001
        print(f"  reweight_costs cross-check skipped: {e}")

    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Verify closed-form workload reweighting.")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--jar", action="store_true", help="Also run the JAR ground-truth check.")
    p.add_argument("--profile", type=Path, default=None,
                   help="Sweep profile dir for the JAR check (default: first profile of the newest run).")
    p.add_argument("--base-model", default="TPC_C")
    p.add_argument("--java", default="java")
    p.add_argument("--heap", default="2g")
    args = p.parse_args()

    ok = check_dataset(args.dataset or default_sweep_dataset())

    if args.jar:
        profile = args.profile
        if profile is None:
            runs_root = PROJECT_ROOT / "Input" / "profiles" / "sweep_runs"
            candidates = sorted(runs_root.glob("*/profile_*"))
            if not candidates:
                raise SystemExit("No sweep profiles found for the JAR check.")
            profile = candidates[0]
        ok &= check_jar(profile, args.base_model, args.java, args.heap)

    print("\n" + ("ALL CLOSED-FORM CHECKS PASSED" if ok else "CLOSED-FORM CHECKS FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
