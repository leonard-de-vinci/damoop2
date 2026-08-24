"""Compute the REAL cost of one user-specified config (ground truth for the predictor).

Given a schema config (selectivities, indexes, counts) + a target (model, volume, servers)
+ a query-frequency workload, this:
  1. generates a one-off profile from the config (scripts.generate_profile),
  2. activates it into Input/ AND conf/ (CostCalculation.jar reads conf/),
  3. runs DMgeneration.jar + CostCalculation.jar,
  4. parses the chosen model's cost at (volume, servers),
  5. reweights by the workload frequencies (lib.scenarios.reweight_costs),
  6. restores the original Input/ + conf/.

Prints a single JSON line with the 3 actual costs. Used by views/predict.py "Validate".

CLI:
    python -m scripts.simulate_one --config-file cfg.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from scripts.generate_profile import NON_KEY_COLUMNS, generate_profile
from scripts.run_sweep import (
    OUTPUT_DIR, OUTPUT_DMS_DIR,
    _activate, _snapshot_input, _restore_input, _run_phase1, _run_phase2,
)
from lib.parse_costs import build_costs_dataframe
from lib.scenarios import reweight_costs
from lib.constants import DEFAULT_QUERY_FREQUENCIES, QUERY_COST_STEMS

TARGETS = ["total_time_cost", "total_environmental_cost", "total_financial_cost"]


def _rmtree_safe(path: Path) -> None:
    """Best-effort recursive delete that tolerates long paths / read-only files.

    The generated model tree nests up to depth 9 (paths ~300 chars). A plain
    rmtree can partially fail on a locked/read-only file and leave a half-tree;
    here we clear the read-only bit and swallow residue (it lives under %TEMP%).
    """
    def _onerror(func, p, _exc):
        try:
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            pass
    shutil.rmtree(path, onerror=_onerror)


def simulate_frame(cfg: dict, base_model: str = "TPC_C", java: str = "java", heap: str = "2g"):
    """Run the REAL pipeline for the config and return the workload-reweighted
    cost frame for ALL generated models at (data_volume, servers).

    One pipeline run costs the same whether you read one model or all 135, so
    callers that need a full actual ranking (Predict page "Recommend" validation)
    get it for free.
    """
    spec = {
        "selectivities": {c: int(cfg["selectivities"][c]) for c in NON_KEY_COLUMNS},
        "index_flags":   {c: bool(cfg["index_flags"][c]) for c in NON_KEY_COLUMNS},
        "customer_count": int(cfg["customer_count"]),
        "order_count":    int(cfg["order_count"]),
        "profile_id": "SIM", "seed_idx": -1,
    }
    volume = int(cfg["data_volume"])
    servers = int(cfg["servers"])
    freqs = {s: int(cfg["freqs"].get(s, DEFAULT_QUERY_FREQUENCIES[s])) for s in QUERY_COST_STEMS}

    snap = Path(tempfile.mkdtemp(prefix="sim_snap_"))
    prof = Path(tempfile.mkdtemp(prefix="sim_prof_"))
    # Move-aside targets live UNDER Output/ (same volume as GeneratedDMs) so the
    # backup/restore are atomic os.replace() renames, not deep copytree walks. The
    # model tree nests to depth 9 (~300-char paths); copying it twice per Validate
    # was slow and, worse, could half-fail on a transient lock and leave a broken
    # live tree. A rename is one metadata op and cannot partially fail.
    aside = OUTPUT_DIR / f".sim_live_{os.path.basename(snap)}"      # holds the user's tree
    freshaway = OUTPUT_DIR / f".sim_fresh_{os.path.basename(prof)}"  # holds the sim's tree
    had_dms = OUTPUT_DMS_DIR.exists()
    try:
        _snapshot_input(snap)
        if had_dms:  # move the user's live cost output aside (fast, atomic)
            os.replace(OUTPUT_DMS_DIR, aside)
        generate_profile(spec, prof)
        _activate(prof)

        log_dir = OUTPUT_DIR / "_sim_logs"
        if not _run_phase1(base_model, java, heap, log_dir / "p1.log", 600):
            raise RuntimeError("DMgeneration.jar failed")
        if not _run_phase2(base_model, java, heap, log_dir / "p2.log", 600):
            raise RuntimeError("CostCalculation.jar failed")

        # Parse ONLY the freshly-generated GeneratedDMs (135 files), not all of Output/
        # — Output/ also holds the sweep archive (13,500+ costs.txt) which would make this
        # ~30x slower and could mix in stale rows.
        df = build_costs_dataframe(OUTPUT_DMS_DIR)
        sub = df[(df["data_volume"] == volume) & (df["servers"] == servers)]
        if sub.empty:
            raise RuntimeError(f"No cost rows for volume={volume} servers={servers}")
        return reweight_costs(sub.copy(), freqs, DEFAULT_QUERY_FREQUENCIES)
    finally:
        _restore_input(snap)
        if had_dms:
            # Restore in two atomic renames: park the sim's fresh tree, then swing
            # the user's tree back. Deleting the fragile deep tree is deferred to a
            # best-effort pass on a throwaway dir, so it can never block the restore.
            if OUTPUT_DMS_DIR.exists():
                os.replace(OUTPUT_DMS_DIR, freshaway)
            if aside.exists():
                os.replace(aside, OUTPUT_DMS_DIR)
        for junk in (snap, prof, aside, freshaway):
            if junk.exists():
                _rmtree_safe(junk)


def simulate(cfg: dict, base_model: str = "TPC_C", java: str = "java", heap: str = "2g") -> dict:
    frame = simulate_frame(cfg, base_model, java, heap)
    model = cfg["model"]          # xml_file id, e.g. "TPC_C.xml" or "Merge_.../..."
    sub = frame[frame["xml_file"] == model]
    if sub.empty:
        # fall back to base model name match if path id differs
        sub = frame[frame["xml_name"] == model]
    if sub.empty:
        raise RuntimeError(f"No cost row for model={model}")
    row = sub.iloc[0]
    return {t: float(row[t]) for t in TARGETS}


def main() -> None:
    p = argparse.ArgumentParser(description="Compute the real cost of one config.")
    p.add_argument("--config-file", type=Path, required=True)
    p.add_argument("--base-model", default="TPC_C")
    p.add_argument("--java", default="java")
    p.add_argument("--heap", default="2g")
    args = p.parse_args()
    cfg = json.loads(args.config_file.read_text(encoding="utf-8"))
    try:
        actual = simulate(cfg, args.base_model, args.java, args.heap)
        print("RESULT " + json.dumps({"ok": True, "actual": actual}))
    except Exception as e:  # noqa: BLE001
        print("RESULT " + json.dumps({"ok": False, "error": str(e)}))


if __name__ == "__main__":
    main()
