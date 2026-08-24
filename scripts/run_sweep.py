"""Run the pipeline (DMG + Cost JARs) over each generated profile.

For each profile_S{NN} under --sweep-dir:
  1. copy its 5 input files into Input/
  2. clear Output/GeneratedDMs/, copy the base model XML
  3. run DMgeneration.jar
  4. copy Output/signatures_<base>(.txt) into Input/ and normalize sig paths
  5. run CostCalculation.jar
  6. archive Output/GeneratedDMs/ to Output/sweep/profile_S{NN}/
  7. copy params.json alongside

Restores the original Input/ files at the end.

CLI:
    python -m scripts.run_sweep --sweep-dir Input/profiles/sweep \\
        --archive-dir Output/sweep --base-model TPC_C --java java --heap 2g
"""
from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

from lib.parse_costs import normalize_sig_paths as _normalize_sig_paths
from scripts.generate_profile import PROFILE_FILES, PROJECT_ROOT


INPUT_DIR  = PROJECT_ROOT / "Input"
CONF_DIR   = PROJECT_ROOT / "conf"
OUTPUT_DIR = PROJECT_ROOT / "Output"
OUTPUT_DMS_DIR = OUTPUT_DIR / "GeneratedDMs"
MODELS_DIR = INPUT_DIR / "models"
JAR_DMG  = PROJECT_ROOT / "jars" / "DMgeneration.jar"
JAR_COST = PROJECT_ROOT / "jars" / "CostCalculation.jar"

# CostCalculation.jar reads these from conf/ (NOT Input/). queries_dmg.txt is
# DMG-only and stays in Input/. Without syncing conf/, every profile computes
# identical baseline costs.
CONF_FILES: tuple[str, ...] = ("config.txt", "keys.txt", "index.txt", "queries.txt")


def _snapshot_input(snapshot_dir: Path) -> None:
    """Snapshot current Input/ profile files + signatures, and conf/ files."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for fname in PROFILE_FILES:
        src = INPUT_DIR / fname
        if src.exists():
            shutil.copy2(src, snapshot_dir / fname)
    for src in INPUT_DIR.glob("signatures_*.txt"):
        shutil.copy2(src, snapshot_dir / src.name)
    # conf/ snapshot kept in a subdir so restore can target the right dir
    conf_snap = snapshot_dir / "_conf"
    conf_snap.mkdir(exist_ok=True)
    for fname in CONF_FILES:
        src = CONF_DIR / fname
        if src.exists():
            shutil.copy2(src, conf_snap / fname)


def _restore_input(snapshot_dir: Path) -> None:
    for f in snapshot_dir.iterdir():
        if f.is_dir():
            continue
        shutil.copy2(f, INPUT_DIR / f.name)
    conf_snap = snapshot_dir / "_conf"
    if conf_snap.is_dir():
        for f in conf_snap.iterdir():
            shutil.copy2(f, CONF_DIR / f.name)


def _activate(profile_dir: Path) -> None:
    """Copy profile files into Input/ (for DMG) AND conf/ (for Cost JAR).

    CostCalculation.jar reads keys/index/config/queries from conf/, so the
    conf/ copy is what actually makes selectivity/index changes take effect.
    """
    for fname in PROFILE_FILES:
        src = profile_dir / fname
        if src.exists():
            shutil.copy2(src, INPUT_DIR / fname)
    CONF_DIR.mkdir(exist_ok=True)
    for fname in CONF_FILES:
        src = profile_dir / fname
        if src.exists():
            shutil.copy2(src, CONF_DIR / fname)


def _run(cmd: list[str], log_file: Path, inactivity_timeout_s: float) -> int:
    """Run a subprocess streaming output to a log file. Returns exit code.

    Kills the process if it produces no output for ``inactivity_timeout_s``.
    Reads stdout on a background thread: a blocking ``for line in proc.stdout``
    loop can never observe silence, so the timeout would never fire.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None

        line_q: queue.Queue[str | None] = queue.Queue()

        def _reader() -> None:
            try:
                for ln in proc.stdout:
                    line_q.put(ln)
            finally:
                line_q.put(None)

        threading.Thread(target=_reader, daemon=True).start()

        last_activity = time.perf_counter()
        while True:
            try:
                line = line_q.get(timeout=1.0)
            except queue.Empty:
                if time.perf_counter() - last_activity > inactivity_timeout_s:
                    proc.kill()
                    logf.write(f"\n[TIMEOUT after {inactivity_timeout_s}s of inactivity]\n")
                    break
                continue
            if line is None:
                break
            logf.write(line)
            logf.flush()
            last_activity = time.perf_counter()
        return proc.wait()


def _run_phase1(base_model: str, java: str, heap: str,
                log_file: Path, inactivity_timeout_s: float) -> bool:
    if OUTPUT_DMS_DIR.exists():
        shutil.rmtree(OUTPUT_DMS_DIR)
    OUTPUT_DMS_DIR.mkdir(parents=True)
    canonical_xml = MODELS_DIR / f"{base_model}.xml"
    if not canonical_xml.exists():
        print(f"  [ERROR] base model {canonical_xml} not found"); return False
    shutil.copy2(canonical_xml, OUTPUT_DMS_DIR / f"{base_model}.xml")

    sig_out = f"Output/signatures_{base_model}"
    cmd = [
        java, f"-Xmx{heap}", "-jar", str(JAR_DMG),
        "Output/GeneratedDMs/", base_model, "Input/queries_dmg.txt", sig_out,
    ]
    rc = _run(cmd, log_file, inactivity_timeout_s)
    if rc != 0:
        print(f"  [ERROR] DMG exited {rc}"); return False

    sig_src = PROJECT_ROOT / f"Output/signatures_{base_model}.txt"
    if not sig_src.exists():
        sig_src = PROJECT_ROOT / f"Output/signatures_{base_model}"
    if not sig_src.exists():
        print(f"  [ERROR] signatures file not produced"); return False
    shutil.copy2(sig_src, INPUT_DIR / f"signatures_{base_model}.txt")
    _normalize_sig_paths(INPUT_DIR / f"signatures_{base_model}.txt")
    return True


def _run_phase2(base_model: str, java: str, heap: str,
                log_file: Path, inactivity_timeout_s: float) -> bool:
    cmd = [
        java, f"-Xmx{heap}", "-jar", str(JAR_COST),
        f"signatures_{base_model}.txt",
    ]
    rc = _run(cmd, log_file, inactivity_timeout_s)
    if rc != 0:
        print(f"  [ERROR] Cost exited {rc}"); return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Run the pipeline over a profile sweep.")
    p.add_argument("--sweep-dir", type=Path,
                   default=PROJECT_ROOT / "Input" / "profiles" / "sweep")
    p.add_argument("--archive-dir", type=Path,
                   default=PROJECT_ROOT / "Output" / "sweep")
    p.add_argument("--base-model", default="TPC_C")
    p.add_argument("--java", default="java")
    p.add_argument("--heap", default="2g")
    p.add_argument("--inactivity-timeout-min", type=float, default=30.0)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip profiles whose archive dir already has 135+ costs.txt files.")
    args = p.parse_args()

    if not JAR_DMG.exists() or not JAR_COST.exists():
        raise SystemExit(f"Missing JAR(s): {JAR_DMG} / {JAR_COST}")
    if not args.sweep_dir.is_dir():
        raise SystemExit(f"Sweep dir not found: {args.sweep_dir}")

    profile_dirs = sorted(d for d in args.sweep_dir.iterdir()
                          if d.is_dir() and d.name.startswith("profile_"))
    if not profile_dirs:
        raise SystemExit(f"No profile_* dirs in {args.sweep_dir}")

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    snapshot = args.archive_dir / "_input_snapshot"
    _snapshot_input(snapshot)
    print(f"Saved current Input/ to {snapshot}")

    inactivity_s = args.inactivity_timeout_min * 60.0
    run_log: list[dict] = []

    try:
        for i, pdir in enumerate(profile_dirs):
            pid = pdir.name.replace("profile_", "")
            arch = args.archive_dir / pdir.name
            log_dir = arch / "_logs"

            if args.skip_existing and arch.exists() and \
                    len(list(arch.rglob("costs.txt"))) >= 135:
                print(f"[{i+1}/{len(profile_dirs)}] {pid}: skipping (already complete)")
                continue

            print(f"[{i+1}/{len(profile_dirs)}] {pid}: activating")
            _activate(pdir)

            t0 = time.perf_counter()
            print(f"  phase1 (DMG)...", flush=True)
            ok1 = _run_phase1(args.base_model, args.java, args.heap,
                              log_dir / "phase1.log", inactivity_s)
            t1 = time.perf_counter()
            ok2 = False
            if ok1:
                print(f"  phase2 (Cost)...", flush=True)
                ok2 = _run_phase2(args.base_model, args.java, args.heap,
                                  log_dir / "phase2.log", inactivity_s)
            t2 = time.perf_counter()

            if ok2:
                if arch.exists():
                    shutil.rmtree(arch)
                shutil.copytree(OUTPUT_DMS_DIR, arch / "GeneratedDMs")
                shutil.copy2(pdir / "params.json", arch / "params.json")
                n_costs = len(list((arch / "GeneratedDMs").rglob("costs.txt")))
                print(f"  archived {n_costs} cost files "
                      f"(phase1: {t1 - t0:.1f}s, phase2: {t2 - t1:.1f}s)")
            else:
                print(f"  FAILED — not archiving")

            run_log.append({
                "profile_id": pid,
                "phase1_ok": ok1,
                "phase2_ok": ok2,
                "phase1_seconds": t1 - t0,
                "phase2_seconds": t2 - t1,
            })

        (args.archive_dir / "run_log.json").write_text(
            json.dumps(run_log, indent=2), encoding="utf-8"
        )
    finally:
        _restore_input(snapshot)
        print(f"Restored Input/ from {snapshot}")


if __name__ == "__main__":
    main()
