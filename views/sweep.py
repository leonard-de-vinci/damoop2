"""Profile Sweep page — generate, run, and inspect parametric profile sweeps.

Each **Generate** creates an independent, kept *run* (a batch of N profiles) under
``Input/profiles/sweep_runs/<run_id>/`` and records it in a persistent registry, so past
generations are never overwritten — you can keep a history, switch the active run, and delete
old ones. The pipeline / assemble / results sections all operate on the **active** run.

Sections:
  1. Generate Profiles   — LHS-sample any N profiles into a new run dir (fast, ~seconds)
  2. Generation History  — every run + save path + status; pick the active run or delete one
  3. Run Pipeline        — run DMG + Cost JARs over the active run's profiles (~hours)
  4. Assemble Dataset    — build the ML training dataset (Output/sweep_ml.parquet) from the active run
  5. Results Explorer    — browse the active run's profiles + preview the CSV
"""
from __future__ import annotations

import html as _html
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from lib.constants import (
    ACCENT,
    BG_CARD,
    BORDER,
    PROJECT_ROOT,
    TEXT_MAIN,
    TEXT_MUTED,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SWEEP_RUNS_DIR   = PROJECT_ROOT / "Input" / "profiles" / "sweep_runs"   # generated batches
ARCHIVE_RUNS_DIR = PROJECT_ROOT / "Output" / "sweep_runs"              # JAR-output archives
REGISTRY_PATH    = SWEEP_RUNS_DIR / "registry.json"
DATASET_PATH     = PROJECT_ROOT / "Output" / "sweep_ml.parquet"        # canonical assembled dataset
LEGACY_CSV_PATH  = PROJECT_ROOT / "Output" / "sweep_ml.csv"            # pre-parquet assemblies


def _dataset_path() -> Path:
    """Canonical parquet dataset, falling back to a legacy CSV if that's all there is."""
    return DATASET_PATH if DATASET_PATH.exists() or not LEGACY_CSV_PATH.exists() else LEGACY_CSV_PATH

# Pre-history single working set (from before runs were tracked) — adopted as run "legacy".
LEGACY_SWEEP_DIR   = PROJECT_ROOT / "Input" / "profiles" / "sweep"
LEGACY_ARCHIVE_DIR = PROJECT_ROOT / "Output" / "sweep"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _format_log_html(text: str) -> str:
    return _html.escape(text)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _abs(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _list_profiles(sweep_dir: Path) -> list[Path]:
    if not sweep_dir.is_dir():
        return []
    return sorted(d for d in sweep_dir.iterdir()
                  if d.is_dir() and d.name.startswith("profile_"))


def _list_archived(archive_dir: Path) -> list[Path]:
    if not archive_dir.is_dir():
        return []
    return sorted(d for d in archive_dir.iterdir()
                  if d.is_dir() and d.name.startswith("profile_"))


# A profile is "complete" once its archive holds at least this many costs.txt.
COMPLETE_MIN_COST_FILES = 135

# Completed archives are immutable (run_sweep copies them wholesale and never
# touches them again), so their expensive rglob counts can be memoized for the
# process lifetime. Incomplete/in-progress archives are always recounted.
_COMPLETE_COUNT_CACHE: dict[str, int] = {}


def _count_archived_cost_files(pdir: Path) -> int:
    gdm = pdir / "GeneratedDMs"
    if not gdm.is_dir():
        return 0
    key = str(gdm)
    cached = _COMPLETE_COUNT_CACHE.get(key)
    if cached is not None:
        return cached
    n = len(list(gdm.rglob("costs.txt")))
    if n >= COMPLETE_MIN_COST_FILES:
        _COMPLETE_COUNT_CACHE[key] = n
    return n


def _count_complete(archive_dir: Path) -> int:
    return sum(1 for p in _list_archived(archive_dir)
               if _count_archived_cost_files(p) >= COMPLETE_MIN_COST_FILES)


# ---------------------------------------------------------------------------
# Run registry (persistent history)
# ---------------------------------------------------------------------------
def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        try:
            reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            if isinstance(reg, dict) and isinstance(reg.get("runs"), list):
                reg.setdefault("active", "")
                return reg
        except Exception:
            pass
    return {"active": "", "runs": []}


def _save_registry(reg: dict) -> None:
    SWEEP_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def _ensure_legacy_registered(reg: dict) -> dict:
    """Adopt a pre-existing Input/profiles/sweep set as run 'legacy' so it isn't orphaned."""
    if any(r.get("run_id") == "legacy" for r in reg["runs"]):
        return reg
    legacy_profiles = _list_profiles(LEGACY_SWEEP_DIR)
    if not legacy_profiles:
        return reg
    reg["runs"].insert(0, {
        "run_id": "legacy",
        "label": "Existing set (pre-history)",
        "n": len(legacy_profiles),
        "seed": None,
        "created": "",
        "sweep_dir": _rel(LEGACY_SWEEP_DIR),
        "archive_dir": _rel(LEGACY_ARCHIVE_DIR),
    })
    if not reg["active"]:
        reg["active"] = "legacy"
    _save_registry(reg)
    return reg


def _record(reg: dict, run_id: str) -> dict | None:
    return next((r for r in reg["runs"] if r.get("run_id") == run_id), None)


def _active_record(reg: dict) -> dict | None:
    return _record(reg, reg.get("active", ""))


def _add_run(reg: dict, *, run_id: str, label: str, n: int, seed: int,
             sweep_dir: Path, archive_dir: Path) -> None:
    reg["runs"].append({
        "run_id": run_id,
        "label": label or run_id,
        "n": int(n),
        "seed": int(seed),
        "created": datetime.now().isoformat(timespec="seconds"),
        "sweep_dir": _rel(sweep_dir),
        "archive_dir": _rel(archive_dir),
    })
    reg["active"] = run_id
    _save_registry(reg)


def _delete_run(reg: dict, run_id: str) -> None:
    rec = _record(reg, run_id)
    if rec is not None:
        for key in ("sweep_dir", "archive_dir"):
            d = _abs(rec.get(key, ""))
            if d.exists() and d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
    reg["runs"] = [r for r in reg["runs"] if r.get("run_id") != run_id]
    if reg.get("active") == run_id:
        reg["active"] = reg["runs"][-1]["run_id"] if reg["runs"] else ""
    _save_registry(reg)


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------
def _run_subprocess_blocking(cmd: list[str], log_ph, status_key: str) -> int:
    """Run a subprocess synchronously, streaming stdout to the UI."""
    st.session_state[status_key] = "running"
    log_text = ""
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_text += line
            log_ph.markdown(f'<div class="log-box">{_format_log_html(log_text)}</div>',
                            unsafe_allow_html=True)
        rc = proc.wait()
        st.session_state[status_key] = "ok" if rc == 0 else "error"
        return rc
    except Exception as e:  # noqa: BLE001
        log_text += f"\n[ERROR] {e}\n"
        log_ph.markdown(f'<div class="log-box">{_format_log_html(log_text)}</div>',
                        unsafe_allow_html=True)
        st.session_state[status_key] = "error"
        return 1


def _run_sweep_streaming(cmd: list[str], log_ph, progress_ph, status_key: str,
                         inactivity_timeout_s: float,
                         sweep_dir: Path, archive_dir: Path) -> int:
    """Run the long sweep subprocess, streaming logs and showing live progress.

    Watches ``archive_dir`` to count completed profiles (>=135 costs.txt) in real time.
    """
    st.session_state[status_key] = "running"
    log_text = ""
    t_start = time.perf_counter()

    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
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

    total = len(_list_profiles(sweep_dir))
    last_activity = time.perf_counter()
    timed_out = False

    while True:
        now = time.perf_counter()
        try:
            line = line_q.get(timeout=1.0)
        except queue.Empty:
            idle_s = now - last_activity
            done = _count_complete(archive_dir)
            elapsed = _fmt_elapsed(now - t_start)
            if total > 0:
                progress_ph.progress(min(done / total, 1.0),
                                     text=f"{done} / {total} profiles complete · {elapsed} elapsed")
            if idle_s > inactivity_timeout_s:
                proc.kill()
                timed_out = True
                log_text += f"\n[TIMEOUT] No activity for {_fmt_elapsed(idle_s)}. Killed.\n"
                break
            continue

        if line is None:
            break

        last_activity = time.perf_counter()
        log_text += line
        log_ph.markdown(f'<div class="log-box">{_format_log_html(log_text)}</div>',
                        unsafe_allow_html=True)

    proc.wait()
    done = _count_complete(archive_dir)
    elapsed = _fmt_elapsed(time.perf_counter() - t_start)
    if total > 0:
        progress_ph.progress(min(done / total, 1.0),
                             text=f"{done} / {total} profiles complete · {elapsed} elapsed")
    st.session_state[status_key] = "timeout" if timed_out else ("ok" if proc.returncode == 0 else "error")
    return proc.returncode


# ---------------------------------------------------------------------------
# Section 1 — Generate
# ---------------------------------------------------------------------------
def _section_generate(reg: dict) -> None:
    st.markdown('<div class="section-heading">1 · Generate Profiles</div>',
                unsafe_allow_html=True)
    st.caption(
        "Sample N profiles via Latin Hypercube across 18 dimensions "
        "(8 selectivities + 8 index flags + customer_count + order_count). "
        "Each generation is saved as its own **run** — previous runs are kept (see history below). "
        "Fast (~seconds)."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        n_profiles = st.number_input("N profiles", min_value=1, max_value=100_000,
                                     value=500, step=1, key="sw_n",
                                     help="Any number — LHS scales to thousands.")
    with c2:
        seed = st.number_input("Seed", min_value=0, max_value=9_999_999, value=42,
                               step=1, key="sw_seed",
                               help="Same seed + N reproduces the exact same profiles.")
    with c3:
        label = st.text_input("Label (optional)", value="", key="sw_label",
                              placeholder="e.g. 500-profile main sweep")

    gen_btn = st.button("Generate Profiles", type="primary", key="sw_gen_btn")

    log_ph = st.empty()
    if gen_btn:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{ts}_n{int(n_profiles)}_seed{int(seed)}"
        sweep_dir = SWEEP_RUNS_DIR / run_id
        archive_dir = ARCHIVE_RUNS_DIR / run_id
        cmd = [
            sys.executable, "-m", "scripts.sweep_profiles",
            "--n", str(int(n_profiles)),
            "--seed", str(int(seed)),
            "--out", str(sweep_dir),
        ]
        with st.spinner(f"Generating {int(n_profiles)} profiles..."):
            rc = _run_subprocess_blocking(cmd, log_ph, "sw_gen_status")
        if rc == 0:
            _add_run(reg, run_id=run_id, label=label, n=int(n_profiles), seed=int(seed),
                     sweep_dir=sweep_dir, archive_dir=archive_dir)
            st.success(f"Generated {int(n_profiles)} profiles.")
            st.info(f"Saved to  `{_rel(sweep_dir)}`  — now the **active** run.")
            st.rerun()
        else:
            st.error("Generation failed — see log above.")


# ---------------------------------------------------------------------------
# Section 2 — Generation History (registry: view / activate / delete)
# ---------------------------------------------------------------------------
def _section_history(reg: dict) -> None:
    st.markdown('<div class="section-heading">2 · Generation History</div>',
                unsafe_allow_html=True)

    runs = reg["runs"]
    if not runs:
        st.info("No runs yet. Generate a batch in section 1 to start a history.")
        return

    # Overview table -------------------------------------------------------
    rows = []
    for r in runs:
        sweep_dir = _abs(r.get("sweep_dir", ""))
        archive_dir = _abs(r.get("archive_dir", ""))
        on_disk = len(_list_profiles(sweep_dir))
        complete = _count_complete(archive_dir)
        rows.append({
            "active": "●" if r.get("run_id") == reg.get("active") else "",
            "label": r.get("label", r.get("run_id")),
            "run_id": r.get("run_id"),
            "N": r.get("n"),
            "on_disk": on_disk,
            "pipeline": f"{complete}/{on_disk}" if on_disk else "—",
            "seed": r.get("seed"),
            "created": r.get("created", "")[:19].replace("T", " "),
            "saved_at": r.get("sweep_dir", ""),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "`on_disk` = generated profiles present · `pipeline` = profiles with JAR costs done "
        "(≥135 files) / total. The **active** run (●) is what sections 3–5 operate on."
    )

    # Active-run selector --------------------------------------------------
    ids = [r["run_id"] for r in runs]
    labels = {r["run_id"]: f'{r.get("label", r["run_id"])}  ·  {r.get("n")} profiles  ·  {r["run_id"]}'
              for r in runs}
    cur = reg.get("active") if reg.get("active") in ids else ids[-1]
    c1, c2 = st.columns([2, 1])
    with c1:
        chosen = st.selectbox("Active run", options=ids, index=ids.index(cur),
                              format_func=lambda i: labels[i], key="sw_active_sel")
    if chosen != reg.get("active"):
        reg["active"] = chosen
        _save_registry(reg)
        st.rerun()

    active = _active_record(reg)
    if active:
        with c2:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.caption(f"Sweep dir: `{active.get('sweep_dir')}`")
        st.caption(f"Archive (JAR outputs): `{active.get('archive_dir')}`  ·  "
                   f"Assembled dataset target: `{_rel(DATASET_PATH)}`")

    # Delete a run ---------------------------------------------------------
    with st.expander("Delete a run"):
        del_id = st.selectbox("Run to delete", options=ids,
                              format_func=lambda i: labels[i], key="sw_del_sel")
        del_rec = _record(reg, del_id)
        st.warning(
            f"This permanently removes the generated profiles **and** any archived JAR outputs "
            f"for this run:\n\n"
            f"- `{del_rec.get('sweep_dir')}`\n- `{del_rec.get('archive_dir')}`\n\n"
            f"The assembled `Output/sweep_ml.parquet` is not touched."
        )
        confirm = st.checkbox("Yes, I understand — delete this run's files", key=f"sw_del_ok_{del_id}")
        if st.button("Delete run", type="secondary", disabled=not confirm, key="sw_del_btn"):
            _delete_run(reg, del_id)
            st.success(f"Deleted run {del_id}.")
            st.rerun()


# ---------------------------------------------------------------------------
# Section 3 — Run pipeline (active run)
# ---------------------------------------------------------------------------
def _section_run_pipeline(reg: dict) -> None:
    st.markdown('<div class="section-heading">3 · Run Pipeline (DMG + Cost)</div>',
                unsafe_allow_html=True)

    active = _active_record(reg)
    if active is None:
        st.info("No active run. Generate a batch first.")
        return

    sweep_dir = _abs(active["sweep_dir"])
    archive_dir = _abs(active["archive_dir"])
    profiles = _list_profiles(sweep_dir)
    if not profiles:
        st.info(f"Active run `{active['run_id']}` has no profiles on disk (`{active['sweep_dir']}`).")
        return

    n_complete = _count_complete(archive_dir)
    st.caption(f"Active run: **{active.get('label')}** (`{active['run_id']}`)")

    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(
            f"Will run JARs on **{len(profiles)}** profiles "
            f"(~6–10 s each, total ~{max(1, len(profiles) * 8 // 60)} min). "
            f"Already complete: **{n_complete} / {len(profiles)}**."
        )
    with c2:
        skip_existing = st.checkbox("Skip existing", value=True, key="sw_skip")

    timeout_min = st.slider("Inactivity timeout (minutes)", min_value=5, max_value=120,
                            value=30, key="sw_timeout",
                            help="Kill the JAR if it produces no output for this long.")

    run_btn = st.button("Run Sweep", type="primary", key="sw_run_btn")
    progress_ph = st.empty()
    log_ph = st.empty()

    if run_btn:
        cmd = [
            sys.executable, "-m", "scripts.run_sweep",
            "--sweep-dir", str(sweep_dir),
            "--archive-dir", str(archive_dir),
            "--inactivity-timeout-min", str(int(timeout_min)),
        ]
        if skip_existing:
            cmd.append("--skip-existing")
        with st.spinner("Running sweep — this may take a while; safe to leave running…"):
            rc = _run_sweep_streaming(cmd, log_ph, progress_ph, "sw_run_status",
                                      inactivity_timeout_s=float(timeout_min) * 60.0,
                                      sweep_dir=sweep_dir, archive_dir=archive_dir)
        if rc == 0:
            st.success("Sweep complete!")
            st.rerun()
        else:
            st.error("Sweep ended with errors — see log above.")


# ---------------------------------------------------------------------------
# Section 4 — Assemble (active run -> canonical CSV)
# ---------------------------------------------------------------------------
def _section_assemble(reg: dict) -> None:
    st.markdown('<div class="section-heading">4 · Assemble ML Dataset</div>',
                unsafe_allow_html=True)
    st.caption(
        "Walk the active run's archived profiles, parse all costs.txt files, broadcast profile "
        "parameters onto every row, and write the master dataset `Output/sweep_ml.parquet` "
        "(the canonical dataset the EDA / training read — ~20x smaller than CSV)."
    )

    active = _active_record(reg)
    if active is None:
        st.info("No active run. Generate + run a batch first.")
        return

    archive_dir = _abs(active["archive_dir"])
    n_complete = _count_complete(archive_dir)
    if n_complete == 0:
        st.info(f"Active run `{active['run_id']}` has no completed archived profiles yet. "
                "Run the pipeline first.")
        return

    st.caption(f"Active run **{active.get('label')}** — **{n_complete}** complete profiles ready.")
    asm_btn = st.button("Build Dataset", type="primary", key="sw_asm_btn")

    log_ph = st.empty()
    if asm_btn:
        cmd = [
            sys.executable, "-m", "scripts.assemble_dataset",
            "--archive-dir", str(archive_dir),
            "--out", str(DATASET_PATH),
        ]
        with st.spinner("Assembling CSV..."):
            rc = _run_subprocess_blocking(cmd, log_ph, "sw_asm_status")
        if rc == 0:
            st.success(f"Dataset written to `{_rel(DATASET_PATH)}`")
            st.rerun()
        else:
            st.error("Assembly failed — see log above.")


# ---------------------------------------------------------------------------
# Section 5 — Results explorer (active run)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _build_profiles_table(sweep_dir_str: str, archive_dir_str: str,
                          _n_profiles: int, _n_complete: int) -> pd.DataFrame:
    """Profiles overview for the Results tab.

    Reading params.json for every profile (possibly hundreds) on each rerun is the
    hot path here; cached on profile/completion counts, which are the only
    things that change while a sweep progresses."""
    from scripts.generate_profile import NON_KEY_COLUMNS
    sweep_dir, archive_dir = Path(sweep_dir_str), Path(archive_dir_str)
    rows = []
    for pdir in _list_profiles(sweep_dir):
        params_file = pdir / "params.json"
        if not params_file.exists():
            continue
        params = json.loads(params_file.read_text(encoding="utf-8"))
        sels = params.get("selectivities", {})
        flags = params.get("index_flags", {})
        row = {
            "profile_id": params.get("profile_id", pdir.name),
            "cost_files": _count_archived_cost_files(archive_dir / pdir.name),
            "customer_count": params.get("customer_count"),
            "order_count": params.get("order_count"),
        }
        for col in NON_KEY_COLUMNS:
            row[f"sel_{col}"] = sels.get(col)
        for col in NON_KEY_COLUMNS:
            row[f"idx_{col}"] = int(bool(flags.get(col, False)))
        rows.append(row)
    return pd.DataFrame(rows)


# Above this size, skip exact row counts and in-browser downloads entirely —
# the assembled CSV can run to many GB and reading it freezes (or OOMs) the app.
_MAX_EXACT_COUNT_BYTES = 50 * 1024 * 1024
_MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024


@st.cache_data(show_spinner=False)
def _csv_row_count(path_str: str, _mtime: float, size: int) -> tuple[int, bool]:
    """Return (data row count, exact?). Huge files get an estimate from an 8 MB sample."""
    path = Path(path_str)
    if size <= _MAX_EXACT_COUNT_BYTES:
        with path.open(encoding="utf-8") as f:
            return sum(1 for _ in f) - 1, True
    with path.open("rb") as f:
        sample = f.read(8 * 1024 * 1024)
    n_lines = sample.count(b"\n")
    if n_lines < 2:
        return 0, False
    avg_line = len(sample) / n_lines
    return max(0, round(size / avg_line) - 1), False


@st.cache_data(show_spinner=False)
def _csv_bytes(path_str: str, _mtime: float, _size: int) -> bytes:
    return Path(path_str).read_bytes()


@st.cache_data(show_spinner=False)
def _parquet_preview(path_str: str, _mtime: float, _size: int) -> tuple[pd.DataFrame, int, bool]:
    """First 100 rows + exact row count straight from parquet metadata (no full scan)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path_str)
    head = next(pf.iter_batches(batch_size=100)).to_pandas()
    return head, pf.metadata.num_rows, True



def _section_results(reg: dict) -> None:
    st.markdown('<div class="section-heading">5 · Results Explorer</div>',
                unsafe_allow_html=True)

    active = _active_record(reg)
    tabs = st.tabs(["Profiles", "Sweep Log", "Dataset Preview"])

    # ---- Tab 1: Profiles (from the active run's generated set) ----
    with tabs[0]:
        if active is None:
            st.info("No active run.")
        else:
            sweep_dir = _abs(active["sweep_dir"])
            archive_dir = _abs(active["archive_dir"])
            profiles = _list_profiles(sweep_dir)
            if not profiles:
                st.info("Active run has no profiles on disk.")
            else:
                df_profiles = _build_profiles_table(
                    str(sweep_dir), str(archive_dir),
                    len(profiles), _count_complete(archive_dir),
                )
                if not df_profiles.empty:
                    st.caption(
                        f"All {len(df_profiles)} profiles of run **{active.get('label')}** with every "
                        f"sampled dimension (8 selectivities + 8 index flags + 2 counts = 18). "
                        f"`cost_files` shows pipeline progress. Scroll right for all columns."
                    )
                    st.dataframe(df_profiles, width="stretch", height=360)

    # ---- Tab 2: Sweep Log (active run) ----
    with tabs[1]:
        if active is None:
            st.info("No active run.")
        else:
            run_log = _abs(active["archive_dir"]) / "run_log.json"
            if run_log.exists():
                log_data = json.loads(run_log.read_text(encoding="utf-8"))
                if log_data:
                    st.dataframe(pd.DataFrame(log_data), width="stretch", height=320)
                else:
                    st.info("Run log is empty.")
            else:
                st.info("No run_log.json yet — run the pipeline first.")

    # ---- Tab 3: Dataset Preview (canonical parquet, legacy CSV fallback) ----
    with tabs[2]:
        ds = _dataset_path()
        if not ds.exists():
            st.info("No dataset yet — assemble it first.")
        else:
            _stat = ds.stat()
            size_mb = _stat.st_size / (1024 * 1024)
            try:
                if ds.suffix == ".parquet":
                    df_head, total_rows, exact = _parquet_preview(str(ds), _stat.st_mtime, _stat.st_size)
                else:
                    df_head = pd.read_csv(ds, nrows=100)
                    total_rows, exact = _csv_row_count(str(ds), _stat.st_mtime, _stat.st_size)
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed to read dataset: {e}")
                return

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Rows", f"{total_rows:,}" + ("" if exact else " (est.)"))
            c2.metric("Columns", f"{len(df_head.columns):,}")
            c3.metric("File Size", f"{size_mb:,.1f} MB")

            st.caption(f"Preview of first 100 rows from `{_rel(ds)}`:")
            st.dataframe(df_head, width="stretch", height=400)

            if _stat.st_size <= _MAX_DOWNLOAD_BYTES:
                st.download_button("Download Full Dataset",
                                   data=_csv_bytes(str(ds), _stat.st_mtime, _stat.st_size),
                                   file_name=ds.name,
                                   mime="application/octet-stream" if ds.suffix == ".parquet" else "text/csv")
            else:
                st.caption(f"File is too large for an in-browser download — "
                           f"use it directly from `{_rel(ds)}`.")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def render() -> None:
    st.markdown(
        f"""
        <div style="background:{BG_CARD};border:1px solid {BORDER};
                    border-left:3px solid {ACCENT};border-radius:6px;
                    padding:14px 18px;margin-bottom:18px;">
            <div style="font-size:0.86rem;color:{TEXT_MAIN};font-weight:600;">
                Parametric Profile Sweep
            </div>
            <div style="font-size:0.74rem;color:{TEXT_MUTED};margin-top:4px;">
                Generate any number of profiles via Latin Hypercube Sampling. Each generation is a
                kept <b>run</b> with its own save location — build a history, switch the active run,
                or delete old ones. Run the pipeline + assemble the ML CSV per run.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    reg = _ensure_legacy_registered(_load_registry())

    _section_generate(reg)
    st.markdown("---")
    _section_history(reg)
    st.markdown("---")
    _section_run_pipeline(reg)
    st.markdown("---")
    _section_assemble(reg)
    st.markdown("---")
    _section_results(reg)
