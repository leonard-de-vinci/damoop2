"""Pipeline page — run DMG and Cost Calculation phases."""
from __future__ import annotations

import html as _html
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from lib.constants import (
    BG_CARD,
    BORDER,
    CONF_DIR,
    INPUT_DIR,
    JAR_COST,
    JAR_DMG,
    MODELS_DIR,
    OUTPUT_DIR,
    OUTPUT_DMS_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    TEXT_MAIN,
    TEXT_MUTED,
    HISTORY_DIR,
)

from lib.data_loading import (
    active_run_query_mismatch,
    count_dms,
    create_run_snapshot_from_live,
    list_run_ids,
    list_snapshots,
    read_run_metadata,
    build_costs_dataframe,
    backup_output,
    output_backup_exists,
    restore_output,
)
from lib.parse_costs import normalize_sig_paths as _normalize_sig_paths

# CostCalculation.jar reads keys/index/config/queries from conf/, not Input/.
# Sync them right before Phase 2 so UI edits and profile switches take effect.
_CONF_FILES = ("config.txt", "keys.txt", "index.txt", "queries.txt")


def _sync_conf_from_input() -> None:
    """Copy the 4 cost-input files from Input/ into conf/ so the Cost JAR sees them."""
    CONF_DIR.mkdir(exist_ok=True)
    for _fname in _CONF_FILES:
        _src = INPUT_DIR / _fname
        if _src.exists():
            shutil.copy2(_src, CONF_DIR / _fname)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_dm_path_from_file(file_path: Path, root_dir: Path) -> str:
    """Return 'A → B → C' from a file sitting inside root_dir/A/B/C/."""
    try:
        parts = list(file_path.parent.relative_to(root_dir).parts)
    except ValueError:
        parts = []
    return " → ".join(parts) if parts else "(base model)"


def _sig_ordered_paths(sig_path: Path, root_dir: Path) -> list[str]:
    """Return DM transformation paths in generation order from a signatures file.

    The signatures file alternates: schema-line, path-line, schema-line, path-line…
    Path lines start with 'Output/GeneratedDMs'.  The order here is the paper's
    canonical Mx numbering (M0 = base model, M1 = first generated, …).
    """
    if not sig_path.exists():
        return []
    paths: list[str] = []
    for ln in sig_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip().rstrip("/")
        if ln.startswith("Output/GeneratedDMs"):
            rel = ln[len("Output/GeneratedDMs"):].lstrip("/")
            paths.append(" → ".join(rel.split("/")) if rel else "(base model)")
    return paths


def _count_sig_paths(sig_path: Path) -> int:
    """Count DM entries in a signatures file (every other line is a path)."""
    return len(_sig_ordered_paths(sig_path, PROJECT_ROOT))


def _format_log_html(text: str) -> str:
    """HTML-escape the log text for safe rendering inside a log-box div."""
    return _html.escape(text)


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds as e.g. '3m 42s' or '1h 12m'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _render_progress(
    ph, n_done: int, total: int | None, elapsed: float,
    dm_elapsed: float | None = None,
) -> None:
    elapsed_str = _fmt_elapsed(elapsed)
    dm_part = ""
    if dm_elapsed is not None and n_done > 0:
        dm_part = f" &nbsp;·&nbsp; current DM: {_fmt_elapsed(dm_elapsed)}"
    if total and total > 0:
        pct = min(n_done / total, 1.0)
        ph.progress(
            pct,
            text=f"{n_done} / {total} &nbsp;·&nbsp; {int(pct * 100)}% &nbsp;·&nbsp; {elapsed_str} elapsed{dm_part}",
        )
    else:
        ph.markdown(
            f'<div style="font-size:0.78rem;color:{TEXT_MUTED};padding:4px 0;">'
            f'{n_done} DM{"s" if n_done != 1 else ""} generated &nbsp;·&nbsp; {elapsed_str} elapsed{dm_part}'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_history(ph, paths: list[str]) -> None:
    """Render the full ordered list of generated DM paths into a placeholder."""
    if not paths:
        ph.markdown(
            f'<div style="color:{TEXT_MUTED};font-size:0.78rem;padding:8px;">No DMs generated yet…</div>',
            unsafe_allow_html=True,
        )
        return
    rows = "".join(
        f'<div style="display:flex;align-items:baseline;gap:10px;'
        f'padding:4px 0;border-bottom:1px solid {BORDER};">'
        f'<span style="color:{TEXT_MUTED};font-size:0.70rem;min-width:2.2rem;text-align:right;">M{i}</span>'
        f'<span style="color:{TEXT_MAIN};font-size:0.76rem;">{_html.escape(p)}</span>'
        f'</div>'
        for i, p in enumerate(paths)
    )
    ph.markdown(
        f'<div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:6px;'
        f'padding:8px 12px;max-height:380px;overflow-y:auto;font-family:Consolas,monospace;">'
        f'{rows}</div>',
        unsafe_allow_html=True,
    )


def _run_jar(
    cmd: list[str],
    log_key: str,
    status_key: str,
    log_ph=None,
    progress_ph=None,
    history_ph=None,
    history_key: str | None = None,
    watch_dir: Path | None = None,
    watch_pattern: str = "costs.txt",
    total_count: int | None = None,
    sig_path: Path | None = None,
    inactivity_timeout_s: float = 1800,  # 30 minutes
    warn_ph=None,
) -> None:
    st.session_state[log_key] = ""
    st.session_state[status_key] = "running"
    if history_key:
        st.session_state[history_key] = []
    log_placeholder = log_ph if log_ph is not None else st.empty()
    t_start = time.perf_counter()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Background thread feeds lines into a queue so the main loop can
        # tick the elapsed timer every second even when the JAR is silent.
        line_q: queue.Queue[str | None] = queue.Queue()

        def _reader() -> None:
            try:
                for ln in proc.stdout:
                    line_q.put(ln)
            finally:
                line_q.put(None)  # sentinel

        threading.Thread(target=_reader, daemon=True).start()

        line_num = 0
        seen_files: set[str] = set()
        path_history: list[str] = []
        last_activity = time.perf_counter()   # last log line or new file
        last_dm_change = time.perf_counter()  # last time a new output file appeared
        timed_out = False

        while True:
            now = time.perf_counter()

            try:
                line = line_q.get(timeout=1.0)
            except queue.Empty:
                # 1 second with no output — tick timers and check inactivity
                idle_s = now - last_activity
                dm_elapsed = now - last_dm_change

                if progress_ph and watch_dir:
                    _render_progress(
                        progress_ph, len(seen_files), total_count,
                        now - t_start, dm_elapsed=dm_elapsed,
                    )

                # Show escalating warnings
                if warn_ph and idle_s > 300:  # 5 min of silence
                    warn_ph.warning(
                        f"No activity for {_fmt_elapsed(idle_s)}. "
                        f"Process will be killed after {_fmt_elapsed(inactivity_timeout_s)} of inactivity. "
                        f"You can re-run Phase 2 to resume from where it stopped."
                    )

                # Kill after inactivity timeout
                if idle_s > inactivity_timeout_s:
                    proc.kill()
                    timed_out = True
                    st.session_state[log_key] += (
                        f"\n[TIMEOUT] No activity for {_fmt_elapsed(idle_s)}. "
                        f"Process killed. Re-run Phase 2 to resume from completed DMs.\n"
                    )
                    break

                continue

            if line is None:  # sentinel: stdout closed
                break

            last_activity = time.perf_counter()
            st.session_state[log_key] += line
            line_num += 1

            # Every 3 lines: scan disk for newly created files
            if watch_dir and line_num % 3 == 0:
                current_files = {str(p) for p in watch_dir.rglob(watch_pattern)}
                new_files = current_files - seen_files
                if new_files:
                    seen_files = current_files
                    last_activity = time.perf_counter()
                    last_dm_change = time.perf_counter()
                    for f in sorted(new_files):
                        path_history.append(_fmt_dm_path_from_file(Path(f), watch_dir))
                    if history_key:
                        st.session_state[history_key] = list(path_history)
                    if history_ph:
                        _render_history(history_ph, path_history)
                if progress_ph:
                    _render_progress(
                        progress_ph, len(seen_files), total_count,
                        time.perf_counter() - t_start,
                        dm_elapsed=time.perf_counter() - last_dm_change,
                    )

            # Update live log
            log_placeholder.markdown(
                f'<div class="log-box">{_format_log_html(st.session_state[log_key])}</div>',
                unsafe_allow_html=True,
            )

        proc.wait()
        if timed_out:
            st.session_state[status_key] = "timeout"
        else:
            st.session_state[status_key] = "ok" if proc.returncode == 0 else "error"

        # Canonicalise: use signatures-file generation order (= paper's Mx numbering)
        # if available; fall back to alphabetical sort otherwise.
        if watch_dir:
            if sig_path:
                path_history = _sig_ordered_paths(sig_path, watch_dir)
            if not path_history:
                final_files = sorted(str(p) for p in watch_dir.rglob(watch_pattern))
                path_history = [_fmt_dm_path_from_file(Path(f), watch_dir) for f in final_files]
            if history_key:
                st.session_state[history_key] = list(path_history)
            if history_ph:
                _render_history(history_ph, path_history)
            if progress_ph:
                n_final = len({str(p) for p in watch_dir.rglob(watch_pattern)})
                _render_progress(
                    progress_ph, n_final, total_count,
                    time.perf_counter() - t_start,
                )

        suffix = (
            "\n[Process exited successfully]"
            if proc.returncode == 0
            else f"\n[Process exited with code {proc.returncode}]"
        )
        if not timed_out:
            st.session_state[log_key] += suffix

    except FileNotFoundError as exc:
        st.session_state[log_key] = f"ERROR: {exc}\nMake sure Java is installed and on PATH."
        st.session_state[status_key] = "error"

    log_placeholder.markdown(
        f'<div class="log-box">{_format_log_html(st.session_state[log_key])}</div>',
        unsafe_allow_html=True,
    )
    st.cache_data.clear()
    st.session_state["data_bust"] = datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

def render() -> None:
    # Apply any run-ID update that was staged after the selectbox widget was
    # already instantiated in a previous run (setting widget keys post-creation
    # raises StreamlitAPIException).
    if "_pending_run_id" in st.session_state:
        _pending = st.session_state["_pending_run_id"]
        del st.session_state["_pending_run_id"]
        st.session_state["active_run_id"] = _pending
        st.session_state["pl_active_run"] = _pending

    if "active_run_id" not in st.session_state:
        st.session_state["active_run_id"] = "LIVE"
    if "pl_active_run" not in st.session_state:
        st.session_state["pl_active_run"] = st.session_state["active_run_id"]

    n_dms = count_dms(OUTPUT_DMS_DIR)

    # ── Summary cards ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DMgeneration.jar", "Ready" if JAR_DMG.exists()  else "Missing")
    c2.metric("CostCalculation.jar", "Ready" if JAR_COST.exists() else "Missing")
    c3.metric("Cost Files Generated", n_dms)
    c4.metric("Saved Snapshots", len(list_snapshots()))

    st.markdown('<div class="section-heading">Data Run Selection</div>', unsafe_allow_html=True)
    _run_ids = list_run_ids()
    _run_opts = ["LIVE", *_run_ids]
    if st.session_state["active_run_id"] not in _run_opts:
        st.session_state["active_run_id"] = "LIVE"
    if st.session_state["pl_active_run"] not in _run_opts:
        st.session_state["pl_active_run"] = "LIVE"

    _active_run = st.selectbox(
        "Active data run",
        options=_run_opts,
        index=_run_opts.index(st.session_state["pl_active_run"]),
        format_func=lambda r: "Live (Output/GeneratedDMs)" if r == "LIVE" else f"Run {r}",
        key="pl_active_run",
    )
    st.session_state["active_run_id"] = _active_run

    if _active_run != "LIVE":
        _meta = read_run_metadata(_active_run) or {}
        _cost_files = _meta.get("cost_files", "?")
        _created_at = _meta.get("created_at", "")
        st.caption(f"Selected run has {_cost_files} cost files. Created: {_created_at}")

    _is_mismatch, _warn_msg = active_run_query_mismatch()
    if _is_mismatch:
        st.warning(_warn_msg)

    st.divider()

    # ── Base model config ─────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Pipeline Configuration</div>', unsafe_allow_html=True)

    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
    base_model = cfg_col1.text_input(
        "Base Model Name",
        value=st.session_state["base_model"],
        help="Passed as the second argument to DMgeneration.jar (e.g. TPC_C)",
    )
    st.session_state["base_model"] = base_model

    java_cmd = cfg_col2.text_input(
        "Java Executable",
        value="java",
        help="Path to java binary, or just 'java' if on PATH",
    )

    java_heap = cfg_col3.text_input(
        "Max Java Heap (-Xmx)",
        value=st.session_state.get("java_heap", "2g"),
        help="e.g. 2g, 4g, 512m — passed as -Xmx to both JARs to prevent OOM crashes",
    )
    st.session_state["java_heap"] = java_heap

    inactivity_min = cfg_col4.number_input(
        "Inactivity Timeout (min)",
        min_value=1, max_value=120,
        value=int(st.session_state.get("inactivity_timeout_min", 30)),
        help="Kill the JAR if no output and no new files for this many minutes. Re-run to resume.",
    )
    st.session_state["inactivity_timeout_min"] = inactivity_min

    with st.expander("Resolved command arguments"):
        st.code(
            f"# Phase 1 — Data Model Generation\n"
            f"{java_cmd} -Xmx{java_heap} -jar jars/DMgeneration.jar "
            f'"Output/GeneratedDMs/" "{base_model}" "Input/queries_dmg.txt" "Output/signatures_{base_model}"\n\n'
            f"# Phase 2 — Cost Calculation\n"
            f"{java_cmd} -Xmx{java_heap} -jar jars/CostCalculation.jar "
            f'"signatures_{base_model}.txt"',
            language="bash",
        )

    st.divider()

    # ── Run buttons ───────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Run Phases</div>', unsafe_allow_html=True)

    btn_col1, btn_col2, btn_col3 = st.columns(3)

    run_dmg  = btn_col1.button("Run Phase 1 — DMG",      width="stretch", disabled=not JAR_DMG.exists())
    run_cost = btn_col2.button("Run Phase 2 — Cost Calc", width="stretch", disabled=not JAR_COST.exists())
    run_all  = btn_col3.button("Run Full Pipeline",       width="stretch", disabled=not (JAR_DMG.exists() and JAR_COST.exists()))

    sig_out = f"Output/signatures_{base_model}"
    sig_in  = INPUT_DIR / f"signatures_{base_model}.txt"

    def _step_badge(status: str) -> str:
        if status == "pending":
            return '<span style="background:#2B4F7A;color:#D9DEE8;padding:2px 10px;border-radius:4px;font-size:0.8rem;">PENDING</span>'
        if status == "running":
            return '<span style="background:#7A6A2B;color:#D9DEE8;padding:2px 10px;border-radius:4px;font-size:0.8rem;">RUNNING…</span>'
        if status == "ok":
            return '<span style="background:#2B7A4A;color:#fff;padding:2px 10px;border-radius:4px;font-size:0.8rem;">DONE</span>'
        if status == "timeout":
            return '<span style="background:#7A4A2B;color:#fff;padding:2px 10px;border-radius:4px;font-size:0.8rem;">TIMEOUT</span>'
        return '<span style="background:#7A2B2B;color:#fff;padding:2px 10px;border-radius:4px;font-size:0.8rem;">FAILED</span>'

    # ── Phase tracker ─────────────────────────────────────────────────────
    if run_dmg or run_cost or run_all:
        st.divider()
        st.markdown('<div class="section-heading">Pipeline Progress</div>', unsafe_allow_html=True)

        phases_to_run = []
        if run_dmg or run_all:
            phases_to_run.append("phase1")
        if run_cost or run_all:
            phases_to_run.append("phase2")

        tracker_rows: dict = {}
        for ph in ["phase1", "phase2"]:
            cols = st.columns([3, 1, 4])
            label = "Phase 1 — Data Model Generation (DMgeneration.jar)" if ph == "phase1" \
                    else "Phase 2 — Cost Calculation (CostCalculation.jar)"
            cols[0].markdown(f'<span style="color:#D9DEE8">{label}</span>', unsafe_allow_html=True)
            initial = "pending" if ph in phases_to_run else ("ok" if st.session_state.get(
                "dmg_status" if ph == "phase1" else "cost_status") == "ok" else "pending")
            badge_ph = cols[1].empty()
            badge_ph.markdown(_step_badge(initial), unsafe_allow_html=True)
            detail_ph = cols[2].empty()
            tracker_rows[ph] = (badge_ph, detail_ph)

        st.divider()

        # ── Run Phase 1 ──────────────────────────────────────────────────
        if "phase1" in phases_to_run:
            badge_ph, detail_ph = tracker_rows["phase1"]
            badge_ph.markdown(_step_badge("running"), unsafe_allow_html=True)
            detail_ph.markdown('<span style="color:#8A93A6;font-size:0.8rem;">Clearing old output…</span>', unsafe_allow_html=True)

            if OUTPUT_DMS_DIR.exists():
                shutil.rmtree(OUTPUT_DMS_DIR)
            OUTPUT_DMS_DIR.mkdir(parents=True)
            canonical_xml = MODELS_DIR / f"{base_model}.xml"
            if not canonical_xml.exists():
                badge_ph.markdown(_step_badge("error"), unsafe_allow_html=True)
                detail_ph.markdown(f'<span style="color:#E05C5C;font-size:0.8rem;">Base model {base_model}.xml not found in Input/models/</span>', unsafe_allow_html=True)
                st.stop()
            shutil.copy(canonical_xml, OUTPUT_DMS_DIR / f"{base_model}.xml")

            detail_ph.markdown('<span style="color:#8A93A6;font-size:0.8rem;">Running JAR…</span>', unsafe_allow_html=True)
            with st.expander("Phase 1 — Live Log", expanded=True):
                ph1_warn = st.empty()
                ph1_progress = st.empty()
                tab_log1, tab_hist1 = st.tabs(["Live Log", "DM History"])
                with tab_log1:
                    ph1_log_ph = st.empty()
                with tab_hist1:
                    ph1_history_ph = st.empty()
                _render_history(ph1_history_ph, st.session_state.get("dmg_path_history", []))
                _run_jar(
                    [java_cmd, f"-Xmx{java_heap}", "-jar", str(JAR_DMG),
                     "Output/GeneratedDMs/", base_model, "Input/queries_dmg.txt", sig_out],
                    "dmg_log", "dmg_status",
                    log_ph=ph1_log_ph,
                    progress_ph=ph1_progress,
                    history_ph=ph1_history_ph,
                    history_key="dmg_path_history",
                    watch_dir=OUTPUT_DMS_DIR,
                    watch_pattern="*.xml",
                    total_count=None,
                    sig_path=PROJECT_ROOT / f"Output/signatures_{base_model}.txt",
                    inactivity_timeout_s=inactivity_min * 60,
                    warn_ph=ph1_warn,
                )

            dmg_ok = st.session_state["dmg_status"] == "ok"
            badge_ph.markdown(_step_badge("ok" if dmg_ok else "error"), unsafe_allow_html=True)
            if dmg_ok:
                n_xml = len(list(OUTPUT_DMS_DIR.rglob("*.xml"))) - 1
                detail_ph.markdown(f'<span style="color:#4CAF82;font-size:0.8rem;">{n_xml} variant models generated</span>', unsafe_allow_html=True)
                sig_src = PROJECT_ROOT / f"Output/signatures_{base_model}.txt"
                if not sig_src.exists():
                    sig_src = PROJECT_ROOT / f"Output/signatures_{base_model}"
                if sig_src.exists():
                    shutil.copy(sig_src, INPUT_DIR / f"signatures_{base_model}.txt")
            else:
                detail_ph.markdown('<span style="color:#E05C5C;font-size:0.8rem;">Phase 1 failed — check log above</span>', unsafe_allow_html=True)
                if "phase2" in phases_to_run:
                    b2, d2 = tracker_rows["phase2"]
                    b2.markdown(_step_badge("error"), unsafe_allow_html=True)
                    d2.markdown('<span style="color:#E05C5C;font-size:0.8rem;">Skipped — Phase 1 did not complete</span>', unsafe_allow_html=True)
                    st.stop()

        # ── Run Phase 2 ──────────────────────────────────────────────────
        if "phase2" in phases_to_run:
            badge_ph, detail_ph = tracker_rows["phase2"]
            badge_ph.markdown(_step_badge("running"), unsafe_allow_html=True)

            total_dms = _count_sig_paths(sig_in)
            n_existing = len(list(OUTPUT_DMS_DIR.rglob("costs.txt"))) if OUTPUT_DMS_DIR.exists() else 0

            _sync_conf_from_input()   # ensure Cost JAR reads the current Input/ files
            _normalize_sig_paths(sig_in)

            if n_existing > 0:
                detail_ph.markdown(
                    f'<span style="color:#4A7FC1;font-size:0.8rem;">'
                    f'{n_existing}/{total_dms} cost files already exist — JAR will recompute all</span>',
                    unsafe_allow_html=True,
                )
            else:
                detail_ph.markdown(
                    '<span style="color:#8A93A6;font-size:0.8rem;">Running JAR…</span>',
                    unsafe_allow_html=True,
                )

            with st.expander("Phase 2 — Live Log", expanded=True):
                ph2_warn = st.empty()
                ph2_progress = st.empty()
                tab_log2, tab_hist2 = st.tabs(["Live Log", "DM History"])
                with tab_log2:
                    ph2_log_ph = st.empty()
                with tab_hist2:
                    ph2_history_ph = st.empty()
                _render_history(ph2_history_ph, st.session_state.get("cost_path_history", []))
                _run_jar(
                    [java_cmd, f"-Xmx{java_heap}", "-jar", str(JAR_COST),
                     f"signatures_{base_model}.txt"],
                    "cost_log", "cost_status",
                    log_ph=ph2_log_ph,
                    progress_ph=ph2_progress,
                    history_ph=ph2_history_ph,
                    history_key="cost_path_history",
                    watch_dir=OUTPUT_DMS_DIR,
                    watch_pattern="costs.txt",
                    total_count=total_dms or None,
                    sig_path=sig_in,
                    inactivity_timeout_s=inactivity_min * 60,
                    warn_ph=ph2_warn,
                )

            cost_status = st.session_state["cost_status"]
            badge_ph.markdown(_step_badge(cost_status), unsafe_allow_html=True)
            n_costs = len(list(OUTPUT_DMS_DIR.rglob("costs.txt")))

            if cost_status == "timeout":
                detail_ph.markdown(
                    f'<span style="color:#E0A05C;font-size:0.8rem;">'
                    f'Timed out after {inactivity_min}min inactivity — {n_costs}/{total_dms} cost files so far. '
                    f'Click "Run Phase 2" again to resume.</span>',
                    unsafe_allow_html=True,
                )
            elif cost_status == "ok":
                detail_ph.markdown(
                    f'<span style="color:#4CAF82;font-size:0.8rem;">{n_costs} cost files generated</span>',
                    unsafe_allow_html=True,
                )
            else:
                detail_ph.markdown(
                    '<span style="color:#E05C5C;font-size:0.8rem;">Phase 2 failed — check log above</span>',
                    unsafe_allow_html=True,
                )

            # Parse cost files and snapshot (even on timeout — partial results are useful)
            if n_costs > 0:
                with st.spinner("Parsing cost files…"):
                    t_parse = time.perf_counter()
                    try:
                        _parse_df = build_costs_dataframe(OUTPUT_DMS_DIR)
                        t_parse_elapsed = time.perf_counter() - t_parse
                        st.caption(
                            f"Cost file parsing: {len(_parse_df)} rows from {n_costs} files "
                            f"in {t_parse_elapsed:.2f}s"
                        )
                    except Exception as _exc:
                        st.caption(f"Cost file parsing error: {_exc}")

                _run_dir = create_run_snapshot_from_live(base_model)
                if _run_dir is not None:
                    st.session_state["active_run_id"] = _run_dir.name
                    st.session_state["_pending_run_id"] = _run_dir.name
                    st.info(f"Saved run snapshot: {_run_dir.name}. This run is now active.")

        # ── Final summary ─────────────────────────────────────────────────
        all_ok = all(
            st.session_state.get(k) == "ok"
            for k in (["dmg_status"] if "phase1" in phases_to_run else []) +
                     (["cost_status"] if "phase2" in phases_to_run else [])
        )
        if all_ok:
            st.success("Pipeline completed successfully.")
        else:
            st.error("Pipeline finished with errors. Review the logs above.")

    st.divider()

    # ── Persistent logs ───────────────────────────────────────────────────
    if not (run_dmg or run_cost or run_all):
        if st.session_state["dmg_log"]:
            with st.expander(f'Phase 1 — Last Run  {_step_badge(st.session_state["dmg_status"])}', expanded=False):
                _t1, _t2 = st.tabs(["Live Log", "DM History"])
                with _t1:
                    st.markdown(f'<div class="log-box">{_format_log_html(st.session_state["dmg_log"])}</div>', unsafe_allow_html=True)
                with _t2:
                    _render_history(st.empty(), st.session_state.get("dmg_path_history", []))
        if st.session_state["cost_log"]:
            with st.expander(f'Phase 2 — Last Run  {_step_badge(st.session_state["cost_status"])}', expanded=False):
                _t1, _t2 = st.tabs(["Live Log", "DM History"])
                with _t1:
                    st.markdown(f'<div class="log-box">{_format_log_html(st.session_state["cost_log"])}</div>', unsafe_allow_html=True)
                with _t2:
                    _render_history(st.empty(), st.session_state.get("cost_path_history", []))

    # ── Reset ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-heading">Reset</div>', unsafe_allow_html=True)
    with st.expander("Fresh Start — clear all generated data"):
        st.markdown(
            "This will permanently delete:<br>"
            "&nbsp;&nbsp;— All generated data model XML files in <code>Output/GeneratedDMs/</code><br>"
            "&nbsp;&nbsp;— All run snapshots in <code>Output/runs/</code><br>"
            "&nbsp;&nbsp;— All signature files in <code>Output/</code> and <code>Input/</code><br>"
            "&nbsp;&nbsp;— All saved snapshots in <code>cost_history/</code><br>"
            "&nbsp;&nbsp;— Phase 1 and Phase 2 session logs<br>"
            "Your input files and base models in <code>Input/models/</code> are NOT affected.",
            unsafe_allow_html=True,
        )
        confirmed = st.checkbox("I understand this cannot be undone", key="reset_confirm")
        if st.button("Reset Everything", disabled=not confirmed, width="stretch"):
            if OUTPUT_DMS_DIR.exists():
                shutil.rmtree(OUTPUT_DMS_DIR)
            OUTPUT_DMS_DIR.mkdir(parents=True)
            for f in OUTPUT_DIR.glob("signatures_*.txt"):
                f.unlink(missing_ok=True)
            for f in INPUT_DIR.glob("signatures_*.txt"):
                f.unlink(missing_ok=True)
            if HISTORY_DIR.exists():
                shutil.rmtree(HISTORY_DIR)
            HISTORY_DIR.mkdir(parents=True)
            if RUNS_DIR.exists():
                shutil.rmtree(RUNS_DIR)
            RUNS_DIR.mkdir(parents=True)
            st.session_state["dmg_log"]    = ""
            st.session_state["cost_log"]   = ""
            st.session_state["dmg_status"] = ""
            st.session_state["cost_status"] = ""
            st.session_state["active_run_id"] = "LIVE"
            st.session_state["_pending_run_id"] = "LIVE"
            del st.session_state["reset_confirm"]
            st.cache_data.clear()
            st.session_state["data_bust"] = datetime.now().isoformat()
            st.success("Reset complete. All generated output has been cleared.")
            st.rerun()

    # ── Output Backup / Restore ──────────────────────────────────────────────
    st.divider()
    with st.expander("Output Backup / Restore", expanded=False):
        st.caption(
            "Backup mirrors `Output/GeneratedDMs/` and `Output/signatures_TPC_C.txt` "
            "into `Output/backup_original/`. "
            "Restore copies them back, overwriting the current output."
        )
        has_out_backup = output_backup_exists()
        ob1, ob2 = st.columns(2)
        if ob1.button("Save Output Backup", key="output_backup_save", width="stretch"):
            with st.spinner("Copying output files…"):
                n_copied = backup_output()
            st.success(f"Output backup saved ({n_copied} file(s) copied to `Output/backup_original/`).")
            st.rerun()
        if ob2.button(
            "Restore Output from Backup",
            key="output_backup_restore",
            disabled=not has_out_backup,
            width="stretch",
        ):
            with st.spinner("Restoring output files…"):
                n_restored = restore_output()
            st.cache_data.clear()
            st.session_state["data_bust"] = datetime.now().isoformat()
            st.success(f"Output restored ({n_restored} file(s) copied from backup).")
            st.rerun()
        if not has_out_backup:
            st.info("No output backup found in `Output/backup_original/`. Click **Save Output Backup** to create one.")
