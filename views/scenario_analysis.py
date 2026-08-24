"""Scenario Analysis — Generate diverse (vol, srv, freq) scenarios for ML training.

Tabs
----
1. Generate          create scenarios with per-query sampling distributions
2. Explore           what was generated: coverage, frequency stats, diversity
3. ML Export         flat training-data preview + CSV download
4. Build ML Dataset  full scenario sweep -> Parquet for the cost-evolution model
"""
from __future__ import annotations

import html as _html
import math
import random as _random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.constants import (
    ACCENT,
    BG_CARD,
    BG_PANEL,
    BORDER,
    METRIC_OPTIONS,
    PROJECT_ROOT,
    TEXT_MAIN,
    TEXT_MUTED,
)
from lib.data_loading import (
    active_run_query_mismatch,
    get_active_dms_dir,
    get_query_cost_stems,
    load_live_data,
    make_label_map,
    parse_query_frequencies,
)
from lib.scenarios import (
    apply_scenario_to_df,
    create_scenario,
    delete_scenario,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fv(v) -> str:
    """Compact volume label."""
    v = int(v)
    if v >= 1_000_000:
        return f"{v // 1_000_000}M"
    if v >= 1_000:
        return f"{v // 1_000}K"
    return str(v)


_SAMPLING_METHODS = ["Uniform", "Log-Uniform", "Gaussian", "Exponential", "Fixed"]

# Short descriptions auto-generated from queries.txt at module level;
# falls back to generic labels if the file is unreadable.
def _build_query_descs() -> list[str]:
    """Parse SELECT/WHERE from each query line for display labels."""
    from lib.constants import INPUT_DIR
    path = INPUT_DIR / "queries.txt"
    descs: list[str] = []
    if not path.exists():
        return descs
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("query/"):
            continue
        fields = line.split("/")[0].split(",")
        # Build compact description from query tokens
        sel_cols, tbl, where_cols, is_join = [], "", [], False
        mode = None
        for tok in fields:
            if tok == "select":
                mode = "sel"
            elif tok == "from":
                mode = "from"
            elif tok == "where":
                mode = "where"
            elif tok == "join":
                is_join = True
                mode = "skip"
            elif mode == "sel":
                sel_cols.append(tok)
            elif mode == "from" and not tbl:
                tbl = tok
            elif mode == "where":
                where_cols.append(tok)
        join_tag = " (join)" if is_join else ""
        desc = f"select {', '.join(sel_cols)} … where {', '.join(where_cols)}{join_tag}" if sel_cols else f"Q{len(descs)+1}"
        descs.append(desc)
    return descs

_Q_DESCS = _build_query_descs() or [f"Q{i+1}" for i in range(len(get_query_cost_stems()))]

_COST_LABELS = {
    "total_time_cost":          "Time (s)",
    "total_environmental_cost": "Env (kg CO₂e)",
    "total_financial_cost":     "Financial (€)",
}

# Color palette that scales to any number of queries
_Q_COLOR_PALETTE = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24


def _sample_freq(method: str, param_a: float, param_b: float, rng: _random.Random) -> int:
    """Sample one frequency value from the named distribution."""
    try:
        if method == "Uniform":
            lo, hi = int(min(param_a, param_b)), int(max(param_a, param_b))
            return rng.randint(max(1, lo), max(2, hi))
        if method == "Log-Uniform":
            lo_log = math.log10(max(1.0, float(param_a)))
            hi_log = math.log10(max(2.0, float(param_b)))
            return max(1, int(10 ** rng.uniform(min(lo_log, hi_log), max(lo_log, hi_log))))
        if method == "Gaussian":
            return max(1, int(rng.gauss(float(param_a), max(1.0, float(param_b)))))
        if method == "Exponential":
            return max(1, int(rng.expovariate(1.0 / max(1.0, float(param_a)))))
        return max(1, int(param_a))  # Fixed
    except Exception:
        return max(1, int(param_a))


def _pt_label(vol, srv) -> str:
    return f"{_fv(vol)} / {int(srv):,}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_full_df() -> pd.DataFrame:
    """Load the active run's cost data via the shared cache.

    ``load_live_data`` is keyed on the ``data_bust`` token and the active DMs
    directory, so switching runs/profiles or refreshing invalidates it — unlike
    a keyless local cache, which would serve stale data across run switches.
    """
    try:
        df, _ = load_live_data(
            st.session_state.get("data_bust", ""), str(get_active_dms_dir())
        )
        return df
    except Exception:
        return pd.DataFrame()


def _get_model_row(df_full, focus_xml, scenario):
    """Apply scenario, return the single row for the focus model (or None)."""
    df_s = apply_scenario_to_df(df_full, scenario)
    if df_s.empty:
        return None
    df_m = df_s[df_s["xml_file"] == focus_xml]
    if df_m.empty:
        return None
    return df_m.iloc[0]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render() -> None:
    mismatch, warn_msg = active_run_query_mismatch()
    if mismatch:
        st.warning(warn_msg)

    scenarios = st.session_state["scenarios"]
    df_full = _load_full_df()

    for col in METRIC_OPTIONS:
        if col in df_full.columns:
            df_full[col] = pd.to_numeric(df_full[col], errors="coerce")
    df_full = df_full.dropna(subset=METRIC_OPTIONS)

    if df_full.empty:
        st.warning("No cost data. Run **Pipeline → CostCalculation.jar** first.")
        return

    lmap = make_label_map(df_full["xml_file"])
    rlmap = {v: k for k, v in lmap.items()}
    all_labels = sorted(lmap.values())

    tabs = st.tabs(["Generate", "Explore", "ML Export", "Build ML Dataset"])
    with tabs[0]:
        _tab_generate(scenarios, df_full, lmap, rlmap, all_labels)
    with tabs[1]:
        _tab_explore(scenarios)
    with tabs[2]:
        _tab_ml_export(scenarios, df_full, rlmap)
    with tabs[3]:
        _tab_build_dataset()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: BUILD ML DATASET  (full scenario sweep -> Parquet, for the cost-evolution model)
# ══════════════════════════════════════════════════════════════════════════════

_SCEN_PARQUET = PROJECT_ROOT / "Output" / "scenario_ml.parquet"


def _run_cli(cmd: list[str], log_ph, status_key: str) -> int:
    """Stream a subprocess to a log placeholder; return exit code."""
    st.session_state[status_key] = "running"
    buf = ""
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if "streamlit" in line or "WARNING" in line:
                continue
            buf += line
            log_ph.markdown(f'<div class="log-box">{_html.escape(buf)}</div>', unsafe_allow_html=True)
        rc = proc.wait()
        st.session_state[status_key] = "ok" if rc == 0 else "error"
        return rc
    except Exception as e:  # noqa: BLE001
        log_ph.markdown(f'<div class="log-box">{_html.escape(buf)}\n[ERROR] {e}</div>',
                        unsafe_allow_html=True)
        st.session_state[status_key] = "error"
        return 1


def _parquet_stats(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return {"rows": pf.metadata.num_rows, "cols": pf.metadata.num_columns,
                "mb": path.stat().st_size / (1024 * 1024)}
    except Exception:  # noqa: BLE001
        return {"rows": "?", "cols": "?", "mb": path.stat().st_size / (1024 * 1024)}


def _tab_build_dataset() -> None:
    st.markdown(
        f'<div style="background:{BG_CARD};border:1px solid {BORDER};border-left:3px solid {ACCENT};'
        f'border-radius:6px;padding:12px 16px;margin-bottom:14px;">'
        f'<div style="font-size:0.82rem;color:{TEXT_MAIN};font-weight:600;">Build the cost-evolution training dataset</div>'
        f'<div style="font-size:0.73rem;color:{TEXT_MUTED};margin-top:4px;">'
        f'Reweights the parametric-sweep costs (<code>sweep_ml.parquet</code>) across M query-frequency mixes '
        f'into <code>scenario_ml.parquet</code> &mdash; the structural + temporal dataset for the predictor. '
        f'Pure Python (no JAR runs). M=420 reproduces &ldquo;420 scenarios per (profile, model)&rdquo; (~238M rows).'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    from scripts.dataset_io import default_sweep_dataset
    if not default_sweep_dataset().exists():
        st.warning("No assembled sweep dataset found. Build `Output/sweep_ml.parquet` first via "
                   "**Profile Sweep → Assemble**.")
        return

    c1, c2, c3 = st.columns([1, 1, 2])
    m = c1.number_input("Frequency mixes (M)", min_value=1, max_value=1000, value=420, step=10, key="scen_m")
    seed = c2.number_input("Seed", min_value=0, max_value=999999, value=42, step=1, key="scen_seed")
    with c3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        src_stats = _parquet_stats(default_sweep_dataset())
        if src_stats and isinstance(src_stats["rows"], int):
            n_struct = src_stats["rows"]
            st.caption(f"Will write **{n_struct * int(m):,}** rows ({n_struct:,} structural rows × "
                       f"{int(m)} mixes) to `Output/scenario_ml.parquet`.")
        else:
            st.caption(f"Will write (structural rows × {int(m)} mixes) to `Output/scenario_ml.parquet`.")

    bcol1, bcol2 = st.columns(2)
    gen = bcol1.button("Generate dataset", type="primary", key="scen_gen_btn")
    val = bcol2.button("Validate dataset", key="scen_val_btn")

    stats = _parquet_stats(_SCEN_PARQUET)
    if stats:
        st.caption(f"Current `scenario_ml.parquet`: **{stats['rows']:,} rows × {stats['cols']} cols · "
                   f"{stats['mb']:.0f} MB**" if isinstance(stats["rows"], int)
                   else f"Current `scenario_ml.parquet`: {stats['mb']:.0f} MB")

    log_ph = st.empty()
    if gen:
        cmd = [sys.executable, "-m", "scripts.scenario_sweep", "--m", str(int(m)), "--seed", str(int(seed))]
        with st.spinner(f"Reweighting across {int(m)} mixes — this may take a few minutes..."):
            rc = _run_cli(cmd, log_ph, "scen_gen_status")
        if rc == 0:
            st.success("Dataset written to `Output/scenario_ml.parquet`.")
            st.rerun()
        else:
            st.error("Generation failed — see log above.")
    if val:
        cmd = [sys.executable, "-m", "scripts.validate_scenario_dataset"]
        with st.spinner("Validating (streaming)..."):
            rc = _run_cli(cmd, log_ph, "scen_val_status")
        st.success("Validation complete — see log.") if rc == 0 else st.error("Validation reported failures.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: GENERATE
# ══════════════════════════════════════════════════════════════════════════════

def _tab_generate(scenarios, df_full, lmap, rlmap, all_labels):
    st.markdown(
        '<div class="section-heading">Scenario Generator</div>',
        unsafe_allow_html=True,
    )

    baseline = scenarios.get("baseline", {})
    all_vols = sorted(baseline.get("volumes", []))
    all_srvs = sorted(baseline.get("servers", []))
    if not all_vols or not all_srvs:
        st.warning("Baseline scenario missing volumes/servers.")
        return

    # Step 1: Focus model
    st.markdown("### Step 1 — Focus Model")
    default_idx = all_labels.index("M35") if "M35" in all_labels else 0
    focus_label = st.selectbox("Model", all_labels, index=default_idx, key="gen_focus")

    # Step 2: Grid
    st.markdown("### Step 2 — Infrastructure Grid")
    gc1, gc2 = st.columns(2)
    with gc1:
        sel_vols = st.multiselect("Volumes", all_vols, default=all_vols,
                                  format_func=_fv, key="gen_vols")
    with gc2:
        sel_srvs = st.multiselect("Servers", all_srvs, default=all_srvs,
                                  format_func=lambda x: f"{x:,}", key="gen_srvs")

    # Step 3: Per-query distributions
    st.markdown("### Step 3 — Query Frequency Distributions")
    st.caption(
        "Each query has its own distribution. "
        "**Param A / B**: Uniform (low, high) · Log-Uniform (min, max) · "
        "Gaussian (mean, std) · Exponential (mean, —) · Fixed (value, —)"
    )

    dfreqs = parse_query_frequencies()
    # Cycle through distribution types for default config rows
    _dist_cycle = ["Uniform", "Gaussian", "Log-Uniform", "Exponential", "Fixed"]
    cfg_rows = []
    for i, stem in enumerate(get_query_cost_stems()):
        freq = dfreqs.get(stem, 100)
        dist = _dist_cycle[i % len(_dist_cycle)]
        desc = _Q_DESCS[i] if i < len(_Q_DESCS) else f"Q{i+1}"
        if dist == "Uniform":
            pa, pb = float(max(1, freq // 10)), float(freq * 5)
        elif dist == "Gaussian":
            pa, pb = float(freq), float(freq * 0.4)
        elif dist == "Log-Uniform":
            pa, pb = float(max(1, freq // 5)), float(freq * 10)
        elif dist == "Exponential":
            pa, pb = float(freq), 0.0
        else:  # Fixed
            pa, pb = float(freq), 0.0
        cfg_rows.append({"Query": f"Q{i+1}", "Desc": desc, "Distribution": dist,
                         "Param A": pa, "Param B": pb})
    default_config = pd.DataFrame(cfg_rows)
    n_queries = len(cfg_rows)
    sampling_config = st.data_editor(
        default_config, num_rows="fixed", key="gen_sampling",
        width="stretch", height=min(230 + (n_queries - 5) * 25, 600),
        column_config={
            "Query":        st.column_config.TextColumn("Query", disabled=True, width="small"),
            "Desc":         st.column_config.TextColumn("Desc", disabled=True),
            "Distribution": st.column_config.SelectboxColumn(
                "Distribution", options=_SAMPLING_METHODS, required=True, width="medium"),
            "Param A": st.column_config.NumberColumn("Param A", min_value=0.0, step=10.0, width="small"),
            "Param B": st.column_config.NumberColumn("Param B", min_value=0.0, step=10.0, width="small"),
        },
        disabled=["Query", "Desc"],
    )

    # Step 4: Generate
    st.markdown("### Step 4 — Generate")
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        n_dists = st.number_input("Freq variants per grid point",
                                  min_value=1, max_value=200, value=10, step=5, key="gen_n")
    with gc2:
        seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="gen_seed")
    with gc3:
        preset = st.text_input("Preset name", value="mix", key="gen_preset").strip() or "mix"

    grid_size = len(sel_vols) * len(sel_srvs)
    total = grid_size * n_dists
    st.caption(f"**{total:,}** scenarios = {grid_size} grid points × {n_dists} freq variants")

    if not sel_vols or not sel_srvs:
        st.info("Select at least one volume and one server count.")
        return

    gen_count = sum(
        1 for sid, sd in scenarios.items()
        if sid != "baseline"
        and sd.get("focus_model_label") == focus_label
        and len(sd.get("volumes", [])) == 1
        and len(sd.get("servers", [])) == 1
    )

    bc1, bc2 = st.columns(2)
    with bc1:
        do_gen = st.button(f"Generate {total:,} Scenarios", disabled=total > 2000,
                           key="gen_btn", width="stretch")
    with bc2:
        if st.button(f"Clear Generated ({gen_count})", disabled=gen_count == 0,
                     key="gen_clear", width="stretch"):
            for sid in [s for s, d in scenarios.items()
                        if s != "baseline"
                        and d.get("focus_model_label") == focus_label
                        and len(d.get("volumes", [])) == 1
                        and len(d.get("servers", [])) == 1]:
                delete_scenario(sid)
            st.rerun()

    if total > 2000:
        st.warning("Too many scenarios (>2,000).")

    if do_gen:
        rng = _random.Random(int(seed))
        created = 0
        cfgs = sampling_config.to_dict("records")
        # Persist distribution config so Explore tab can display it
        st.session_state["last_gen_sampling"] = cfgs
        for di in range(n_dists):
            fsample: dict[str, int] = {}
            for cfg in cfgs:
                qi = cfg["Query"].lower()
                fsample[qi] = _sample_freq(str(cfg["Distribution"]),
                                           float(cfg["Param A"]), float(cfg["Param B"]), rng)
            fadj = {k: v for k, v in fsample.items() if v != dfreqs.get(k)}
            for v in sorted(sel_vols):
                for s in sorted(sel_srvs):
                    sc_id = f"{focus_label}_v{_fv(v)}_s{s}_{preset}_{di:03d}"[:52]
                    if sc_id in scenarios:
                        continue
                    create_scenario(
                        scenario_id=sc_id,
                        label=f"{focus_label} · {_fv(v)}/{s:,}/{preset}[{di:03d}]",
                        base_scenario=baseline,
                        modifications={
                            "description": f"model={focus_label}, vol={v}, srv={s}, freq={fsample}",
                            "focus_model_label": focus_label,
                            "volumes": [v], "servers": [s],
                            "query_freq_adjustments": fadj,
                            "freq_preset": f"{preset}_{di:03d}",
                        },
                    )
                    created += 1
        if created > 0:
            st.success(f"Created **{created:,}** scenarios.")
            st.rerun()
        else:
            st.warning("All scenarios already exist.")

    # Preview
    with st.expander("Preview sampled frequencies (first 10)", expanded=False):
        prng = _random.Random(int(seed))
        prows = []
        for i in range(min(10, n_dists)):
            row: dict = {"#": i}
            for cfg in sampling_config.to_dict("records"):
                row[cfg["Query"]] = _sample_freq(str(cfg["Distribution"]),
                                                 float(cfg["Param A"]), float(cfg["Param B"]), prng)
            prows.append(row)
        st.dataframe(pd.DataFrame(prows).set_index("#"), width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: EXPLORE
# ══════════════════════════════════════════════════════════════════════════════

def _tab_explore(scenarios):
    st.markdown(
        '<div class="section-heading">Explore Generated Scenarios</div>',
        unsafe_allow_html=True,
    )

    dfreqs = parse_query_frequencies()

    # Collect all generated single-point scenarios
    rows = []
    for sid, sd in scenarios.items():
        if sid == "baseline":
            continue
        vols = sd.get("volumes", [])
        srvs = sd.get("servers", [])
        if len(vols) != 1 or len(srvs) != 1:
            continue
        eff = {**dfreqs, **sd.get("query_freq_adjustments", {})}
        row_data = {
            "scenario_id": sid,
            "model": sd.get("focus_model_label", "?"),
            "volume": vols[0], "servers": srvs[0],
            "preset": sd.get("freq_preset", "custom"),
        }
        for stem in get_query_cost_stems():
            row_data[stem] = eff.get(stem, 0)
        rows.append(row_data)

    if not rows:
        st.info("No generated scenarios. Use the **Generate** tab.")
        return

    gdf = pd.DataFrame(rows)
    n_sc = len(gdf)
    n_pts = gdf.groupby(["volume", "servers"]).ngroups
    n_presets = gdf["preset"].nunique()

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Scenarios", f"{n_sc:,}")
    mc2.metric("Grid Points", n_pts)
    mc3.metric("Freq Variants", n_presets)

    # ── Frequency stats vs defaults ──────────────────────────────────────
    st.markdown("#### Sampled Frequency Ranges vs Defaults")
    st.caption(
        "Compares what was sampled against the original query frequencies from `queries.txt`. "
        "Wider range = more diversity for ML training."
    )

    # Pull distribution config saved when scenarios were last generated
    dist_map: dict[str, str] = {}
    gen_cfg = st.session_state.get("last_gen_sampling")
    if gen_cfg:
        try:
            for rec in gen_cfg:
                q_key = rec["Query"].lower()
                dist_map[q_key] = f"{rec['Distribution']} (A={rec['Param A']:.0f}, B={rec['Param B']:.0f})"
        except Exception:
            pass

    stat_rows = []
    for i, q in enumerate(get_query_cost_stems()):
        vals = gdf[q]
        row = {
            "Query": f"Q{i+1}",
            "Distribution Used": dist_map.get(q, "—"),
            "Default Freq": dfreqs[q],
            "Sampled Min": int(vals.min()),
            "Sampled Max": int(vals.max()),
            "Sampled Mean": round(vals.mean(), 0),
            "Sampled Std": round(vals.std(), 1),
            "Coverage": f"{vals.min()}–{vals.max()} ({vals.max()/max(vals.min(),1):.0f}× range)",
        }
        stat_rows.append(row)
    st.dataframe(pd.DataFrame(stat_rows), width="stretch", hide_index=True)

    # ── Per-query histograms ─────────────────────────────────────────────
    st.markdown("#### Frequency Distributions per Query")
    st.caption(
        "Each histogram shows how that query's frequency was sampled across all scenarios. "
        "The dashed line marks the default frequency."
    )

    n_q = len(get_query_cost_stems())
    cols_per_row = min(n_q, 6)
    for row_start in range(0, n_q, cols_per_row):
        chunk = get_query_cost_stems()[row_start:row_start + cols_per_row]
        hcols = st.columns(len(chunk))
        for j, q in enumerate(chunk):
            i = row_start + j
            with hcols[j]:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=gdf[q], nbinsx=20,
                    marker_color=_Q_COLOR_PALETTE[i % len(_Q_COLOR_PALETTE)], opacity=0.85,
                    name=f"Q{i+1}",
                ))
                fig.add_vline(x=dfreqs[q], line_dash="dash", line_color="#fff",
                              annotation_text=f"default={dfreqs[q]}",
                              annotation_font_color="#fff",
                              annotation_font_size=9)
                fig.update_layout(
                    title=dict(text=f"Q{i+1}", font=dict(size=13, color=TEXT_MAIN)),
                    paper_bgcolor=BG_PANEL, plot_bgcolor=BG_CARD, font_color=TEXT_MAIN,
                    height=220, margin=dict(t=35, b=30, l=30, r=10),
                    xaxis=dict(title=None), yaxis=dict(title=None),
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch")

    # ── Grid coverage table ──────────────────────────────────────────────
    st.markdown("#### Scenarios per Grid Point")
    cov = gdf.groupby(["volume", "servers"]).size().reset_index(name="count")
    cov_pivot = cov.pivot(index="servers", columns="volume", values="count").fillna(0).astype(int)
    cov_pivot.columns = [_fv(c) for c in cov_pivot.columns]
    cov_pivot.index = [f"{int(s):,}" for s in cov_pivot.index]
    cov_pivot.index.name = "Servers \\ Volume"
    st.dataframe(cov_pivot, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ML EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def _tab_ml_export(scenarios, df_full, rlmap):
    st.markdown(
        '<div class="section-heading">ML Training Data Export</div>',
        unsafe_allow_html=True,
    )
    n_q = len(get_query_cost_stems())
    st.caption(
        f"Each row = one scenario: a specific (volume, servers, q1_freq … q{n_q}_freq) configuration "
        "with the focus model's resulting costs. Features are the inputs an ML model sees; "
        "targets are what it predicts."
    )

    dfreqs = parse_query_frequencies()
    d_total = sum(dfreqs.values()) or 1

    ml_rows = []
    for sid, sd in scenarios.items():
        if sid == "baseline":
            continue
        vols = sd.get("volumes", [])
        srvs = sd.get("servers", [])
        if len(vols) != 1 or len(srvs) != 1:
            continue
        focus_label = sd.get("focus_model_label", "")
        focus_xml = rlmap.get(focus_label, "")
        if not focus_xml:
            continue

        r = _get_model_row(df_full, focus_xml, sd)
        if r is None:
            continue

        eff = {**dfreqs, **sd.get("query_freq_adjustments", {})}
        eff_total = sum(eff.values()) or 1

        row = {
            "scenario_id": sid,
            "model": focus_label,
            # Infrastructure features
            "volume": vols[0],
            "servers": srvs[0],
        }
        # Frequency features — dynamic for any number of queries
        for stem in get_query_cost_stems():
            row[f"freq_{stem}"] = eff.get(stem, 0)
        for stem in get_query_cost_stems():
            row[f"freq_{stem}_ratio"] = round(eff.get(stem, 0) / eff_total, 4)
        row["freq_total"] = eff_total
        row["freq_total_vs_base"] = round(eff_total / d_total, 4)

        # Per-query unit costs (fixed per operating point — important features)
        for stem in get_query_cost_stems():
            for sfx in ["_time_cost", "_env_cost", "_fin_cost"]:
                col = f"{stem}{sfx}"
                if col in r.index:
                    row[col] = r[col]

        # Target costs
        for m in METRIC_OPTIONS:
            if m in r.index:
                row[m] = r[m]

        ml_rows.append(row)

    if not ml_rows:
        st.info("No scenarios to export. Use the **Generate** tab.")
        return

    ml_df = pd.DataFrame(ml_rows).sort_values(["model", "volume", "servers", "scenario_id"])

    # ── Summary ──────────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Rows", f"{len(ml_df):,}")
    mc2.metric("Models", ml_df["model"].nunique())
    n_feat = len([c for c in ml_df.columns
                  if c.startswith("freq_") or c in ["volume", "servers"]
                  or (c.startswith("q") and "_cost" in c)])
    mc3.metric("Features", n_feat)
    mc4.metric("Targets", len(METRIC_OPTIONS))

    # ── Column layout ────────────────────────────────────────────────────
    st.markdown("#### Column Layout")
    schema = []
    for col in ml_df.columns:
        if col in ["scenario_id", "model"]:
            role = "ID"
        elif col in ["volume", "servers"]:
            role = "Feature — infrastructure"
        elif col.startswith("freq_q") and col.endswith("_ratio"):
            role = "Feature — freq ratio"
        elif col.startswith("freq_q"):
            role = "Feature — freq absolute"
        elif col.startswith("freq_"):
            role = "Feature — freq meta"
        elif col.startswith("q") and "_cost" in col:
            role = "Feature — unit cost"
        elif col in METRIC_OPTIONS:
            role = "Target"
        else:
            role = "Other"
        schema.append({"Column": col, "Role": role, "Type": str(ml_df[col].dtype)})
    st.dataframe(pd.DataFrame(schema), width="stretch", hide_index=True, height=350)

    # ── Preview ──────────────────────────────────────────────────────────
    st.markdown("#### Data Preview (first 30 rows)")
    st.dataframe(ml_df.head(30), width="stretch", hide_index=True, height=400)

    # ── Download ─────────────────────────────────────────────────────────
    csv = ml_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"Download CSV ({len(ml_df):,} rows)",
        data=csv,
        file_name=f"ml_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        width="stretch",
    )
