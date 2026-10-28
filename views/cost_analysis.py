"""Cost Analysis page — 5 tabs that scale to 100+ data models."""
from __future__ import annotations

import math
import random
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.chart_helpers import apply_log_axis_readability
from lib.constants import (
    ACCENT,
    BG_CARD,
    BG_PANEL,
    BORDER,
    GREEN,
    LAYOUT_BASE,
    LEGEND_BASE,
    METRIC_OPTIONS,
    AXIS_OPTIONS,
    PLOTLY_COLORS,
    RED,
    TEXT_MAIN,
    TEXT_MUTED,
    YELLOW,
    fmt_metric,
    fmt_volume,
)
from lib.data_loading import (
    active_run_query_mismatch,
    detect_active_profile,
    get_active_dms_dir,
    get_active_run_id,
    list_run_ids,
    load_live_data,
    load_snapshot,
    list_snapshots,
    read_run_metadata,
    save_snapshot,
    make_label_map,
)
from lib.equations import (
    eq14_normalize_metrics,
    eq14_prepare_scored_frame,
    eq17_build_stability_scores,
)
from lib.pareto import (
    build_tree_dataframe,
    chain_to_root,
    compute_pareto_mask,
)
from lib.ui_helpers import persistent_tabs


@st.cache_data(show_spinner=False)
def _stability_ranking(
    df: pd.DataFrame,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    avg_col: str = "avg",
    count_col: str | None = None,
) -> pd.DataFrame:
    """Eq. 14 scoring + Eq. 17 stability ranking, cached on the input frame.

    Overview, the leaderboard fragment and Compare all need this ranking; caching
    it means the normalisation/scoring pass runs once per (data, weights) instead
    of once per call site."""
    df_scored = eq14_prepare_scored_frame(df, weights=weights)
    return eq17_build_stability_scores(
        df_scored, score_col="C_score", avg_col=avg_col, count_col=count_col,
    )


def render() -> None:
    # ── Data run selector (sidebar) ───────────────────────────────────────
    # Each run is labelled by the profile it was generated under, so switching
    # run here is effectively switching which profile's cost data is shown.
    # Runs before the selector so get_active_dms_dir() below sees the choice.
    def _run_label(r: str) -> str:
        if r == "LIVE":
            p = detect_active_profile()
            return f"Live · Profile {p}" if p else "Live · Custom Input/"
        meta = read_run_metadata(r) or {}
        p = meta.get("profile")
        return f"Run {r} · Profile {p}" if p else f"Run {r} · profile n/a"

    _run_opts = ["LIVE", *list_run_ids()]
    _cur_run = get_active_run_id()
    if _cur_run not in _run_opts:
        _cur_run = "LIVE"
    st.session_state["ca_active_run"] = _cur_run  # keep widget in sync with global state
    st.sidebar.markdown('<div class="sidebar-section">&#128202; Data Run</div>', unsafe_allow_html=True)
    st.session_state["active_run_id"] = st.sidebar.selectbox(
        "Data run", options=_run_opts, format_func=_run_label,
        label_visibility="collapsed", key="ca_active_run",
    )

    # Load data
    _active_dir = get_active_dms_dir()
    _active_run_id = get_active_run_id()
    _source_label = "Live" if _active_run_id == "LIVE" else f"Run {_active_run_id}"
    df_live, loaded_at = load_live_data(st.session_state["data_bust"], str(_active_dir), _source_label)

    _is_mismatch, _warn_msg = active_run_query_mismatch()
    if _is_mismatch:
        st.warning(_warn_msg)

    # Snapshot selector in sidebar
    st.sidebar.markdown('<div class="sidebar-section">&#128190; Snapshots</div>', unsafe_allow_html=True)
    available_snaps = list_snapshots()
    snap_dict: dict[str, str] = {p.stem: str(p) for p in available_snaps}
    selected_snap_names: list[str] = []

    if available_snaps:
        selected_snap_names = st.sidebar.multiselect(
            "Snapshots", options=list(snap_dict.keys()), default=[],
            label_visibility="collapsed",
        )
    else:
        st.sidebar.caption("No snapshots saved yet.")

    # Refresh & save
    st.sidebar.markdown('<div class="sidebar-section">&#9881;&#65039; Actions</div>', unsafe_allow_html=True)
    if st.sidebar.button("&#8635;  Refresh Live Data", width="stretch"):
        st.cache_data.clear()
        st.session_state["data_bust"] = datetime.now().isoformat()
        st.rerun()
    if loaded_at:
        st.sidebar.caption(f"Last loaded: {loaded_at}")

    snap_label = st.sidebar.text_input("Snapshot Label", placeholder="e.g. baseline_v1", label_visibility="collapsed")
    if st.sidebar.button("&#128190;  Save Snapshot", width="stretch"):
        if not df_live.empty:
            dest = save_snapshot(df_live, snap_label or "snapshot")
            load_snapshot.clear()
            st.sidebar.success(f"Saved: {dest.name}")
        else:
            st.sidebar.warning("No live data to save.")

    # ── Scenario Mode ─────────────────────────────────────────────────────
    st.sidebar.markdown('<div class="sidebar-section">&#128300; Scenario Mode</div>', unsafe_allow_html=True)
    _sc_mode = st.sidebar.radio(
        "Scenario view",
        ["Baseline only", "Compare scenarios"],
        horizontal=True,
        label_visibility="collapsed",
        key="ca_sc_mode",
    )
    _ca_selected_scns: list[str] = ["baseline"]
    if _sc_mode == "Compare scenarios":
        _sc_opts = {
            sid: sdata["label"]
            for sid, sdata in st.session_state["scenarios"].items()
        }
        _ca_selected_scns = st.sidebar.multiselect(
            "Scenarios to compare",
            options=list(_sc_opts.keys()),
            default=list(_sc_opts.keys()),
            format_func=lambda x: _sc_opts[x],
            label_visibility="collapsed",
            key="ca_sc_sel",
        ) or ["baseline"]
    _sc_compare_active = (_sc_mode == "Compare scenarios") and len(_ca_selected_scns) > 0

    if df_live.empty:
        st.info("No cost results found for the selected run. Run the pipeline first or switch run in Pipeline tab.")
        st.stop()

    # Merge live + snapshots
    frames = [df_live] + [load_snapshot(snap_dict[n]) for n in selected_snap_names]
    df_all = pd.concat(frames, ignore_index=True) if len(frames) > 1 else df_live.copy()

    # Build short M-labels
    _label_map     = make_label_map(df_all["xml_file"])
    _reverse_label = {v: k for k, v in _label_map.items()}
    df_all["model_full_path"] = df_all["xml_file"]
    df_all["xml_file"]        = df_all["xml_file"].map(_label_map).fillna(df_all["xml_file"])

    # Per-scenario filtered dfs
    _ca_sc_dfs: dict[str, pd.DataFrame] = {}
    if _sc_compare_active:
        for _sc_cid in _ca_selected_scns:
            _sc_cfg = st.session_state["scenarios"].get(_sc_cid, {})
            _df_sc  = df_all.copy()
            if _sc_cfg.get("volumes"):
                _sc_vols = [int(v) for v in _sc_cfg["volumes"]]
                _df_sc = _df_sc[_df_sc["data_volume"].isin(_sc_vols)]
            if _sc_cfg.get("servers"):
                _sc_srvs = [int(s) for s in _sc_cfg["servers"]]
                _df_sc = _df_sc[_df_sc["servers"].isin(_sc_srvs)]
            if not _df_sc.empty:
                _ca_sc_dfs[_sc_cid] = _df_sc

    # Filters (top-level filter set; tabs apply their own narrowing controls)
    st.sidebar.markdown('<div class="sidebar-section">&#127968; Filters</div>', unsafe_allow_html=True)
    all_models   = sorted(df_all["xml_file"].dropna().unique().tolist(), key=lambda s: int(s[1:]) if s[1:].isdigit() else -1)
    sel_models   = st.sidebar.multiselect(
        "Models", all_models, all_models,
        format_func=lambda x: f"{x}  –  {_reverse_label.get(x, x)}",
        label_visibility="collapsed",
        key="ca_filt_models",
    )
    all_volumes  = [int(v) for v in sorted(df_all["data_volume"].dropna().unique())]
    sel_volumes  = st.sidebar.multiselect("Volumes", all_volumes, all_volumes, format_func=fmt_volume, label_visibility="collapsed", key="ca_filt_volumes")
    all_modes    = sorted(df_all["mode"].dropna().unique().tolist())
    sel_modes    = st.sidebar.multiselect("Modes",   all_modes,   all_modes,   format_func=lambda x: f" {x}", label_visibility="collapsed", key="ca_filt_modes")
    all_sources  = sorted(df_all["source"].unique().tolist())
    sel_sources  = st.sidebar.multiselect("Sources", all_sources, all_sources, format_func=lambda x: f" {x}", label_visibility="collapsed", key="ca_filt_sources")
    sv_min, sv_max = int(df_all["servers"].min()), int(df_all["servers"].max())
    sv_range = st.sidebar.slider("Servers", sv_min, sv_max, (sv_min, sv_max), label_visibility="collapsed", key="ca_filt_servers")

    with st.sidebar.expander("Model Reference", expanded=False):
        _ref_md = "  \n".join(
            f"**{lbl}** — `{p_}`"
            for p_, lbl in sorted(_label_map.items(), key=lambda kv: int(kv[1][1:]) if kv[1][1:].isdigit() else -1)
        )
        st.markdown(_ref_md)

    df = df_all[
        df_all["xml_file"].isin(sel_models)
        & df_all["data_volume"].isin(sel_volumes)
        & df_all["mode"].isin(sel_modes)
        & df_all["source"].isin(sel_sources)
        & df_all["servers"].between(sv_range[0], sv_range[1])
    ].copy()

    # Profile this output was generated under: stamped into run metadata at
    # pipeline time for saved runs; for LIVE, inferred from the current Input/.
    if _active_run_id == "LIVE":
        _gen_profile = detect_active_profile()
        _profile_txt = (
            f"Profile {_gen_profile} (current Input/)" if _gen_profile
            else "Custom Input/ (no matching profile)"
        )
    else:
        _meta = read_run_metadata(_active_run_id) or {}
        _gen_profile = _meta.get("profile")
        _profile_txt = (
            f"Profile {_gen_profile}" if _gen_profile
            else "Profile not recorded (run predates profile tracking)"
        )

    st.caption(
        f"{len(df):,} records &nbsp;|&nbsp; {df['xml_file'].nunique()} models &nbsp;|&nbsp; "
        f"{df['source'].nunique()} source(s) &nbsp;|&nbsp; Generated under: {_profile_txt} "
        f"&nbsp;|&nbsp; Last refresh: {loaded_at}",
    )

    if df.empty:
        st.warning("No records match current filters.")
        st.stop()

    # Top KPI strip
    _model_med = df.groupby("xml_file")[METRIC_OPTIONS].median()
    _n_models  = df["xml_file"].nunique()
    _vols_per_model   = df.groupby("xml_file")["data_volume"].apply(set)
    _complete_volumes = set.intersection(*_vols_per_model) if len(_vols_per_model) else set()
    _srvs_per_model   = df.groupby("xml_file")["servers"].apply(set)
    _complete_servers = set.intersection(*_srvs_per_model) if len(_srvs_per_model) else set()
    _complete_configs = len(_complete_volumes) * len(_complete_servers) * _n_models

    _best_kpi  = _model_med["total_time_cost"].idxmin()
    _best_time = float(_model_med.loc[_best_kpi, "total_time_cost"])

    _base_label = "M0"
    if _base_label in _model_med.index:
        _base_time    = float(_model_med.loc[_base_label, "total_time_cost"])
        _better_count = int((_model_med["total_time_cost"] < _base_time).sum())
    else:
        _better_count = 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Models", f"{_n_models:,}", help="Distinct data models in the current filter")
    k2.metric("Configurations", f"{_complete_configs:,}", help="Models x shared volumes x server counts")
    k3.metric("Best (Time)", _best_kpi, help=f"Lowest median time cost: {_best_time:,.2f}s")
    k4.metric("Beating Baseline", f"{_better_count} / {_n_models}", help=f"Models with median time < {_base_label}")

    st.divider()

    _COST_TABS = ["Overview", "Trade-offs", "Compare", "Transformation Tree", "Data Table"]
    _cost_tab = persistent_tabs(_COST_TABS, "cost_tab")

    if _cost_tab == "Overview":
        _render_overview(df, _sc_compare_active, _ca_sc_dfs)
    elif _cost_tab == "Trade-offs":
        _render_tradeoffs(df)
    elif _cost_tab == "Compare":
        _render_compare(df)
    elif _cost_tab == "Transformation Tree":
        _render_tree(df)
    elif _cost_tab == "Data Table":
        _render_data_table(df)


@st.fragment
def _render_leaderboard(df: pd.DataFrame) -> None:
    sw1, sw2, sw3, _ = st.columns([1, 1, 1, 2])
    wT = sw1.slider("Time", 0.0, 1.0, 1/3, 0.05, key="ov_wt")
    wE = sw2.slider("Env",  0.0, 1.0, 1/3, 0.05, key="ov_we")
    wF = sw3.slider("Fin",  0.0, 1.0, 1/3, 0.05, key="ov_wf")
    w_total = wT + wE + wF
    if w_total < 1e-9:
        wT = wE = wF = 1 / 3
        w_total = 1.0
    wT /= w_total; wE /= w_total; wF /= w_total
    st.caption(f"Effective weights — Time: {wT:.0%} · Env: {wE:.0%} · Fin: {wF:.0%}")

    df_rank = _stability_ranking(df, weights=(wT, wE, wF), avg_col="Avg Score", count_col="# Configs")
    if df_rank.empty:
        st.info("No models to rank.")
        return
    df_rank_top = df_rank.head(10).copy()
    df_rank_top.index = range(1, len(df_rank_top) + 1)
    df_rank_top = df_rank_top.rename(columns={
        "xml_file": "Model", "delta": "Stability", "d_tilde": "D̃ (lower=better)",
    })
    fig_rank = px.bar(
        df_rank_top, x="Model", y="D̃ (lower=better)",
        color="D̃ (lower=better)",
        color_continuous_scale=[[0, GREEN], [0.5, YELLOW], [1.0, RED]],
    )
    fig_rank.update_layout(height=300, coloraxis_showscale=False, **LAYOUT_BASE)
    fig_rank.update_yaxes(range=[0, 1])
    #st.plotly_chart(fig_rank, width="stretch")
    st.plotly_chart(fig_rank)
    st.dataframe(df_rank_top.round(5), width="stretch", height=min(400, 40 + 36 * len(df_rank_top)))
    with st.expander(f"Show full ranking ({len(df_rank)} models)", expanded=False):
        df_full = df_rank.rename(columns={
            "xml_file": "Model", "delta": "Stability", "d_tilde": "D̃",
        }).round(5)
        df_full.index = range(1, len(df_full) + 1)
        st.dataframe(df_full, width="stretch", height=min(600, 40 + 36 * len(df_full)))


# ═══════════════════════════════════════════════════════════════════════════
# Tab 1: Overview — best models + distributions + stability leaderboard
# ═══════════════════════════════════════════════════════════════════════════
@st.fragment
def _render_overview(df: pd.DataFrame, _sc_compare_active: bool, _ca_sc_dfs: dict[str, pd.DataFrame]) -> None:
    st.markdown(
        '<div class="section-heading">Best Models</div>',
        unsafe_allow_html=True,
    )

    rec_agg = df.groupby("xml_file", as_index=False)[METRIC_OPTIONS].mean()
    _fp_map = df[["xml_file", "model_full_path"]].drop_duplicates("xml_file") if "model_full_path" in df.columns else None
    if _fp_map is not None:
        rec_agg = rec_agg.merge(_fp_map, on="xml_file", how="left")

    best_time = rec_agg.loc[rec_agg["total_time_cost"].idxmin(), "xml_file"]
    best_fin  = rec_agg.loc[rec_agg["total_financial_cost"].idxmin(), "xml_file"]
    best_env  = rec_agg.loc[rec_agg["total_environmental_cost"].idxmin(), "xml_file"]

    df_stab = _stability_ranking(df)
    best_overall = df_stab.loc[0, "xml_file"] if not df_stab.empty else best_time

    def _val(model: str, metric: str) -> float:
        row = rec_agg[rec_agg["xml_file"] == model]
        return float(row[metric].iloc[0]) if not row.empty else float("nan")

    def _full(model: str) -> str:
        if _fp_map is None:
            return ""
        m = _fp_map[_fp_map["xml_file"] == model]
        return str(m["model_full_path"].iloc[0]) if not m.empty else ""

    st.markdown(
        f"""<style>
        .ov-card {{
            background:{BG_CARD}; border:1px solid {BORDER}; border-radius:10px;
            padding:16px 18px; margin-bottom:6px;
        }}
        .ov-card.balanced {{ border-left:4px solid {ACCENT}; border-top:1px solid {ACCENT}; }}
        .ov-card.gold     {{ border-left:4px solid #F5C518; }}
        .ov-card.silver   {{ border-left:4px solid #A8A9AD; }}
        .ov-card.bronze   {{ border-left:4px solid #CD7F32; }}
        .ov-title {{
            font-size:0.70rem; font-weight:700; text-transform:uppercase;
            letter-spacing:0.10em; color:{TEXT_MAIN}; margin-bottom:4px;
        }}
        .ov-model {{ font-size:1.10rem; font-weight:700; color:{TEXT_MAIN}; margin-bottom:4px; }}
        .ov-value {{ font-size:0.80rem; color:{TEXT_MAIN}; }}
        .ov-path  {{ font-size:0.66rem; color:{TEXT_MAIN}; font-family:monospace; margin-top:6px; word-break:break-all; }}
        </style>""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "balanced", "Balanced (D&#771; best)", best_overall),
        (c2, "gold", "Best for Speed", best_time),
        (c3, "silver", "Best for Finance", best_fin),
        (c4, "bronze", "Best for Environment", best_env),
    ]
    for col, cls, title, model in cards:
        col.markdown(
            f"""<div class="ov-card {cls}">
            <div class="ov-title">{title}</div>
            <div class="ov-model">{model}</div>
            <div class="ov-value">
                T: <b>{_val(model, 'total_time_cost'):.3g}s</b> &nbsp;|&nbsp;
                F: <b>${_val(model, 'total_financial_cost'):,.0f}</b> &nbsp;|&nbsp;
                E: <b>{_val(model, 'total_environmental_cost'):.3g}</b>
            </div>
            <div class="ov-path">{_full(model)}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Cost distribution strip plots ───────────────────────────────────
    st.markdown(
        '<div class="section-heading">Cost Distribution Across All Models <span class="badge-raw">&#128202; Raw · log-x</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("Each dot is one model. Star markers = best-of-each. Hover to identify any dot.")

    h1, h2, h3 = st.columns(3)
    highlights = [
        (best_overall, ACCENT,    "Balanced (D̃)"),
        (best_time,    "#F5C518", "Best Time"),
        (best_fin,     "#A8A9AD", "Best Finance"),
        (best_env,     "#CD7F32", "Best Env"),
    ]
    rng = random.Random(42)
    jitter = [rng.uniform(0.1, 0.9) for _ in range(len(rec_agg))]
    label_to_jitter = dict(zip(rec_agg["xml_file"].tolist(), jitter))

    for col, metric, title, color in [
        (h1, "total_time_cost", "Time Cost", "#4A7FC1"),
        (h2, "total_financial_cost", "Financial Cost", "#3DAA6E"),
        (h3, "total_environmental_cost", "Environmental Cost", "#C89820"),
    ]:
        plot_df = rec_agg.copy()
        plot_df = plot_df[pd.to_numeric(plot_df[metric], errors="coerce") > 0]
        if plot_df.empty:
            continue

        plot_df = plot_df.assign(_y=plot_df["xml_file"].map(label_to_jitter))
        fp_safe = plot_df.get("model_full_path", pd.Series([""] * len(plot_df))).fillna("").astype(str)

        fig = go.Figure()
        # All models
        fig.add_trace(go.Scatter(
            x=plot_df[metric], y=plot_df["_y"],
            mode="markers",
            marker=dict(size=8, color=color, opacity=0.55, line=dict(width=0)),
            customdata=list(zip(plot_df["xml_file"].tolist(), fp_safe.tolist())),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{fmt_metric(metric)}: %{{x:,.4g}}<br>"
                "<span style='font-size:10px'>%{customdata[1]}</span><extra></extra>"
            ),
            showlegend=False,
        ))
        # Highlight best-of-each as labeled stars (all 4 best models on every chart)
        for hm, hc, hname in highlights:
            v = _val(hm, metric)
            if pd.notna(v) and v > 0 and hm in label_to_jitter:
                fig.add_trace(go.Scatter(
                    x=[v], y=[label_to_jitter[hm]],
                    mode="markers+text",
                    marker=dict(size=16, color=hc, line=dict(width=1.5, color="#fff"), symbol="star"),
                    text=[f"{hname[:8]}: {hm}"],
                    textposition="top center",
                    textfont=dict(size=9, color=hc),
                    name=f"{hname} — {hm}",
                    hovertemplate=f"<b>{hname}</b><br>{hm}<br>{fmt_metric(metric)}: %{{x:,.4g}}<extra></extra>",
                    showlegend=False,
                ))

        fig.update_layout(
            height=200,
            autosize=True,
            title=dict(text=title, font=dict(size=11, color=TEXT_MUTED)),
            margin=dict(t=30, b=40, l=20, r=10),
            **{k: v for k, v in LAYOUT_BASE.items() if k not in ("margin", "autosize")},
            showlegend=False,
        )
        fig.update_xaxes(title_text=fmt_metric(metric), type="log")
        fig.update_yaxes(visible=False, range=[0.05, 0.95])
        col.plotly_chart(fig, key=f"ov_strip_{metric}")

    st.divider()

    # ── Stability leaderboard ──────────────────────────────────────────
    st.markdown(
        '<div class="section-heading">Stability Leaderboard '
        '<span class="badge-norm">&#128208; Normalized Eq. 14&ndash;17</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("Top 10 models by D̃ — combines low cost AND stability across (volume, servers).")
    _render_leaderboard(df)

    # ── Scenario summary ──────────────────────────────────────────────
    if _sc_compare_active and len(_ca_sc_dfs) >= 2:
        st.divider()
        st.markdown(
            '<div class="section-heading">Best Model per Scenario</div>',
            unsafe_allow_html=True,
        )
        _sc_metric = st.selectbox("Metric", METRIC_OPTIONS, format_func=fmt_metric, key="ov_sc_m")
        _cols = st.columns(min(len(_ca_sc_dfs), 4))
        for i, (sid, df_sc) in enumerate(_ca_sc_dfs.items()):
            label = st.session_state["scenarios"].get(sid, {}).get("label", sid)
            agg = df_sc.groupby("xml_file", as_index=False)[_sc_metric].mean()
            if agg.empty:
                continue
            best = agg.loc[agg[_sc_metric].idxmin(), "xml_file"]
            best_v = float(agg.loc[agg[_sc_metric].idxmin(), _sc_metric])
            with _cols[i % len(_cols)]:
                st.markdown(
                    f"""<div class="ov-card balanced" style="text-align:center">
                    <div class="ov-title">{label}</div>
                    <div class="ov-model">{best}</div>
                    <div class="ov-value" style="color:{GREEN}">{best_v:.4g}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════
# Tab 2: Trade-offs — Pareto frontier
# ═══════════════════════════════════════════════════════════════════════════
@st.fragment
def _render_tradeoffs(df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-heading">Pareto Trade-off Frontier '
        '<span class="badge-raw">&#128202; Raw · per-model means</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Each point = one model. Points on the Pareto frontier (highlighted) are non-dominated — "
        "no other model has both lower X and lower Y. Cuts the candidate set from many to few."
    )

    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 1, 1])
    xa = c1.selectbox("X axis", AXIS_OPTIONS, index=2, format_func=fmt_metric, key="po_x")
    ya = c2.selectbox("Y axis", AXIS_OPTIONS, index=0, format_func=fmt_metric, key="po_y")
    color_by = c3.selectbox("Color by", ["(none)"] + AXIS_OPTIONS, index=0, format_func=lambda v: "—" if v == "(none)" else fmt_metric(v), key="po_c")
    show_only_pareto = c4.checkbox("Pareto only", value=False, key="po_only")
    log_axes = c5.checkbox("Log axes", value=True, key="po_log")

    if xa == ya:
        st.warning("Choose two different axes.")
        return

    # Setting filters so the user sees what (volume, servers) this view represents
    s1, s2 = st.columns(2)
    vol_opts = sorted(df["data_volume"].dropna().unique().astype(int).tolist())
    srv_opts = sorted(df["servers"].dropna().unique().astype(int).tolist())
    sel_vol = s1.selectbox(
        "Data volume", ["All (averaged)"] + vol_opts,
        format_func=lambda v: v if v == "All (averaged)" else fmt_volume(v),
        key="po_vol",
    )
    sel_srv = s2.selectbox("Servers", ["All (averaged)"] + srv_opts, key="po_srv")

    df_filt = df.copy()
    if sel_vol != "All (averaged)":
        df_filt = df_filt[df_filt["data_volume"] == sel_vol]
    if sel_srv != "All (averaged)":
        df_filt = df_filt[df_filt["servers"] == sel_srv]
    if df_filt.empty:
        st.info("No data at this (volume, servers) combination.")
        return

    _vol_disp = "all volumes" if sel_vol == "All (averaged)" else fmt_volume(sel_vol)
    _srv_disp = "all server counts" if sel_srv == "All (averaged)" else f"{sel_srv} servers"
    st.caption(f"Showing models at **{_vol_disp}** · **{_srv_disp}** — each point is one model.")

    agg = df_filt.groupby("xml_file", as_index=False)[AXIS_OPTIONS].mean()
    fp_map = df[["xml_file", "model_full_path"]].drop_duplicates("xml_file") if "model_full_path" in df.columns else None
    if fp_map is not None:
        agg = agg.merge(fp_map, on="xml_file", how="left")

    if agg.empty:
        st.info("No models to plot.")
        return

    pareto = compute_pareto_mask(agg[xa], agg[ya])
    agg = agg.assign(is_pareto=pareto.values)

    fig = go.Figure()
    # Determine shared color array (color_by applies to BOTH layers when chosen)
    use_color_axis = color_by != "(none)"

    # Dominated points (back layer, dim, smaller)
    if not show_only_pareto:
        dom = agg[~agg["is_pareto"]]
        if not dom.empty:
            dom_marker = dict(size=8, opacity=0.5, line=dict(width=0))
            if use_color_axis:
                dom_marker.update(
                    color=dom[color_by], colorscale="Viridis",
                    cmin=float(agg[color_by].min()), cmax=float(agg[color_by].max()),
                    showscale=False,
                )
            else:
                dom_marker.update(color=TEXT_MUTED)
            fig.add_trace(go.Scatter(
                x=dom[xa], y=dom[ya],
                mode="markers",
                marker=dom_marker,
                name="Dominated",
                customdata=list(zip(dom["xml_file"].tolist(), dom["model_full_path"].fillna("").astype(str).tolist())),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{fmt_metric(xa)}: %{{x:,.4g}}<br>"
                    f"{fmt_metric(ya)}: %{{y:,.4g}}<br>"
                    "<span style='font-size:10px'>%{customdata[1]}</span><extra>Dominated</extra>"
                ),
            ))

    # Pareto points (front layer, larger + outlined)
    par = agg[agg["is_pareto"]].sort_values(xa)
    par_marker = dict(size=16, line=dict(width=2, color="#fff"))
    if use_color_axis:
        par_marker.update(
            color=par[color_by], colorscale="Viridis",
            cmin=float(agg[color_by].min()), cmax=float(agg[color_by].max()),
            showscale=True,
            colorbar=dict(title=fmt_metric(color_by), thickness=12),
        )
    else:
        par_marker.update(color=ACCENT)

    # Only label Pareto if their count is small enough to read (<= 25)
    show_par_text = len(par) <= 25
    fig.add_trace(go.Scatter(
        x=par[xa], y=par[ya],
        mode="markers+text" if show_par_text else "markers",
        marker=par_marker,
        text=par["xml_file"] if show_par_text else None,
        textposition="top center",
        textfont=dict(size=10, color=TEXT_MAIN),
        name="Pareto-optimal",
        customdata=list(zip(par["xml_file"].tolist(), par["model_full_path"].fillna("").astype(str).tolist())),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            f"{fmt_metric(xa)}: %{{x:,.4g}}<br>"
            f"{fmt_metric(ya)}: %{{y:,.4g}}<br>"
            "<span style='font-size:10px'>%{customdata[1]}</span><extra>Pareto</extra>"
        ),
    ))

    # Connect Pareto frontier with a dashed line for visual continuity
    if len(par) >= 2:
        fig.add_trace(go.Scatter(
            x=par[xa], y=par[ya],
            mode="lines",
            line=dict(color=ACCENT, width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        height=600,
        xaxis_title=fmt_metric(xa),
        yaxis_title=fmt_metric(ya),
        legend=dict(**LEGEND_BASE),
        **LAYOUT_BASE,
    )
    if log_axes:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        apply_log_axis_readability(fig, x=True, y=True)

    st.plotly_chart(fig, width="stretch")

    n_pareto = int(pareto.sum())
    st.caption(
        f"**{n_pareto}** Pareto-optimal model(s) out of **{len(agg)}** total — "
        f"these are the only candidates worth comparing on this trade-off."
    )

    # Pareto table
    st.markdown('<div class="section-heading">Pareto-Optimal Models</div>', unsafe_allow_html=True)
    par_show = par[["xml_file", "model_full_path"] + [c for c in [xa, ya] + [v for v in AXIS_OPTIONS if v not in (xa, ya)] if c in par.columns]]
    par_show = par_show.rename(columns={"xml_file": "Model", "model_full_path": "Path"})
    par_show.index = range(1, len(par_show) + 1)
    st.dataframe(par_show.round(5), width="stretch", height=min(500, 40 + 36 * len(par_show)))


# ═══════════════════════════════════════════════════════════════════════════
# Tab 3: Compare — focused subset comparison
# ═══════════════════════════════════════════════════════════════════════════
@st.fragment
def _render_compare(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-heading">Compare Selected Models</div>', unsafe_allow_html=True)
    st.caption("Select 2–10 models to compare deeply. All charts below render only on your subset.")

    all_models = sorted(df["xml_file"].dropna().unique().tolist(), key=lambda s: int(s[1:]) if s[1:].isdigit() else -1)
    fp_map = df[["xml_file", "model_full_path"]].drop_duplicates("xml_file") if "model_full_path" in df.columns else None
    fp_lookup = dict(zip(fp_map["xml_file"], fp_map["model_full_path"])) if fp_map is not None else {}

    # Smart default: pick the top 5 by composite score
    if "cmp_models" not in st.session_state:
        try:
            df_stab_def = _stability_ranking(df)
            default_sel = df_stab_def["xml_file"].head(5).tolist()
        except Exception:
            default_sel = all_models[:5]
        st.session_state["cmp_models"] = default_sel

    # Quick-load buttons
    _btn_pareto, _btn_top5, _ = st.columns([2, 2, 6])
    if _btn_pareto.button("Load Pareto models", key="cmp_load_pareto"):
        _metrics = ["total_time_cost", "total_financial_cost", "total_environmental_cost"]
        _agg = df.groupby("xml_file")[_metrics].median()
        _agg = _agg.dropna()
        _is_dominated = pd.Series(False, index=_agg.index)
        for _i in _agg.index:
            _row = _agg.loc[_i]
            _dom = (_agg <= _row).all(axis=1) & (_agg < _row).any(axis=1)
            _dom[_i] = False
            if _dom.any():
                _is_dominated[_i] = True
        _pareto = _agg.index[~_is_dominated].tolist()
        st.session_state["cmp_models"] = _pareto[:10]
        st.rerun()
    if _btn_top5.button("Load Top 5 (score)", key="cmp_load_top5"):
        try:
            _df_stab = _stability_ranking(df)
            st.session_state["cmp_models"] = _df_stab["xml_file"].head(5).tolist()
        except Exception:
            st.session_state["cmp_models"] = all_models[:5]
        st.rerun()

    sel = st.multiselect(
        "Models",
        all_models,
        default=st.session_state.get("cmp_models", []),
        format_func=lambda x: f"{x} — {fp_lookup.get(x, '')}",
        max_selections=10,
        key="cmp_models",
    )

    if len(sel) < 2:
        st.info("Pick at least 2 models above.")
        return

    df_sub = df[df["xml_file"].isin(sel)].copy()
    color_map = {m: PLOTLY_COLORS[i % len(PLOTLY_COLORS)] for i, m in enumerate(sel)}

    # ── Section 1: Cost vs Servers ───────────────────────────────────────
    st.markdown('<div class="section-heading" style="margin-top:14px">Cost vs. Server Count</div>', unsafe_allow_html=True)
    cv1, cv2, cv3 = st.columns([2, 3, 1])
    metric = cv1.selectbox("Metric", METRIC_OPTIONS, format_func=fmt_metric, key="cmp_metric")
    vol_opts = sorted(df_sub["data_volume"].dropna().unique().astype(int).tolist())
    sel_vols = cv2.multiselect(
        "Data volume(s)",
        vol_opts,
        default=[vol_opts[0]] if vol_opts else [],
        format_func=fmt_volume,
        key="cmp_vols",
    )
    use_log = cv3.checkbox("Log Y", value=True, key="cmp_log")

    if sel_vols:
        df_curve = df_sub[df_sub["data_volume"].astype(int).isin(sel_vols)]
        grp = df_curve.groupby(["xml_file", "servers", "data_volume"], as_index=False)[metric].mean()
        grp["series"] = grp["xml_file"] + (" @ " + grp["data_volume"].astype(int).map(fmt_volume) if len(sel_vols) > 1 else "")
        fig_curve = px.line(
            grp.sort_values(["xml_file", "servers"]),
            x="servers", y=metric, color="series", markers=True,
            color_discrete_sequence=PLOTLY_COLORS,
            labels={"servers": "Server count", metric: fmt_metric(metric), "series": "Model"},
            log_y=use_log,
        )
        fig_curve.update_traces(line=dict(width=2), marker=dict(size=7))
        fig_curve.update_layout(height=440, legend=dict(**LEGEND_BASE), **LAYOUT_BASE)
        if use_log:
            apply_log_axis_readability(fig_curve, y=True)
        st.plotly_chart(fig_curve, width="stretch")

    # ── Section 2: Bar at fixed setting ─────────────────────────────────
    st.markdown('<div class="section-heading" style="margin-top:14px">Ranked at Fixed Setting</div>', unsafe_allow_html=True)
    bv1, bv2, bv3 = st.columns([2, 2, 3])
    fix_vol = bv1.selectbox("Volume", vol_opts, format_func=fmt_volume, key="cmp_bar_v")
    srv_opts = sorted(df_sub["servers"].dropna().unique().astype(int).tolist())
    fix_srv = bv2.selectbox("Servers", srv_opts, key="cmp_bar_s")
    bar_metric = bv3.radio("Metric", METRIC_OPTIONS, format_func=fmt_metric, horizontal=True, key="cmp_bar_m")

    df_bar = df_sub[(df_sub["data_volume"] == fix_vol) & (df_sub["servers"] == fix_srv)]
    if df_bar.empty:
        st.info("No data for this (volume, servers) combination among the selected models.")
    else:
        agg_bar = df_bar.groupby("xml_file", as_index=False)[bar_metric].mean().sort_values(bar_metric)
        fig_bar = px.bar(
            agg_bar, x="xml_file", y=bar_metric, color="xml_file",
            color_discrete_map=color_map,
            labels={"xml_file": "Model", bar_metric: fmt_metric(bar_metric)},
        )
        fig_bar.update_layout(height=380, showlegend=False, **LAYOUT_BASE)
        st.plotly_chart(fig_bar, width="stretch")

    # ── Section 3: Per-query breakdown ──────────────────────────────────
    # Numeric sort so q10 comes after q9, not between q1 and q2
    q_cols = sorted(
        (c for c in df_sub.columns if re.match(r"q\d+_time_cost$", c)),
        key=lambda c: int(c.split("_")[0][1:]),
    )
    if q_cols:
        st.markdown('<div class="section-heading" style="margin-top:14px">Per-Query Time Cost</div>', unsafe_allow_html=True)
        pq1, pq2 = st.columns(2)
        pq_vol = pq1.selectbox("Volume", vol_opts, format_func=fmt_volume, key="cmp_pq_v")
        pq_srv = pq2.selectbox("Servers", srv_opts, key="cmp_pq_s")
        df_pq_src = df_sub[(df_sub["data_volume"] == pq_vol) & (df_sub["servers"] == pq_srv)]

        df_pq = (
            df_pq_src.melt(
                id_vars=["xml_file"], value_vars=q_cols,
                var_name="Query", value_name="Time Cost (s)",
            )
            .dropna(subset=["Time Cost (s)"])
            .rename(columns={"xml_file": "Model"})
        )
        df_pq["Query"] = "Q" + df_pq["Query"].str.extract(r"q(\d+)_time_cost")[0]
        if not df_pq.empty:
            fig_pq = px.bar(
                df_pq, x="Query", y="Time Cost (s)", color="Model",
                barmode="group", color_discrete_map=color_map,
            )
            fig_pq.update_layout(height=380, **LAYOUT_BASE, legend=dict(**LEGEND_BASE), yaxis_type="log")
            apply_log_axis_readability(fig_pq, y=True)
            st.plotly_chart(fig_pq, width="stretch")
        else:
            st.info("No per-query data at this setting.")

    # ── Section 4: Normalized radar + companion table ───────────────────
    st.markdown('<div class="section-heading" style="margin-top:14px">Normalized Cost Profile <span class="badge-norm">&#128208; Eq. 14</span></div>', unsafe_allow_html=True)
    rec_agg = df.groupby("xml_file", as_index=False)[METRIC_OPTIONS].mean()
    norm = eq14_normalize_metrics(rec_agg.copy(), suffix="")
    norm_sub = norm[norm["xml_file"].isin(sel)]

    # Cosmetic floor: shift the [0, 1] range up to [FLOOR, 1] so the cheapest
    # model doesn't collapse to the origin. Ranking and D̃ are unaffected —
    # the underlying Eq. 14 normalisation in `norm` is untouched.
    FLOOR = 0.05
    _shift = lambda v: FLOOR + (1.0 - FLOOR) * v

    axes = ["Time Cost", "Environmental Cost", "Financial Cost"]
    cols = ["total_time_cost", "total_environmental_cost", "total_financial_cost"]
    radar = go.Figure()
    table_rows = []
    for m in sel:
        r = norm_sub[norm_sub["xml_file"] == m]
        if r.empty:
            continue
        raw_vals = [float(r[c].iloc[0]) for c in cols]
        vals = [_shift(v) for v in raw_vals]
        radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=axes + [axes[0]],
            fill="none",
            name=m,
            line=dict(color=color_map[m], width=3),
            marker=dict(size=10, color=color_map[m], line=dict(width=2, color="#fff")),
            mode="lines+markers",
            customdata=[[rv] for rv in raw_vals + [raw_vals[0]]],
            hovertemplate=f"<b>{m}</b><br>%{{theta}}: %{{customdata[0]:.3f}} (raw)<extra></extra>",
        ))
        table_rows.append({
            "Model": m,
            **{ax: round(v, 3) for ax, v in zip(axes, vals)},
            "Sum (lower=better)": round(sum(vals), 3),
        })
    radar.update_layout(
        polar=dict(
            bgcolor=BG_PANEL,
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=BORDER, linecolor=BORDER, tickfont=dict(size=10, color=TEXT_MUTED)),
            angularaxis=dict(tickfont=dict(size=11, color="#fff"), gridcolor=BORDER, linecolor=BORDER),
        ),
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_MUTED),
        legend=dict(**LEGEND_BASE),
        height=460,
        autosize=True,
        margin=dict(l=40, r=140, t=20, b=20),
    )
    rd_left, rd_right = st.columns([3, 2])
    rd_left.plotly_chart(radar, width="stretch")
    if table_rows:
        df_radar = pd.DataFrame(table_rows).sort_values("Sum (lower=better)").reset_index(drop=True)
        df_radar.index = range(1, len(df_radar) + 1)
        rd_right.markdown("&nbsp;")
        rd_right.dataframe(df_radar, width="stretch", height=min(400, 40 + 36 * len(df_radar)))
    st.caption(f"Lines closer to centre = better overall. Floor at {FLOOR:.2f} (cosmetic — keeps cheapest model visible); 1 = worst. Hover shows the raw [0, 1] value. Sum column ranks the selected models on overall profile.")


# ═══════════════════════════════════════════════════════════════════════════
# Tab 4: Transformation Tree — drill into merge/split chains
# ═══════════════════════════════════════════════════════════════════════════
@st.fragment
def _render_tree(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-heading">Transformation Tree</div>', unsafe_allow_html=True)
    st.caption(
        "Surfaces where the transformation tree breaks (cost cliffs), which branch families "
        "stayed healthy, and how cost evolves along a chain from root to leaf."
    )

    if "model_full_path" not in df.columns:
        st.info("Path information unavailable.")
        return

    t1, t2, t3, t4 = st.columns([2, 2, 2, 2])
    metric = t1.selectbox("Metric", METRIC_OPTIONS, format_func=fmt_metric, key="tree_m")
    _vol_opts = sorted(df["data_volume"].dropna().unique().astype(int).tolist())
    _srv_opts = sorted(df["servers"].dropna().unique().astype(int).tolist())
    tree_vol = t2.selectbox(
        "Data volume", ["All (aggregate)"] + _vol_opts,
        format_func=lambda v: v if v == "All (aggregate)" else fmt_volume(v),
        key="tree_vol",
    )
    tree_srv = t3.selectbox("Servers", ["All (aggregate)"] + _srv_opts, key="tree_srv")
    agg_mode = t4.radio("Aggregate", ["mean", "median"], horizontal=True, key="tree_agg")
    agg_fn = "mean" if agg_mode == "mean" else "median"

    df_tree_src = df.copy()
    if tree_vol != "All (aggregate)":
        df_tree_src = df_tree_src[df_tree_src["data_volume"] == tree_vol]
    if tree_srv != "All (aggregate)":
        df_tree_src = df_tree_src[df_tree_src["servers"] == tree_srv]
    if df_tree_src.empty:
        st.info("No data at this (volume, servers) combination.")
        return

    _vol_disp = "all volumes" if tree_vol == "All (aggregate)" else fmt_volume(tree_vol)
    _srv_disp = "all server counts" if tree_srv == "All (aggregate)" else f"{tree_srv} servers"
    st.caption(f"Tree built from **{_vol_disp}** · **{_srv_disp}** · {agg_mode} aggregation.")

    agg = df_tree_src.groupby(["xml_file", "model_full_path"], as_index=False)[metric].agg(agg_fn)

    tree = build_tree_dataframe(agg, metric=metric, label_col="xml_file", path_col="model_full_path")
    if tree.empty:
        st.info("No tree data to render.")
        return

    # Helpers used across sections
    parent_value_map = dict(zip(tree["id"], tree["value"]))
    parent_label_map = dict(zip(tree["id"], tree["label"]))
    parent_map = dict(zip(tree["id"], tree["parent"]))
    root_row = tree[tree["depth"] == 0]
    root_cost = float(root_row["value"].iloc[0]) if not root_row.empty else 1.0
    root_id = root_row["id"].iloc[0] if not root_row.empty else ""

    def _first_segment(node_id: str) -> str:
        if not node_id or node_id == root_id:
            return "(root)"
        return str(node_id).replace("\\", "/").split("/")[0]

    tree["root_branch"] = tree["id"].apply(_first_segment)
    tree["log_cost"] = tree["value"].clip(lower=1e-9).apply(math.log10)
    tree["cost_vs_root"] = tree["value"] / max(root_cost, 1e-12)

    # ── Section 1: Cost Cliffs ──────────────────────────────────────────
    st.markdown('<div class="section-heading" style="margin-top:6px">Cost Cliffs &nbsp;<span class="badge-raw">Top parent→child jumps</span></div>', unsafe_allow_html=True)
    st.caption(
        "Each bar = one parent→child transition, ranked by how much cost changed. "
        "Red = transformation made cost worse; green = made it better. Bar length is log₁₀(child/parent)."
    )

    cliffs = tree[tree["parent"] != ""].copy()
    cliffs["parent_value"] = cliffs["parent"].map(parent_value_map)
    cliffs["ratio"] = cliffs["value"] / cliffs["parent_value"].replace(0, float("nan"))
    cliffs = cliffs.dropna(subset=["ratio"])
    cliffs = cliffs[cliffs["ratio"] > 0]
    cliffs["log_ratio"] = cliffs["ratio"].apply(math.log10)
    cliffs["abs_log"] = cliffs["log_ratio"].abs()

    if cliffs.empty:
        st.info("No transitions to analyse.")
    else:
        n_show = min(20, len(cliffs))
        top_cliffs = cliffs.nlargest(n_show, "abs_log").sort_values("log_ratio").copy()
        top_cliffs["parent_label"] = top_cliffs["parent"].map(parent_label_map)
        top_cliffs["transition"] = top_cliffs.apply(
            lambda r: f"{r['parent_label']} → {r['label']}  ·  {r['op']}", axis=1,
        )
        top_cliffs["bar_color"] = top_cliffs["log_ratio"].apply(lambda v: RED if v > 0 else GREEN)
        top_cliffs["ratio_label"] = top_cliffs["ratio"].apply(
            lambda r: f"x{r:,.0f}" if r >= 1 else f"x{1 / r:,.0f}"
        )

        fig_cliff = go.Figure(go.Bar(
            x=top_cliffs["log_ratio"],
            y=top_cliffs["transition"],
            orientation="h",
            marker=dict(color=top_cliffs["bar_color"]),
            text=top_cliffs["ratio_label"],
            textposition="outside",
            textfont=dict(size=10, color=TEXT_MAIN),
            customdata=top_cliffs[["ratio", "value", "parent_value"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "child / parent: %{customdata[0]:,.4g}<br>"
                f"{fmt_metric(metric)}: %{{customdata[1]:,.4g}} (parent: %{{customdata[2]:,.4g}})<extra></extra>"
            ),
        ))
        fig_cliff.update_layout(
            height=max(360, 26 * n_show),
            xaxis_title="log₁₀(child / parent) — positive = worse, negative = better",
            margin=dict(l=260, r=100, t=20, b=40),
            **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        )
        fig_cliff.add_vline(x=0, line_dash="dot", line_color=BORDER)
        st.plotly_chart(fig_cliff, width="stretch")
        st.caption(f"Showing top {n_show} of {len(cliffs)} transitions by magnitude. Sign tells direction; bar length tells size on a log scale.")

    # ── Section 2: Cost vs Depth scatter ────────────────────────────────
    st.markdown('<div class="section-heading" style="margin-top:14px">Cost vs. Depth &nbsp;<span class="badge-raw">Branch families</span></div>', unsafe_allow_html=True)
    st.caption(
        "Each dot = one model. X = how many transformations have been applied (tree depth). "
        "Y = log₁₀(cost). Colored by root-level branch — clusters that stay low are healthy branches; "
        "clusters that shoot up are broken."
    )

    scatter_df = tree.copy()
    fig_scatter = px.scatter(
        scatter_df,
        x="depth",
        y="log_cost",
        color="root_branch",
        hover_data={
            "label": True, "id": True, "value": ":,.4g",
            "depth": True, "log_cost": ":.2f", "root_branch": True,
        },
        labels={
            "depth": "Tree depth (# operations applied)",
            "log_cost": f"log₁₀({fmt_metric(metric)})",
            "root_branch": "Root branch",
            "label": "Model",
            "value": fmt_metric(metric),
        },
        color_discrete_sequence=PLOTLY_COLORS,
    )
    fig_scatter.update_traces(marker=dict(size=11, line=dict(width=1, color="#fff"), opacity=0.85))
    fig_scatter.update_layout(height=440, legend=dict(**LEGEND_BASE), **LAYOUT_BASE)
    st.plotly_chart(fig_scatter, width="stretch")

    # ── Section 3: Tree structure (icicle, improved) ────────────────────
    st.markdown('<div class="section-heading" style="margin-top:14px">Tree Structure</div>', unsafe_allow_html=True)
    tc1, tc2 = st.columns([2, 2])
    color_mode = tc1.radio(
        "Color by",
        ["log(cost)", "log(cost / root)"],
        horizontal=True,
        key="tree_color_mode",
    )
    hide_broken = tc2.checkbox(
        "Hide broken subtrees (>100× root cost)",
        value=False,
        key="tree_hide_broken",
    )
    st.caption("Click a tile to zoom in. Tile text = Δ% vs the parent model.")

    if color_mode == "log(cost / root)":
        colors_vec = tree["cost_vs_root"].clip(lower=1e-12).apply(math.log10)
        cb_title = "log₁₀(cost / root)"
        # green (cheaper than root) -> yellow (= root) -> red (more expensive)
        colorscale = "RdYlGn_r"
    else:
        colors_vec = tree["log_cost"]
        cb_title = f"log₁₀({fmt_metric(metric)})"
        # green (cheap) -> red (expensive)
        colorscale = "RdYlGn_r"

    # Determine which nodes to drop (broken + all descendants of broken)
    tree_render = tree
    colors_render = colors_vec
    delta_label = tree["delta_vs_parent"].apply(
        lambda v: f"{v:+.0%}" if pd.notna(v) and abs(v) > 1e-6 else ""
    )
    delta_label_render = delta_label
    n_hidden = 0

    if hide_broken:
        hidden = set(tree.loc[tree["cost_vs_root"] > 100, "id"].tolist())
        # Expand: any node with an ancestor in `hidden` is also hidden
        changed = True
        while changed:
            changed = False
            for node_id, par in parent_map.items():
                if par in hidden and node_id not in hidden:
                    hidden.add(node_id)
                    changed = True
        if hidden:
            mask = ~tree["id"].isin(hidden)
            tree_render = tree[mask].reset_index(drop=True)
            colors_render = colors_vec[mask.values].reset_index(drop=True)
            delta_label_render = delta_label[mask.values].reset_index(drop=True)
            n_hidden = int((~mask).sum())

    # Rescale colorbar to the rendered subset so contrast isn't crushed by hidden outliers.
    # For the cost/root mode, center the diverging scale on 0 (= same as root).
    if color_mode == "log(cost / root)":
        lim = max(abs(float(colors_render.min())), abs(float(colors_render.max())), 1e-9)
        cmin, cmax = -lim, lim
    else:
        cmin, cmax = float(colors_render.min()), float(colors_render.max())
        if cmin == cmax:
            cmin, cmax = cmin - 1, cmax + 1

    if tree_render.empty:
        st.info("All nodes are broken — nothing to render. Toggle off 'Hide broken subtrees' to inspect.")
    else:
        fig_tree = go.Figure(go.Icicle(
            ids=tree_render["id"],
            parents=tree_render["parent"],
            labels=tree_render["label"] + " · " + tree_render["op"],
            values=[1] * len(tree_render),
            branchvalues="remainder",
            marker=dict(
                colors=colors_render,
                colorscale=colorscale,
                cmin=cmin, cmax=cmax,
                colorbar=dict(title=cb_title, thickness=12),
                line=dict(width=1, color=BORDER),
            ),
            text=delta_label_render,
            textinfo="label+text",
            customdata=tree_render[["value", "delta_vs_parent"]].fillna(0).values,
            hovertemplate=(
                "<b>%{label}</b><br>"
                f"{fmt_metric(metric)}: %{{customdata[0]:,.4g}}<br>"
                "Δ vs parent: %{customdata[1]:+.1%}<extra></extra>"
            ),
        ))
        fig_tree.update_layout(height=520, autosize=True, paper_bgcolor=BG_PANEL, font=dict(color=TEXT_MAIN))
        st.plotly_chart(fig_tree, width="stretch")
        if n_hidden > 0:
            st.caption(f"_{n_hidden} broken node(s) hidden._")

    # ── Section 4: Chain comparison ─────────────────────────────────────
    st.markdown('<div class="section-heading" style="margin-top:14px">Chain Comparison &nbsp;<span class="badge-raw">Log Y</span></div>', unsafe_allow_html=True)
    st.caption(
        "Pick up to 5 leaf models to trace each chain from root → leaf. Y axis is log-scaled. "
        "Biggest jump in each chain is marked."
    )

    leaf_options = tree.sort_values("depth", ascending=False)["id"].tolist()
    label_for_id = dict(zip(tree["id"], tree["label"] + " — " + tree["id"]))
    default_chains = [leaf_options[0]] if leaf_options else []
    chosen_chains = st.multiselect(
        "Chains",
        options=leaf_options,
        default=default_chains,
        format_func=lambda x: label_for_id.get(x, x),
        max_selections=5,
        key="tree_chains_multi",
    )

    if not chosen_chains:
        st.info("Pick at least one chain above.")
    else:
        chain_frames = []
        chain_lookup = {}
        for leaf in chosen_chains:
            c = chain_to_root(tree, leaf)
            if c.empty:
                continue
            c = c.copy()
            c["step"] = range(len(c))
            c["chain_label"] = label_for_id.get(leaf, leaf)
            chain_frames.append(c)
            chain_lookup[leaf] = c

        if not chain_frames:
            st.info("No chain data for the selected leaves.")
        else:
            df_chains = pd.concat(chain_frames, ignore_index=True)
            df_chains["safe_value"] = df_chains["value"].clip(lower=1e-9)
            # Pretty hover text: per-row model + operation
            df_chains["hover_text"] = (
                df_chains["label"].astype(str) + " (" + df_chains["op"].astype(str) + ")"
            )

            fig_chains = go.Figure()
            chain_colors = {
                lbl: PLOTLY_COLORS[i % len(PLOTLY_COLORS)]
                for i, lbl in enumerate(df_chains["chain_label"].drop_duplicates().tolist())
            }
            for lbl, sub in df_chains.groupby("chain_label", sort=False):
                sub = sub.sort_values("step")
                fig_chains.add_trace(go.Scatter(
                    x=sub["step"], y=sub["safe_value"],
                    mode="lines+markers",
                    name=lbl,
                    line=dict(width=2.5, color=chain_colors[lbl]),
                    marker=dict(size=10, line=dict(width=1.5, color="#fff"), color=chain_colors[lbl]),
                    customdata=list(zip(sub["label"].astype(str), sub["op"].astype(str))),
                    hovertemplate=(
                        f"<b>{lbl}</b><br>"
                        "Step %{x} — %{customdata[0]}<br>"
                        "Op: %{customdata[1]}<br>"
                        f"{fmt_metric(metric)}: %{{y:,.4g}}<extra></extra>"
                    ),
                ))

            # Annotate biggest jump per chain
            for leaf, c in chain_lookup.items():
                if len(c) < 2:
                    continue
                vals = c["value"].clip(lower=1e-9).reset_index(drop=True).tolist()
                log_jumps = [math.log10(vals[i] / vals[i - 1]) for i in range(1, len(vals))]
                if not log_jumps:
                    continue
                jump_idx = max(range(len(log_jumps)), key=lambda i: abs(log_jumps[i]))
                jump_ratio = vals[jump_idx + 1] / vals[jump_idx]
                arrow_color = RED if jump_ratio > 1 else GREEN
                ann_text = f"x{jump_ratio:,.0f}" if jump_ratio >= 1 else f"x{1 / jump_ratio:,.0f}"
                fig_chains.add_annotation(
                    x=jump_idx + 1,
                    y=vals[jump_idx + 1],
                    text=ann_text,
                    showarrow=True, arrowhead=2, arrowcolor=arrow_color,
                    font=dict(size=11, color=arrow_color),
                    bgcolor=BG_CARD, bordercolor=arrow_color, borderwidth=1,
                    ax=0, ay=-35,
                )

            # X-axis tick labels: show the operation name at each step from the longest chain
            longest = max(chain_frames, key=len)
            max_steps = int(longest["step"].max()) if not longest.empty else 0
            tick_vals = list(range(max_steps + 1))
            tick_text = []
            for s in tick_vals:
                if s == 0:
                    tick_text.append("root")
                else:
                    op_at = longest[longest["step"] == s]["op"]
                    tick_text.append(f"#{s}<br>{op_at.iloc[0]}" if not op_at.empty else f"#{s}")

            fig_chains.update_layout(
                height=480,
                legend=dict(**LEGEND_BASE),
                yaxis_type="log",
                yaxis_title=fmt_metric(metric),
                margin=dict(t=30, b=80, l=70, r=20),
                **{k: v for k, v in LAYOUT_BASE.items() if k not in ("margin", "xaxis")},
            )
            fig_chains.update_xaxes(
                tickmode="array", tickvals=tick_vals, ticktext=tick_text,
                tickfont=dict(size=9), title_text="Step (root → leaf)",
            )
            apply_log_axis_readability(fig_chains, y=True)
            st.plotly_chart(fig_chains, width="stretch")

            # Per-chain tables in expanders
            for leaf in chosen_chains:
                c = chain_lookup.get(leaf)
                if c is None or c.empty:
                    continue
                with st.expander(f"Steps for {label_for_id.get(leaf, leaf)}", expanded=False):
                    c_table = c[["label", "op", "value", "delta_vs_parent", "delta_abs"]].rename(columns={
                        "label": "Model", "op": "Operation", "value": fmt_metric(metric),
                        "delta_vs_parent": "Δ% vs parent", "delta_abs": "Δ abs",
                    })
                    c_table.index = range(len(c_table))
                    c_table.index.name = "Step"
                    st.dataframe(c_table.round(5), width="stretch", height=min(360, 40 + 36 * len(c_table)))


# ═══════════════════════════════════════════════════════════════════════════
# Tab 5: Data Table
# ═══════════════════════════════════════════════════════════════════════════
@st.fragment
def _render_data_table(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-heading">Filtered Record Set</div>', unsafe_allow_html=True)
    default_cols = ["xml_file", "mode", "data_volume", "servers"] + METRIC_OPTIONS
    show_cols = st.multiselect(
        "Columns", df.columns.tolist(),
        default=[c for c in default_cols if c in df.columns],
        key="an_tbl_cols",
    )
    st.dataframe(df[show_cols].reset_index(drop=True), width="stretch", height=520)
    st.download_button(
        "Download Filtered Data (CSV)",
        data=df[show_cols].to_csv(index=False).encode(),
        file_name="filtered_costs.csv", mime="text/csv",
    )
