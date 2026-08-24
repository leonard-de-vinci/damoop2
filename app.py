"""
DaMoOp* — Unified Pipeline UI
Run with:  streamlit run app.py
  (from inside DataModelExplorer_V2/)

Architecture:
  lib/         — shared constants, equations, data helpers, scenario logic
  views/       — one module per page (pipeline, input_config, cost_analysis, …)
  app.py       — slim entry point: config, CSS, sidebar, routing
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from lib.constants import JAR_DMG, JAR_COST, build_css
from lib.data_loading import count_dms, detect_active_profile, get_active_dms_dir, get_active_run_id
from lib.persistence import load_ui_state, save_ui_state
from lib.scenarios import load_scenarios, save_scenarios

# NOTE: view modules are imported lazily in the routing section below — importing
# them all here pulls in plotly (~1-2 s) on every cold start even for pages that
# don't use it. Same for build_costs_dataframe (used only in the one-time init).

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DaMoOp*",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
st.markdown(build_css(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Restore persisted state (runs before session-state defaults so persisted
# values win over the fallback literals below)
# ---------------------------------------------------------------------------
_persisted = load_ui_state()
for _k, _v in _persisted.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Session state defaults  (only applied when key is still missing)
# ---------------------------------------------------------------------------
if "data_bust"          not in st.session_state: st.session_state["data_bust"]          = datetime.now().isoformat()
if "dmg_log"            not in st.session_state: st.session_state["dmg_log"]            = ""
if "cost_log"           not in st.session_state: st.session_state["cost_log"]           = ""
if "dmg_status"         not in st.session_state: st.session_state["dmg_status"]         = None
if "cost_status"        not in st.session_state: st.session_state["cost_status"]        = None
if "base_model"         not in st.session_state: st.session_state["base_model"]         = "TPC_C"
if "scenario_edit_mode" not in st.session_state: st.session_state["scenario_edit_mode"] = None
if "active_run_id"      not in st.session_state: st.session_state["active_run_id"]      = "LIVE"
if "dmg_path_history"   not in st.session_state: st.session_state["dmg_path_history"]   = []
if "cost_path_history"  not in st.session_state: st.session_state["cost_path_history"]  = []

if "scenarios" not in st.session_state:
    _loaded_scenarios = load_scenarios()
    if _loaded_scenarios is not None:
        st.session_state["scenarios"] = _loaded_scenarios
    else:
        try:
            from lib.parse_costs import build_costs_dataframe
            _init_df   = build_costs_dataframe(get_active_dms_dir())
            _init_vols = sorted(_init_df["data_volume"].dropna().unique().astype(int).tolist()) if not _init_df.empty else [1000, 10000, 100000, 1000000, 100000000]
            _init_srvs = sorted(_init_df["servers"].dropna().unique().astype(int).tolist())     if not _init_df.empty else [10, 50, 100, 500, 1000, 5000, 10000]
        except Exception:
            _init_vols = [1000, 10000, 100000, 1000000, 100000000]
            _init_srvs = [10, 50, 100, 500, 1000, 5000, 10000]
        st.session_state["scenarios"] = {
            "baseline": {
                "label":                    "Baseline (Current)",
                "description":              "Current configuration from Input files",
                "volumes":                  _init_vols,
                "servers":                  _init_srvs,
                "query_freq_adjustments":   {},
                "created_at":               datetime.now().isoformat(),
            }
        }
        save_scenarios(st.session_state["scenarios"])

if "custom_rules" not in st.session_state:
    st.session_state["custom_rules"] = []

# ---------------------------------------------------------------------------
# Sidebar — branding
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
      <div class="sidebar-brand-icon">&#9707;</div>
      <div class="sidebar-brand-text">
        <span class="sidebar-brand-name">DaMoOp*</span>
        <span class="sidebar-brand-sub">Data Model Optimizer</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — navigation  (page persisted via URL query param)
# ---------------------------------------------------------------------------
PAGES = ["Input Configuration", "Pipeline", "Cost Analysis", "Profile Sweep", "Scenario Analysis", "Predict", "History"]

# On refresh the browser re-sends the URL, so read the query param first
_qp_page = st.query_params.get("page", "")
if _qp_page in PAGES and "active_page" not in st.session_state:
    st.session_state["active_page"] = _qp_page

if "active_page" not in st.session_state:
    st.session_state["active_page"] = PAGES[0]

page = st.sidebar.radio(
    "Navigation", PAGES,
    index=PAGES.index(st.session_state["active_page"]),
    label_visibility="collapsed",
    key="nav_radio"
)
st.session_state["active_page"] = page

# Keep URL in sync so refresh lands on the same page
st.query_params["page"] = page

# ---------------------------------------------------------------------------
# Sidebar — system status
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-section">System Status</div>', unsafe_allow_html=True)

_dmg_dot  = "sidebar-dot-ok" if JAR_DMG.exists()  else "sidebar-dot-err"
_cost_dot = "sidebar-dot-ok" if JAR_COST.exists() else "sidebar-dot-err"
st.sidebar.markdown(
    f"""
    <div class="sidebar-status-row">
      <span class="sidebar-status-label">DMgeneration.jar</span>
      <span class="{_dmg_dot}"></span>
    </div>
    <div class="sidebar-status-row">
      <span class="sidebar-status-label">CostCalculation.jar</span>
      <span class="{_cost_dot}"></span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="sidebar-section">Active Profile</div>', unsafe_allow_html=True)
_active_profile = detect_active_profile()
if _active_profile is not None:
    st.session_state["active_profile"] = _active_profile
_profile_label = f"Profile {_active_profile}" if _active_profile else "Custom (Input/ edited)"
_profile_dot = "sidebar-dot-ok" if _active_profile else "sidebar-dot-warn"
st.sidebar.markdown(
    f'<div class="sidebar-output-badge">'
    f'<span class="{_profile_dot}"></span>'
    f'{_profile_label}'
    f'</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="sidebar-section">Output</div>', unsafe_allow_html=True)
n_dms    = count_dms()
_out_dot = "sidebar-dot-ok" if n_dms > 0 else "sidebar-dot-warn"
_active_run_label = "Live" if get_active_run_id() == "LIVE" else f"Run {get_active_run_id()}"
st.sidebar.markdown(
    f'<div class="sidebar-output-badge">'
    f'<span class="{_out_dot}"></span>'
    f'{n_dms} cost file{"s" if n_dms != 1 else ""} in {_active_run_label}'
    f'</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
page_subtitles = {
    "Pipeline":            "Run DMG and Cost Calculation phases",
    "Input Configuration": "Edit queries, schema, config, and server settings",
    "Cost Analysis":       "Visualize and compare cost results",
    "Scenario Analysis":   "Create parameter variations and compare model rankings",
    "Profile Sweep":       "Generate parametric profiles for ML training data",
    "Predict":             "Live cost prediction from schema + workload (trained model)",
    "History":             "Manage saved snapshots",
}
st.markdown(
    f"""
    <div class="app-header">
        <h1>DaMoOp*</h1>
        <p>{page_subtitles[page]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
# Lazy import: only the active page's module (and its deps, e.g. plotly) is loaded.
# Python caches modules across reruns, so repeat visits stay instant.
if page == "Pipeline":
    from views import pipeline
    pipeline.render()
elif page == "Input Configuration":
    from views import input_config
    input_config.render()
elif page == "Cost Analysis":
    from views import cost_analysis
    cost_analysis.render()
elif page == "Scenario Analysis":
    from views import scenario_analysis
    scenario_analysis.render()
elif page == "Profile Sweep":
    from views import sweep
    sweep.render()
elif page == "Predict":
    from views import predict
    predict.render()
elif page == "History":
    from views import history
    history.render()

# ---------------------------------------------------------------------------
# Persist state to disk at end of every render so refresh restores it
# ---------------------------------------------------------------------------
save_ui_state(st.session_state)
