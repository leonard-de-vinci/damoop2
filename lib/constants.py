"""Design-system constants, paths, and CSS for the DaMoOp* dashboard."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).parent.parent.resolve()
JAR_DMG        = PROJECT_ROOT / "jars" / "DMgeneration.jar"
JAR_COST       = PROJECT_ROOT / "jars" / "CostCalculation.jar"
INPUT_DIR         = PROJECT_ROOT / "Input"
CONF_DIR          = PROJECT_ROOT / "conf"   # CostCalculation.jar reads keys/index/config/queries from here
INPUT_BACKUP_DIR  = INPUT_DIR / "backup_original"
PROFILES_DIR      = INPUT_DIR / "profiles"
OUTPUT_DIR        = PROJECT_ROOT / "Output"
OUTPUT_BACKUP_DIR = OUTPUT_DIR / "backup_original"
OUTPUT_DMS_DIR    = OUTPUT_DIR / "GeneratedDMs"
RUNS_DIR          = OUTPUT_DIR / "runs"
HISTORY_DIR       = PROJECT_ROOT / "cost_history"
MODELS_DIR        = INPUT_DIR / "models"
SCENARIOS_FILE    = PROJECT_ROOT / "scenarios" / "scenarios.json"
HISTORY_DIR.mkdir(exist_ok=True)
INPUT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DMS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
SCENARIOS_FILE.parent.mkdir(parents=True, exist_ok=True)

RUN_METADATA_FILE = "run_metadata.json"
RUN_QUERIES_FILE  = "queries.txt"

FILE_QUERIES_DMG  = INPUT_DIR / "queries_dmg.txt"
FILE_QUERIES_COST = INPUT_DIR / "queries.txt"
FILE_CONFIG       = INPUT_DIR / "config.txt"
FILE_INDEX        = INPUT_DIR / "index.txt"
FILE_KEYS         = INPUT_DIR / "keys.txt"
FILE_SIGNATURES   = INPUT_DIR / "signatures_TPCC.txt"

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
ACCENT      = "#1f77b4"
ACCENT_SOFT = "#d4e6f1"
BG_CARD     = "#ffffff"
BG_PANEL    = "#f8f9fa"
TEXT_MUTED  = "#6c757d"
TEXT_MAIN   = "#212529"
BORDER      = "#dee2e6"
GREEN       = "#28a745"
RED         = "#dc3545"
YELLOW      = "#ffc107"

# Badge colors — light backgrounds with dark text for light theme
BADGE_RAW_BG    = "#e7f0f9"
BADGE_RAW_TEXT  = "#0052a3"
BADGE_NORM_BG   = "#e7f5f0"
BADGE_NORM_TEXT = "#00632e"
BADGE_MIX_BG    = "#f5e8d9"
BADGE_MIX_TEXT  = "#663300"

# plotly.express.colors.qualitative.Bold, hardcoded so importing lib.constants does
# not drag in plotly (~1-2 s) at app startup. Identical values to px's Bold palette.
PLOTLY_COLORS = [
    "rgb(127, 60, 141)", "rgb(17, 165, 121)", "rgb(57, 105, 172)", "rgb(242, 183, 1)",
    "rgb(231, 63, 116)", "rgb(128, 186, 90)", "rgb(230, 131, 16)", "rgb(0, 134, 149)",
    "rgb(207, 28, 144)", "rgb(249, 123, 114)", "rgb(165, 170, 153)",
]

METRIC_OPTIONS = ["total_time_cost", "total_environmental_cost", "total_financial_cost"]

# Default per-query execution frequencies matching Input/queries.txt values.
# Used as fallback when a scenario has no query_freq_adjustments.
DEFAULT_QUERY_FREQUENCIES: dict[str, int] = {
    "q1": 3000,  "q2": 5000,  "q3": 4000,  "q4": 1000,
    "q5": 1500,  "q6": 200,   "q7": 150,   "q8": 500,
    "q9": 400,   "q10": 300,  "q11": 100,  "q12": 80,
    "q13": 50,   "q14": 75,   "q15": 200,  "q16": 45,
    "q17": 55,   "q18": 15,   "q19": 8,    "q20": 12,
}

# Ordered list of per-query cost column stems present in the costs DataFrame.
QUERY_COST_STEMS: list[str] = [f"q{i}" for i in range(1, 21)]
AXIS_OPTIONS   = METRIC_OPTIONS + ["servers", "data_volume"]

LAYOUT_BASE = dict(
    template="plotly",
    paper_bgcolor=BG_PANEL,
    plot_bgcolor=BG_PANEL,
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12, color=TEXT_MAIN),
    margin=dict(t=40, b=60, l=60, r=20),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
    autosize=True,
)
LEGEND_BASE = dict(bgcolor=BG_CARD, bordercolor=BORDER, borderwidth=1, font=dict(size=11, color=TEXT_MAIN))


def fmt_metric(key: str) -> str:
    _units = {
        "total_time_cost":          "Time Cost (s)",
        "total_environmental_cost": "Environmental Cost (kg CO\u2082e)",
        "total_financial_cost":     "Financial Cost (\u20ac)",
        "data_model_size":          "Data Model Size (GB)",
    }
    return _units.get(key, key.replace("total_", "").replace("_", " ").title())


def fmt_volume(v: int) -> str:
    if v >= 1_000_000: return f" {v // 1_000_000}M"
    if v >= 1_000:     return f" {v // 1_000}K"
    return f" {v}"


def jar_status_html(exists: bool) -> str:
    if exists:
        return '<span class="status-ok">Found</span>'
    return '<span class="status-err">Missing</span>'


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def build_css() -> str:
    """Return the full <style> block (cached — the CSS is static)."""
    return f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {BG_PANEL};
    color: {TEXT_MAIN};
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stMainMenuButton"],
#MainMenu {{
    display: none !important;
}}
header[data-testid="stHeader"] {{
    background: transparent !important;
}}
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 999999 !important;
}}
.block-container {{
    padding-top: 0 !important;
    padding-bottom: 2rem;
    max-width: 100%;
}}
.app-header {{
    background: linear-gradient(90deg, {ACCENT_SOFT} 0%, {BG_CARD} 100%);
    border-bottom: 2px solid {ACCENT};
    padding: 20px 36px 16px 36px;
    margin-bottom: 20px;
}}
.app-header h1 {{ margin:0; font-size:1.55rem; font-weight:700; color:{TEXT_MAIN}; letter-spacing:0.03em; }}
.app-header p  {{ margin:4px 0 0 0; font-size:0.82rem; color:{TEXT_MUTED}; }}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {BG_PANEL} 0%, {BG_CARD} 100%);
    border-right: 1px solid {BORDER};
}}
.sidebar-brand {{
    display:flex; align-items:center; gap:10px;
    padding:22px 4px 16px 4px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 6px;
}}
.sidebar-brand-icon {{
    width:34px; height:34px; border-radius:8px;
    background: linear-gradient(135deg, {ACCENT} 0%, #2a5fa5 100%);
    display:flex; align-items:center; justify-content:center;
    font-size:1.0rem; line-height:1;
}}
.sidebar-brand-text {{ display:flex; flex-direction:column; }}
.sidebar-brand-name {{
    font-size:0.90rem; font-weight:800; letter-spacing:0.12em;
    text-transform:uppercase; color:{TEXT_MAIN};
    line-height:1.2;
}}
.sidebar-brand-sub {{
    font-size:0.60rem; font-weight:500; color:{TEXT_MUTED};
    letter-spacing:0.06em; text-transform:uppercase;
}}
.sidebar-section {{
    font-size:0.60rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.13em; color:{TEXT_MUTED};
    margin:18px 0 6px 0; padding-left:2px;
}}
.sidebar-status-row {{
    display:flex; align-items:center; justify-content:space-between;
    padding:5px 6px; border-radius:6px; margin-bottom:3px;
    background:{BG_PANEL};
}}
.sidebar-status-label {{ font-size:0.74rem; color:{TEXT_MAIN}; font-weight:500; }}
.sidebar-dot-ok   {{ width:8px; height:8px; border-radius:50%; background:{GREEN}; display:inline-block; }}
.sidebar-dot-err  {{ width:8px; height:8px; border-radius:50%; background:{RED};   display:inline-block; }}
.sidebar-dot-warn {{ width:8px; height:8px; border-radius:50%; background:{YELLOW}; display:inline-block; }}
.sidebar-output-badge {{
    display:inline-flex; align-items:center; gap:6px;
    background:{ACCENT_SOFT}; border:1px solid {ACCENT};
    border-radius:20px; padding:3px 12px 3px 8px;
    font-size:0.72rem; font-weight:600; color:{ACCENT};
    margin-top:2px;
}}
.section-heading {{
    font-size:0.70rem; font-weight:700; letter-spacing:0.12em;
    text-transform:uppercase; color:{TEXT_MUTED};
    margin:20px 0 12px 0; padding-bottom:6px; border-bottom:1px solid {BORDER};
}}
.badge-raw  {{ display:inline-block; background:{BADGE_RAW_BG}; color:{BADGE_RAW_TEXT};
               font-size:0.58rem; font-weight:700; padding:2px 8px;
               border-radius:20px; letter-spacing:0.06em; margin-left:10px;
               text-transform:uppercase; vertical-align:middle; }}
.badge-norm {{ display:inline-block; background:{BADGE_NORM_BG}; color:{BADGE_NORM_TEXT};
               font-size:0.58rem; font-weight:700; padding:2px 8px;
               border-radius:20px; letter-spacing:0.06em; margin-left:10px;
               text-transform:uppercase; vertical-align:middle; }}
.badge-mix  {{ display:inline-block; background:{BADGE_MIX_BG}; color:{BADGE_MIX_TEXT};
               font-size:0.58rem; font-weight:700; padding:2px 8px;
               border-radius:20px; letter-spacing:0.06em; margin-left:10px;
               text-transform:uppercase; vertical-align:middle; }}
.status-ok   {{ color:{GREEN}; font-weight:600; font-size:0.80rem; }}
.status-warn {{ color:{YELLOW}; font-weight:600; font-size:0.80rem; }}
.status-err  {{ color:{RED}; font-weight:600; font-size:0.80rem; }}
.log-box {{
    background:{BG_CARD}; border:1px solid {BORDER}; border-radius:6px;
    padding:12px 16px; font-family:"Consolas","Courier New",monospace;
    font-size:0.76rem; color:{TEXT_MAIN}; white-space:pre-wrap;
    max-height:400px; overflow-y:auto;
}}
[data-testid="stMetric"] {{
    background:{BG_CARD}; border:1px solid {BORDER};
    border-left:3px solid {ACCENT}; border-radius:6px; padding:14px 18px;
}}
[data-testid="stMetricLabel"] {{ font-size:0.72rem !important; text-transform:uppercase;
    letter-spacing:0.08em; color:{TEXT_MUTED} !important; }}
[data-testid="stMetricValue"] {{ font-size:1.45rem !important; font-weight:700; color:{TEXT_MAIN} !important; }}
[data-testid="stTabs"] [role="tablist"] {{ border-bottom:2px solid {BORDER}; gap:2px; }}
[data-testid="stTabs"] [role="tab"] {{
    font-size:0.78rem; font-weight:600; text-transform:uppercase;
    letter-spacing:0.07em; color:{TEXT_MUTED}; padding:8px 20px;
    border-radius:4px 4px 0 0; border:1px solid transparent; background:transparent;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color:{ACCENT}; border-color:{BORDER} {BORDER} {BG_PANEL}; background:{BG_PANEL};
}}
.stSelectbox label, .stMultiSelect label, .stSlider label, .stRadio label,
.stTextInput label, .stTextArea label, .stNumberInput label {{
    font-size:0.73rem !important; font-weight:600; text-transform:uppercase;
    letter-spacing:0.07em; color:{TEXT_MUTED} !important;
}}
hr {{ border-color:{BORDER} !important; margin:18px 0 !important; }}

/* Regular buttons */
[data-testid="stButton"] button,
[data-testid="baseButton-secondary"],
button[kind="secondary"],
.stButton > button {{
    background: {ACCENT} !important;
    background-color: {ACCENT} !important;
    color: #ffffff !important;
    border: 1px solid {ACCENT} !important;
    border-radius: 4px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
}}
[data-testid="stButton"] button:hover,
[data-testid="baseButton-secondary"]:hover,
button[kind="secondary"]:hover,
.stButton > button:hover {{
    background: {ACCENT_SOFT} !important;
    background-color: {ACCENT_SOFT} !important;
    color: #ffffff !important;
    border-color: {ACCENT} !important;
}}

/* Download button */
[data-testid="stDownloadButton"] button {{
    background: transparent;
    color: {ACCENT} !important;
    border: 1px solid {ACCENT};
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 6px 18px;
}}
[data-testid="stDownloadButton"] button:hover {{ background:{ACCENT_SOFT}; color:#fff !important; }}

/* Text inputs */
input, textarea, [data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {{
    background: {BG_CARD} !important;
    color: {TEXT_MAIN} !important;
    border-color: {BORDER} !important;
    caret-color: {TEXT_MAIN};
}}
input::placeholder, textarea::placeholder {{
    color: {TEXT_MUTED} !important;
    opacity: 1;
}}

/* Selectbox */
[data-baseweb="select"] > div {{
    background: {BG_CARD} !important;
    border-color: {BORDER} !important;
    color: {TEXT_MAIN} !important;
}}
[data-baseweb="select"] span,
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p {{
    color: {TEXT_MAIN} !important;
}}
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] li {{
    background: {BG_CARD} !important;
    color: {TEXT_MAIN} !important;
}}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] li:hover {{
    background: {ACCENT_SOFT} !important;
}}

/* Multiselect tags */
[data-baseweb="tag"] {{
    background: {ACCENT_SOFT} !important;
    color: #ffffff !important;
}}
[data-baseweb="tag"] span {{ color: #ffffff !important; }}

/* Pills (tab navigation) */
[data-testid="stPills"] button,
[data-testid="stButtonGroup"] button,
[data-testid="stSegmentedControl"] button {{
    background: {BG_CARD} !important;
    background-color: {BG_CARD} !important;
    color: {TEXT_MUTED} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
}}
[data-testid="stPills"] button:hover,
[data-testid="stButtonGroup"] button:hover,
[data-testid="stSegmentedControl"] button:hover {{
    background: {ACCENT_SOFT} !important;
    background-color: {ACCENT_SOFT} !important;
    color: #ffffff !important;
    border-color: {ACCENT} !important;
}}
[data-testid="stPills"] button[aria-checked="true"],
[data-testid="stPills"] button[data-selected="true"],
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[data-selected="true"],
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[data-selected="true"] {{
    background: {ACCENT} !important;
    background-color: {ACCENT} !important;
    color: #ffffff !important;
    border-color: {ACCENT} !important;
}}

/* Radio buttons */
[data-testid="stRadio"] label {{ color: {TEXT_MAIN} !important; }}
[data-testid="stRadio"] div[role="radiogroup"] p {{ color: {TEXT_MAIN} !important; }}

/* Checkbox */
[data-testid="stCheckbox"] label {{ color: {TEXT_MAIN} !important; }}

/* Slider */
[data-testid="stSlider"] p {{ color: {TEXT_MAIN} !important; }}

/* Caption */
[data-testid="stCaptionContainer"] p,
small, .caption {{
    color: {TEXT_MUTED} !important;
}}

/* Alert boxes */
[data-testid="stAlert"] {{
    background: {BG_CARD} !important;
    border-color: {BORDER};
    color: {TEXT_MAIN} !important;
}}
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {{
    color: {TEXT_MAIN} !important;
}}

/* Code blocks */
[data-testid="stCode"],
[data-testid="stCode"] pre,
.stCode pre {{
    background: {BG_CARD} !important;
    color: {TEXT_MAIN} !important;
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
[data-testid="stCode"] code,
.stCode code {{
    color: {TEXT_MAIN} !important;
}}

/* Expander */
[data-testid="stExpander"] details {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
}}
[data-testid="stExpander"] details summary {{
    background: {BG_CARD} !important;
    color: {TEXT_MAIN} !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
}}
[data-testid="stExpander"] details summary:hover {{
    background: {BG_PANEL} !important;
}}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary div {{
    color: {TEXT_MAIN} !important;
    background: transparent !important;
}}
[data-testid="stExpander"] > div,
[data-testid="stExpander"] details > div {{
    background: {BG_CARD} !important;
    border-color: {BORDER} !important;
}}
.streamlit-expanderHeader {{
    background: {BG_CARD} !important;
    color: {TEXT_MAIN} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
}}
.streamlit-expanderContent {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-top: none !important;
}}

/* Data editor */
[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT_MAIN};
}}
.dvn-scroller, .dvn-cell-container {{
    color: {TEXT_MAIN} !important;
}}

/* Markdown */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {{
    color: {TEXT_MAIN};
}}

/* Sidebar text */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{
    color: {TEXT_MAIN};
}}
[data-testid="stSidebar"] [data-baseweb="tag"] span {{ color: #ffffff !important; }}

/* Sidebar nav radio pills */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {{
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 4px 0 10px 0;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    display: flex !important;
    align-items: center;
    padding: 8px 14px !important;
    border-radius: 8px !important;
    cursor: pointer;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: {TEXT_MUTED} !important;
    transition: background 0.15s, color 0.15s;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background: {ACCENT_SOFT} !important;
    color: #ffffff !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] ~ div label {{
    background: {ACCENT_SOFT} !important;
    color: {TEXT_MAIN} !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] {{
    display: none !important;
}}
</style>
"""
