"""Persist UI state to disk so it survives Streamlit page refreshes."""
from __future__ import annotations

import json

from .constants import PROJECT_ROOT

STATE_DIR     = PROJECT_ROOT / "state"
STATE_FILE    = STATE_DIR / "ui_state.json"
DMG_LOG_FILE  = STATE_DIR / "dmg_log.txt"
COST_LOG_FILE = STATE_DIR / "cost_log.txt"

STATE_DIR.mkdir(exist_ok=True)

# Session-state keys persisted in JSON (logs are saved as separate text files)
_JSON_KEYS = [
    "active_page",
    "active_run_id",
    "base_model",
    "java_heap",
    "dmg_status",
    "cost_status",
    "dmg_path_history",
    "cost_path_history",
    "inactivity_timeout_min",
]


def save_ui_state(ss) -> None:
    """Write relevant session-state keys and log files to disk.

    Runs at the end of every rerun, so it skips writes when nothing changed
    (hashes stashed in session state) to avoid 3 disk writes per interaction.
    """
    STATE_DIR.mkdir(exist_ok=True)

    payload = json.dumps({k: ss[k] for k in _JSON_KEYS if k in ss}, indent=2)
    if ss.get("_ui_state_hash") != hash(payload):
        try:
            STATE_FILE.write_text(payload, encoding="utf-8")
            ss["_ui_state_hash"] = hash(payload)
        except Exception:
            pass

    for key, file, hkey in (("dmg_log", DMG_LOG_FILE, "_dmg_log_hash"),
                            ("cost_log", COST_LOG_FILE, "_cost_log_hash")):
        if key not in ss:
            continue
        text = ss[key] or ""
        if ss.get(hkey) != hash(text):
            try:
                file.write_text(text, encoding="utf-8")
                ss[hkey] = hash(text)
            except Exception:
                pass


def load_ui_state() -> dict:
    """Return previously persisted state; empty dict if nothing saved yet."""
    result: dict = {}
    if STATE_FILE.exists():
        try:
            result = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    if DMG_LOG_FILE.exists():
        try:
            result["dmg_log"] = DMG_LOG_FILE.read_text(encoding="utf-8")
        except Exception:
            pass
    if COST_LOG_FILE.exists():
        try:
            result["cost_log"] = COST_LOG_FILE.read_text(encoding="utf-8")
        except Exception:
            pass
    return result
