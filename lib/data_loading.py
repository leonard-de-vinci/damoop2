"""File I/O, data parsing and snapshot management."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .constants import (
    DEFAULT_QUERY_FREQUENCIES,
    INPUT_DIR,
    INPUT_BACKUP_DIR,
    OUTPUT_DIR,
    OUTPUT_BACKUP_DIR,
    HISTORY_DIR,
    METRIC_OPTIONS,
    OUTPUT_DMS_DIR,
    PROFILES_DIR,
    RUN_METADATA_FILE,
    RUN_QUERIES_FILE,
    RUNS_DIR,
)

from .parse_costs import build_costs_dataframe


_PROFILE_FILES = ["config.txt", "keys.txt", "index.txt", "queries.txt", "queries_dmg.txt"]


def detect_active_profile() -> str | None:
    """Return the profile name whose stored files exactly match the live Input/ files.

    Returns None when no profile matches (Input/ has been edited directly, so the
    current configuration is a 'custom' state that belongs to no saved profile).

    Cached on the live Input/ file mtimes so the byte-compare (run in the sidebar on
    every rerun) only re-executes after a profile switch or an Input edit.
    """
    if not PROFILES_DIR.is_dir():
        return None
    key = tuple(
        (INPUT_DIR / f).stat().st_mtime if (INPUT_DIR / f).exists() else 0.0
        for f in _PROFILE_FILES
    )
    return _detect_active_profile_cached(key)


@st.cache_data(show_spinner=False)
def _detect_active_profile_cached(_input_mtimes: tuple) -> str | None:
    for pdir in sorted(PROFILES_DIR.iterdir()):
        if not pdir.is_dir() or not pdir.name.startswith("profile_"):
            continue
        profile_files = [f for f in _PROFILE_FILES if (pdir / f).exists()]
        if not profile_files:
            continue
        if all(
            (INPUT_DIR / f).exists()
            and (pdir / f).read_bytes() == (INPUT_DIR / f).read_bytes()
            for f in profile_files
        ):
            return pdir.name.replace("profile_", "")
    return None


# ---------------------------------------------------------------------------
# Simple file helpers
# ---------------------------------------------------------------------------
def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def parse_query_frequencies(queries_file: Path | None = None) -> dict[str, int]:
    """Read query execution frequencies from Input/queries.txt.

    Format of each data line: fields.../size/frequency/max_time
    Frequency is the second-to-last slash-delimited token.

    Returns DEFAULT_QUERY_FREQUENCIES on any parse error or missing file,
    so callers never need to guard against None.
    """
    path = queries_file or (INPUT_DIR / "queries.txt")
    if not path.exists():
        return dict(DEFAULT_QUERY_FREQUENCIES)
    mtime = os.path.getmtime(path)
    return _parse_query_frequencies_cached(str(path), mtime)


@st.cache_data(show_spinner=False)
def _parse_query_frequencies_cached(path_str: str, _mtime: float) -> dict[str, int]:
    """Cached query frequency parsing (invalidated when file changes)."""
    path = Path(path_str)
    try:
        freqs: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("query/"):
                continue
            parts = line.split("/")
            if len(parts) >= 3:
                idx = len(freqs) + 1
                freqs[f"q{idx}"] = int(parts[-2])
        if freqs:
            return freqs
    except Exception:
        pass
    return dict(DEFAULT_QUERY_FREQUENCIES)


def get_query_cost_stems(queries_file: Path | None = None) -> list[str]:
    """Return query cost column stems based on the current queries.txt.

    Derives stems from ``parse_query_frequencies`` so caching and
    profile-switch invalidation are handled automatically.
    """
    freqs = parse_query_frequencies(queries_file)
    return list(freqs.keys())


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Backup / Restore helpers
# ---------------------------------------------------------------------------
_INPUT_BACKUP_FILENAMES = [
    "config.txt",
    "index.txt",
    "keys.txt",
    "queries.txt",
    "queries_dmg.txt",
    "signatures_TPC_C.txt",
]


def backup_input() -> list[str]:
    """Copy each known Input file into Input/backup_original/.
    Returns list of filenames that were backed up."""
    INPUT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for fname in _INPUT_BACKUP_FILENAMES:
        src = INPUT_DIR / fname
        if src.exists():
            shutil.copy2(src, INPUT_BACKUP_DIR / fname)
            copied.append(fname)
    return copied


def restore_input() -> list[str]:
    """Copy each file from Input/backup_original/ back into Input/.
    Returns list of filenames that were restored."""
    restored: list[str] = []
    for fname in _INPUT_BACKUP_FILENAMES:
        src = INPUT_BACKUP_DIR / fname
        if src.exists():
            shutil.copy2(src, INPUT_DIR / fname)
            restored.append(fname)
    return restored


def backup_output() -> int:
    """Mirror Output/GeneratedDMs and Output/signatures_TPC_C.txt into
    Output/backup_original/.  Returns count of files copied."""
    OUTPUT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    # signatures file
    sig_src = OUTPUT_DIR / "signatures_TPC_C.txt"
    if sig_src.exists():
        shutil.copy2(sig_src, OUTPUT_BACKUP_DIR / "signatures_TPC_C.txt")
        count += 1

    # GeneratedDMs tree
    gdms_backup = OUTPUT_BACKUP_DIR / "GeneratedDMs"
    if OUTPUT_DMS_DIR.exists():
        if gdms_backup.exists():
            shutil.rmtree(gdms_backup)
        shutil.copytree(OUTPUT_DMS_DIR, gdms_backup)
        count += sum(1 for _ in gdms_backup.rglob("*") if _.is_file())

    return count


def restore_output() -> int:
    """Copy files from Output/backup_original/ back into Output/.
    Returns count of files restored."""
    count = 0
    # signatures file
    sig_src = OUTPUT_BACKUP_DIR / "signatures_TPC_C.txt"
    if sig_src.exists():
        shutil.copy2(sig_src, OUTPUT_DIR / "signatures_TPC_C.txt")
        count += 1

    # GeneratedDMs tree
    gdms_backup = OUTPUT_BACKUP_DIR / "GeneratedDMs"
    if gdms_backup.exists():
        if OUTPUT_DMS_DIR.exists():
            shutil.rmtree(OUTPUT_DMS_DIR)
        shutil.copytree(gdms_backup, OUTPUT_DMS_DIR)
        count += sum(1 for _ in OUTPUT_DMS_DIR.rglob("*") if _.is_file())

    return count


def input_backup_exists() -> bool:
    return any((INPUT_BACKUP_DIR / f).exists() for f in _INPUT_BACKUP_FILENAMES)


def output_backup_exists() -> bool:
    return (OUTPUT_BACKUP_DIR / "GeneratedDMs").exists() or (OUTPUT_BACKUP_DIR / "signatures_TPC_C.txt").exists()


def _sha256_of_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def list_run_dirs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in RUNS_DIR.iterdir():
        if d.is_dir() and (d / "GeneratedDMs").exists():
            runs.append(d)
    return sorted(runs, reverse=True)


def list_run_ids() -> list[str]:
    return [p.name for p in list_run_dirs()]


def get_run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def get_active_run_id() -> str:
    return st.session_state.get("active_run_id", "LIVE")


def get_active_dms_dir() -> Path:
    run_id = get_active_run_id()
    if run_id == "LIVE":
        return OUTPUT_DMS_DIR
    run_dir = get_run_dir(run_id)
    dms_dir = run_dir / "GeneratedDMs"
    if dms_dir.exists():
        return dms_dir
    st.session_state["active_run_id"] = "LIVE"
    return OUTPUT_DMS_DIR


def count_dms(root_dir: Path | None = None) -> int:
    target = root_dir or get_active_dms_dir()
    if not target.exists():
        return 0
    # Cached on the dir mtime so the rglob tree-walk (sidebar, every rerun) only
    # re-runs when the output dir actually changes (a pipeline run recreates it).
    return _count_dms_cached(str(target), target.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _count_dms_cached(target_str: str, _mtime: float) -> int:
    return len(list(Path(target_str).rglob("costs.txt")))


def read_run_metadata(run_id: str) -> dict[str, Any] | None:
    if run_id == "LIVE":
        return None
    meta_file = get_run_dir(run_id) / RUN_METADATA_FILE
    if not meta_file.exists():
        return None
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def active_run_query_mismatch() -> tuple[bool, str]:
    run_id = get_active_run_id()
    if run_id == "LIVE":
        return False, ""
    meta = read_run_metadata(run_id)
    if not meta:
        return False, ""

    current_hash = _sha256_of_file(INPUT_DIR / "queries.txt")
    selected_hash = str(meta.get("queries_sha256", ""))
    if current_hash and selected_hash and current_hash != selected_hash:
        ts = str(meta.get("created_at", ""))
        return True, f"Selected run '{run_id}' was created with a different queries.txt (snapshot time: {ts})."
    return False, ""


def create_run_snapshot_from_live(base_model: str) -> Path | None:
    if not OUTPUT_DMS_DIR.exists() or not any(OUTPUT_DMS_DIR.rglob("costs.txt")):
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = ts
    suffix = 1
    while (RUNS_DIR / run_id).exists():
        suffix += 1
        run_id = f"{ts}_{suffix}"

    run_dir = RUNS_DIR / run_id
    dms_out = run_dir / "GeneratedDMs"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(OUTPUT_DMS_DIR, dms_out)

    queries_src = INPUT_DIR / "queries.txt"
    queries_hash = _sha256_of_file(queries_src)
    if queries_src.exists():
        shutil.copy2(queries_src, run_dir / RUN_QUERIES_FILE)

    meta = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "base_model": base_model,
        "profile": detect_active_profile(),
        "queries_file": RUN_QUERIES_FILE,
        "queries_sha256": queries_hash,
        "cost_files": len(list(dms_out.rglob("costs.txt"))),
        "source": "pipeline_phase2",
    }
    (run_dir / RUN_METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
def paper_model_order() -> list[str]:
    """Read signatures_TPC_C.txt and return paths in the paper's model order.

    Handles two path formats produced by different pipeline versions:
      - ``Output/GeneratedDMs/path/`` (original format, with leading Output/)
      - ``GeneratedDMs/path``         (new pipeline format, no Output/ prefix)
    In both cases the root model (empty suffix) maps to ``TPC_C.xml``.
    """
    for name in ("signatures_TPC_C.txt", "signatures_TPCC.txt"):
        sig_file = INPUT_DIR / name
        if sig_file.exists():
            # Cached on mtime: make_label_map runs on every rerun of several
            # pages, so avoid re-reading/re-parsing the file each time.
            return _paper_model_order_cached(str(sig_file), os.path.getmtime(sig_file))
    return []


@st.cache_data(show_spinner=False)
def _paper_model_order_cached(sig_path: str, _mtime: float) -> list[str]:
    order: list[str] = []
    for raw in Path(sig_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("Output/GeneratedDMs/"):
            rel = line.removeprefix("Output/GeneratedDMs/").rstrip("/")
            order.append(rel if rel else "TPC_C.xml")
        elif line.startswith("GeneratedDMs"):
            rel = line[len("GeneratedDMs"):].lstrip("/").rstrip("/")
            order.append(rel if rel else "TPC_C.xml")
    return order


def make_label_map(xml_files: pd.Series) -> dict[str, str]:
    """Return a stable {full_path: 'M0'} mapping using the paper's numbering."""
    paper_order = paper_model_order()
    if paper_order:
        label_map = {path: f"M{i}" for i, path in enumerate(paper_order)}
        known = set(paper_order)
        extras = sorted(v for v in xml_files.dropna().unique() if v not in known)
        offset = len(paper_order)
        for j, v in enumerate(extras):
            label_map[v] = f"M{offset + j}"
        return label_map
    unique = sorted(xml_files.dropna().unique().tolist())
    return {v: f"M{i}" for i, v in enumerate(unique)}


def tidy(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    df = df.copy()
    if df.empty or "xml_file" not in df.columns:
        # No parseable cost rows (e.g. empty/corrupt costs.txt) — return an empty frame
        # with the expected schema so callers show "no data" instead of crashing.
        return pd.DataFrame(columns=["xml_file", "mode", "source"])
    df["xml_file"] = df["xml_file"].fillna("unknown")
    df["mode"]     = df["mode"].fillna("unknown")
    for col in ["servers", "data_volume", "data_model_size"] + METRIC_OPTIONS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["source"] = source_label
    return df


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Parsing Output/ directory…")
def load_live_data(_bust: str, dms_dir_str: str, source_label: str = "Live") -> tuple[pd.DataFrame, str]:
    dms_dir = Path(dms_dir_str)
    if not dms_dir.exists() or not any(dms_dir.rglob("costs.txt")):
        return pd.DataFrame(), ""
    df = build_costs_dataframe(dms_dir)
    loaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return tidy(df, source_label), loaded_at


@st.cache_data(show_spinner="Loading snapshot…")
def load_snapshot(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str)
    label = Path(path_str).stem
    return tidy(df, label)


def list_snapshots() -> list[Path]:
    return sorted(HISTORY_DIR.glob("*.csv"), reverse=True)


def save_snapshot(df_live: pd.DataFrame, label: str) -> Path:
    safe = label.strip().replace(" ", "_").replace("/", "-").replace("\\", "-") or "snapshot"
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = HISTORY_DIR / f"{ts}__{safe}.csv"
    df_live.drop(columns=["source"], errors="ignore").to_csv(dest, index=False)
    return dest
