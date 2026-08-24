"""Input Configuration page — edit queries, config, schema, keys, and profiles."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from lib.constants import (
    CONF_DIR,
    FILE_CONFIG,
    FILE_INDEX,
    FILE_KEYS,
    FILE_QUERIES_COST,
    FILE_QUERIES_DMG,
    INPUT_DIR,
    OUTPUT_DIR,
    OUTPUT_DMS_DIR,
    PROFILES_DIR,
)
from lib.data_loading import (
    count_dms,
    read_file,
    write_file,
)
from lib.ui_helpers import persistent_tabs


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------
_PROFILE_FILES = ["config.txt", "keys.txt", "index.txt", "queries.txt", "queries_dmg.txt"]
# CostCalculation.jar reads these from conf/, not Input/. Profile activation must
# sync them or cost output ignores the profile's selectivity/index/config.
_CONF_FILES = ["config.txt", "keys.txt", "index.txt", "queries.txt"]


def _list_profiles() -> list[str]:
    """Return sorted list of profile names (e.g. ['A', 'B', 'C'])."""
    if not PROFILES_DIR.is_dir():
        return []
    return sorted(
        d.name.replace("profile_", "")
        for d in PROFILES_DIR.iterdir()
        if d.is_dir() and d.name.startswith("profile_")
    )


def _profile_dir(name: str) -> Path:
    return PROFILES_DIR / f"profile_{name}"


def _activate_profile(name: str) -> list[str]:
    """Copy profile files into Input/ (for DMG) and conf/ (for Cost JAR).

    CostCalculation.jar reads keys/index/config/queries from conf/, so without
    the conf/ copy the cost output ignores the activated profile entirely.
    """
    src_dir = _profile_dir(name)
    copied = []
    for fname in _PROFILE_FILES:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, INPUT_DIR / fname)
            copied.append(fname)
    CONF_DIR.mkdir(exist_ok=True)
    for fname in _CONF_FILES:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, CONF_DIR / fname)
    return copied


def _save_current_as_profile(name: str) -> list[str]:
    """Save current Input files as a named profile. Returns list of saved filenames."""
    dst_dir = _profile_dir(name)
    dst_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fname in _PROFILE_FILES:
        src = INPUT_DIR / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)
            saved.append(fname)
    return saved


def _archive_output(name: str) -> Path:
    """Copy Output/GeneratedDMs/ to Output/profile_<name>/. Returns dest path."""
    dest = OUTPUT_DIR / f"profile_{name}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(OUTPUT_DMS_DIR, dest)
    return dest


def _count_archive_files(root: Path, pattern: str) -> int:
    """Cached rglob count for archived profile trees.

    Keyed on the archive's top-level mtime: archives are only ever replaced
    wholesale (rmtree + copytree in _archive_output), so the top dir mtime
    changes exactly when the contents do. Avoids walking ~135-model trees on
    every render of the Profiles tab."""
    if not root.is_dir():
        return 0
    return _count_archive_files_cached(str(root), pattern, root.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _count_archive_files_cached(root_str: str, pattern: str, _mtime: float) -> int:
    return len(list(Path(root_str).rglob(pattern)))


def _list_archived_profiles() -> list[str]:
    """Return profile names that have archived outputs."""
    if not OUTPUT_DIR.is_dir():
        return []
    return sorted(
        d.name.replace("profile_", "")
        for d in OUTPUT_DIR.iterdir()
        if d.is_dir() and d.name.startswith("profile_")
    )


def _profile_summary(name: str) -> dict:
    """Read key metadata from a profile's files for display."""
    pdir = _profile_dir(name)
    if not pdir.is_dir():
        return {"exists": False, "files": []}
    # Delegate to the cached version keyed on directory mtime
    mtime = max(
        (os.path.getmtime(pdir / f) for f in _PROFILE_FILES if (pdir / f).exists()),
        default=0.0,
    )
    return _profile_summary_cached(name, mtime)


@st.cache_data(show_spinner=False)
def _profile_summary_cached(name: str, _mtime: float) -> dict:
    """Cached profile metadata (invalidated when files change)."""
    pdir = _profile_dir(name)
    info: dict = {"exists": pdir.is_dir(), "files": []}
    for fname in _PROFILE_FILES:
        f = pdir / fname
        info["files"].append(fname if f.exists() else f"({fname} missing)")
    # Parse selectivities from keys.txt
    keys_f = pdir / "keys.txt"
    if keys_f.exists():
        sels = {}
        for line in keys_f.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 3:
                attr = parts[0].strip()
                card_part = parts[2].strip().split("/")[0]
                try:
                    sels[attr] = int(card_part)
                except ValueError:
                    pass
        info["selectivities"] = sels
    # Parse table volumes from config.txt
    cfg_f = pdir / "config.txt"
    if cfg_f.exists():
        vols = {}
        in_data = False
        for line in cfg_f.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip() == "Data":
                in_data = True
            elif in_data and line.strip():
                parts = line.split()
                if parts:
                    vols[parts[0]] = " ".join(parts[1:])
        info["volumes"] = vols
    # Parse index info from index.txt
    idx_f = pdir / "index.txt"
    if idx_f.exists():
        no_idx = []
        for line in idx_f.read_text(encoding="utf-8", errors="ignore").splitlines():
            tokens = [t.strip() for t in line.strip().split(",")]
            if len(tokens) < 6:
                continue
            # tokens[0] = table name, then groups of 5: attr, asc_flag, asc_lvl, desc_flag, desc_lvl
            i = 1
            while i + 4 < len(tokens):
                attr = tokens[i]
                asc_flag = tokens[i + 1].lower()
                desc_flag = tokens[i + 3].lower()
                if asc_flag == "no" and desc_flag == "no":
                    no_idx.append(attr)
                i += 5  # skip to next attribute group
        info["unindexed"] = no_idx
    # Parse query count from queries.txt
    q_f = pdir / "queries.txt"
    if q_f.exists():
        n_queries = sum(
            1 for line in q_f.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip() and not line.strip().startswith("query/")
        )
        info["n_queries"] = n_queries
    return info


def render() -> None:
    _input_tab = persistent_tabs(
        ["Profiles", "DMG Queries", "Cost Queries", "Config", "Schema (Index)", "Keys"],
        "input_tab",
    )

    # ── Tab 0: Profiles ──────────────────────────────────────────────────
    if _input_tab == "Profiles":
        _render_profiles_tab()

    # ── Tab 1: DMG queries ───────────────────────────────────────────────
    if _input_tab == "DMG Queries":
        st.markdown('<div class="section-heading">queries_dmg.txt — Field Groups per Query</div>', unsafe_allow_html=True)
        st.caption("One query per line. Each line is a comma-separated list of field names involved in that query. "
                   "This file is passed to DMgeneration.jar.")

        raw_dmg = read_file(FILE_QUERIES_DMG)
        rows_dmg = [line.strip() for line in raw_dmg.strip().splitlines() if line.strip()]

        updated_rows: list[str] = []
        for idx, row in enumerate(rows_dmg):
            edited = st.text_input(
                f"Query {idx + 1}",
                value=row,
                key=f"dmg_q_{idx}",
                help="Comma-separated field names",
            )
            updated_rows.append(edited)

        nc1, nc2 = st.columns([1, 5])
        if nc1.button("Add Query", key="dmg_add"):
            updated_rows.append("")
            st.rerun()

        new_content_dmg = "\n".join(updated_rows) + "\n"
        if st.button("Save queries_dmg.txt", key="save_dmg"):
            write_file(FILE_QUERIES_DMG, new_content_dmg)
            st.success("Saved.")

        with st.expander("Raw file content"):
            raw_edit = st.text_area("Direct edit", value=read_file(FILE_QUERIES_DMG), height=200, key="dmg_raw")
            if st.button("Save raw", key="save_dmg_raw"):
                write_file(FILE_QUERIES_DMG, raw_edit)
                st.success("Saved.")

    # ── Tab 2: Cost queries ───────────────────────────────────────────────
    if _input_tab == "Cost Queries":
        st.markdown('<div class="section-heading">queries.txt — Annotated Queries for Cost Calculation</div>', unsafe_allow_html=True)
        st.caption("Format: `query_definition / result_size / frequency / max_time (s)`  "
                   "The τ threshold (max_time) is the per-query time constraint in seconds. "
                   "The first line is a header and must remain unchanged.")

        raw_cost = read_file(FILE_QUERIES_COST)
        lines_cost = raw_cost.strip().splitlines()
        header     = lines_cost[0] if lines_cost else "query/size/frequency/max_time in s"
        data_lines = lines_cost[1:] if len(lines_cost) > 1 else []

        st.caption(f"Header: `{header}`")

        table_rows: list[dict] = []
        for line in data_lines:
            if not line.strip():
                continue
            parts = line.split("/")
            table_rows.append({
                "Query Definition": parts[0].strip() if len(parts) > 0 else "",
                "Result Size (bytes)": parts[1].strip() if len(parts) > 1 else "",
                "Frequency":        parts[2].strip() if len(parts) > 2 else "",
                "Max Time (s)":     parts[3].strip() if len(parts) > 3 else "",
            })

        edited_df = st.data_editor(
            pd.DataFrame(table_rows),
            width="stretch",
            num_rows="dynamic",
            key="cost_q_table",
            column_config={
                "Query Definition": st.column_config.TextColumn("Query Definition", width="large"),
                "Result Size (bytes)": st.column_config.NumberColumn("Result Size (bytes)", min_value=0),
                "Frequency":        st.column_config.NumberColumn("Frequency",    min_value=0),
                "Max Time (s)":     st.column_config.NumberColumn("Max Time (s)", min_value=0),
            },
        )

        if st.button("Save queries.txt", key="save_cost_q"):
            new_lines = [header]
            for _, row in edited_df.iterrows():
                q    = str(row.get("Query Definition", "")).strip()
                sz   = str(int(row.get("Result Size (bytes)", 0) or 0))
                freq = str(int(row.get("Frequency",   0) or 0))
                mt   = str(int(row.get("Max Time (s)", 0) or 0))
                if q:
                    new_lines.append(f"{q}/{sz}/{freq}/{mt}")
            write_file(FILE_QUERIES_COST, "\n".join(new_lines) + "\n")
            st.success("Saved.")

    # ── Tab 3: Config ─────────────────────────────────────────────────────
    if _input_tab == "Config":
        st.markdown('<div class="section-heading">config.txt — Server Counts and Data Volumes</div>', unsafe_allow_html=True)

        raw_cfg = read_file(FILE_CONFIG)
        cfg_lines = raw_cfg.strip().splitlines()

        server_line = ""
        table_volumes: dict[str, str] = {}
        in_data = False
        for line in cfg_lines:
            if line.strip() == "Servers":
                continue
            elif line.startswith("nbrServers"):
                server_line = line
            elif line.strip() == "Data":
                in_data = True
            elif in_data and line.strip():
                parts = line.split()
                if parts:
                    table_volumes[parts[0]] = "\t".join(parts[1:])

        st.markdown("**Server Counts**")
        st.caption("Space- or tab-separated list of server counts to test")
        current_servers = server_line.replace("nbrServers", "").strip().replace("\t", "  ")
        new_servers_str = st.text_input(
            "Server counts",
            value=current_servers,
            help="e.g.  10  50  100  500  1000  5000  10000",
            key="cfg_servers",
        )

        st.markdown("**Data Volumes per Table**")
        st.caption(
            "Each row defines a table name and its volume value(s) (space-separated). "
            "Warehouse takes two values: the minimum and maximum warehouse count (e.g. 1000 → 100000000). "
            "Customer and Order each take a single scaling factor relative to the warehouse count."
        )
        vol_rows = [{"Table": t, "Volume Values": v.replace("\t", " ")} for t, v in table_volumes.items()]
        edited_vol = st.data_editor(
            pd.DataFrame(vol_rows),
            width="stretch",
            num_rows="dynamic",
            key="cfg_vol_table",
        )

        if st.button("Save config.txt", key="save_cfg"):
            lines_out = ["Servers", f"nbrServers\t{new_servers_str.strip()}", "Data"]
            for _, row in edited_vol.iterrows():
                t = str(row.get("Table", "")).strip()
                v = str(row.get("Volume Values", "")).strip()
                if t:
                    lines_out.append(f"{t}\t{v}")
            write_file(FILE_CONFIG, "\n".join(lines_out) + "\n")
            st.success("Saved.")

    # ── Tab 4: Index ──────────────────────────────────────────────────────
    if _input_tab == "Schema (Index)":
        st.markdown('<div class="section-heading">index.txt — Attribute Index Configuration</div>', unsafe_allow_html=True)
        st.caption(
            "Format: `TableName, AttrName, asc_indexed, asc_level, desc_indexed, desc_level [, AttrName2, ...]`  "
            "Each attribute has 4 values: ascending-indexed (yes/no), asc-level, descending-indexed (yes/no), desc-level. "
            "One row per table."
        )

        raw_idx = read_file(FILE_INDEX)
        idx_rows = []
        for line in raw_idx.strip().splitlines():
            if line.strip():
                idx_rows.append({"Row": line.strip()})

        edited_idx = st.data_editor(
            pd.DataFrame(idx_rows),
            width="stretch",
            num_rows="dynamic",
            key="idx_table",
            column_config={"Row": st.column_config.TextColumn("Row", width="large")},
        )

        if st.button("Save index.txt", key="save_idx"):
            content = "\n".join(str(r.get("Row", "")).strip() for _, r in edited_idx.iterrows() if str(r.get("Row", "")).strip())
            write_file(FILE_INDEX, content + "\n")
            st.success("Saved.")

    # ── Tab 5: Keys ───────────────────────────────────────────────────────
    if _input_tab == "Keys":
        st.markdown('<div class="section-heading">keys.txt — Attribute Metadata</div>', unsafe_allow_html=True)
        st.caption("Format: `attribute, byte_size, cardinality/table, type (pk|fk|null)`")

        raw_keys = read_file(FILE_KEYS)
        keys_rows = []
        for line in raw_keys.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            keys_rows.append({
                "Attribute":          parts[0] if len(parts) > 0 else "",
                "Byte Size":          parts[1] if len(parts) > 1 else "",
                "Cardinality/Table":  parts[2] if len(parts) > 2 else "",
                "Type":               parts[3] if len(parts) > 3 else "",
            })

        edited_keys = st.data_editor(
            pd.DataFrame(keys_rows),
            width="stretch",
            num_rows="dynamic",
            key="keys_table",
            column_config={
                "Type": st.column_config.SelectboxColumn("Type", options=["pk", "fk", "null"]),
            },
        )

        if st.button("Save keys.txt", key="save_keys"):
            lines_out = []
            for _, row in edited_keys.iterrows():
                attr = str(row.get("Attribute",         "")).strip()
                sz   = str(row.get("Byte Size",         "")).strip()
                card = str(row.get("Cardinality/Table", "")).strip()
                typ  = str(row.get("Type",              "")).strip()
                if attr:
                    lines_out.append(f"{attr},{sz},{card},{typ}")
            write_file(FILE_KEYS, "\n".join(lines_out) + "\n")
            st.success("Saved.")




# ---------------------------------------------------------------------------
# Profiles tab
# ---------------------------------------------------------------------------

_PROFILE_DISPLAY_NAMES: dict[str, str] = {
    "TPC_C_ORIGINAL": "TPC-C Original (paper's exact 5-query, 10-key setup)",
    "A": "A — Extended Baseline (paper infrastructure + 3 added attributes)",
    "B": "B — Degraded Selectivity",
    "C": "C — Sparse Indexes",
}

@st.fragment
def _render_profiles_tab() -> None:
    st.markdown(
        '<div class="section-heading">Configuration Profiles</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Each profile stores a complete set of input files "
        "(**config.txt**, **keys.txt**, **index.txt**, **queries.txt**, **queries_dmg.txt**) "
        "with different selectivities, table volumes, index configurations, and queries. "
        "Activate a profile to copy its files into `Input/`, then run the pipeline. "
        "Archive the output before switching to the next profile."
    )

    profiles = _list_profiles()
    archived = _list_archived_profiles()

    # ── Active profile indicator ─────────────────────────────────────────
    active = st.session_state.get("active_profile", "—")
    active_label = _PROFILE_DISPLAY_NAMES.get(active, active)
    st.markdown(f"**Active profile:** `{active_label}`")

    if not profiles:
        st.info("No profiles found in `Input/profiles/`. Create one below.")

    # ── Profile cards ────────────────────────────────────────────────────
    for pname in profiles:
        info = _profile_summary(pname)
        is_archived = pname in archived
        archive_dir = OUTPUT_DIR / f"profile_{pname}"
        n_costs = _count_archive_files(archive_dir, "costs.txt") if is_archived else 0

        display_name = _PROFILE_DISPLAY_NAMES.get(pname, f"Profile {pname}")
        with st.expander(
            display_name
            + (" ✓ archived" if is_archived else "")
            + (f" — {n_costs} cost files" if n_costs else ""),
            expanded=(active == pname),
        ):
            # Summary info
            if info.get("selectivities"):
                sels = info["selectivities"]
                # Show only non-PK/FK columns (selectivity > 1 or explicitly interesting)
                display_sels = {k: v for k, v in sels.items() if v > 1}
                if display_sels:
                    sel_str = " · ".join(f"`{k}`={v}" for k, v in sorted(display_sels.items()))
                    st.markdown(f"**Selectivities:** {sel_str}")

            if info.get("volumes"):
                vol_str = " · ".join(f"`{k}` {v}" for k, v in info["volumes"].items())
                st.markdown(f"**Table volumes:** {vol_str}")

            if info.get("unindexed"):
                st.markdown(f"**Unindexed columns:** {', '.join(f'`{c}`' for c in info['unindexed'])}")
            else:
                st.markdown("**Indexes:** All columns indexed")

            if info.get("n_queries"):
                n_keys = len(info.get("selectivities", {}))
                st.markdown(f"**Queries:** {info['n_queries']} · **Keys:** {n_keys}")

            st.markdown(f"**Files:** {', '.join(info.get('files', []))}")

            # Action buttons
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                if st.button(
                    f"Activate Profile {pname}",
                    key=f"activate_{pname}",
                    width="stretch",
                ):
                    copied = _activate_profile(pname)
                    st.session_state["active_profile"] = pname
                    st.success(f"Profile {pname} activated — copied {', '.join(copied)} into Input/")
                    st.rerun()

            with ac2:
                has_output = count_dms(OUTPUT_DMS_DIR) > 0
                if st.button(
                    f"Archive Output as {pname}",
                    key=f"archive_{pname}",
                    disabled=not has_output,
                    width="stretch",
                ):
                    dest = _archive_output(pname)
                    st.success(f"Archived Output/GeneratedDMs/ → {dest.relative_to(OUTPUT_DIR)}/")
                    st.rerun(scope="fragment")

            with ac3:
                if st.button(
                    f"Update Profile {pname} from Current",
                    key=f"update_{pname}",
                    width="stretch",
                ):
                    saved = _save_current_as_profile(pname)
                    _profile_summary_cached.clear()
                    st.success(f"Saved current Input files to profile {pname}: {', '.join(saved)}")
                    st.rerun(scope="fragment")

            # Show file contents in sub-expander
            with st.expander(f"View Profile {pname} files", expanded=False):
                pdir = _profile_dir(pname)
                for fname in _PROFILE_FILES:
                    f = pdir / fname
                    if f.exists():
                        st.markdown(f"**{fname}**")
                        st.code(f.read_text(encoding="utf-8", errors="ignore"), language="text")

    # ── Create new profile ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Create New Profile")
    st.caption(
        "Save the current `Input/` config, keys, and index as a new named profile. "
        "You can then edit the Input files on the other tabs and save again as a different profile."
    )
    nc1, nc2 = st.columns([1, 3])
    with nc1:
        new_name = st.text_input(
            "Profile name",
            value="",
            placeholder="e.g. D",
            key="new_profile_name",
            help="Short name like A, B, C, D…",
            max_chars=20,
        )
    with nc2:
        st.markdown("")  # spacer
        st.markdown("")
        if st.button(
            "Save Current Input as Profile",
            key="create_profile_btn",
            disabled=not new_name.strip(),
            width="stretch",
        ):
            name = new_name.strip().upper()
            saved = _save_current_as_profile(name)
            _profile_summary_cached.clear()
            st.success(f"Created profile {name} with {len(saved)} file(s): {', '.join(saved)}")
            st.rerun(scope="fragment")

    # ── Archived outputs summary ─────────────────────────────────────────
    if archived:
        st.markdown("---")
        st.markdown("#### Archived Pipeline Outputs")
        st.caption("These are previously archived `Output/GeneratedDMs/` snapshots per profile.")
        for aname in archived:
            adir = OUTPUT_DIR / f"profile_{aname}"
            n_costs = _count_archive_files(adir, "costs.txt")
            n_xml = _count_archive_files(adir, "*.xml")
            st.markdown(
                f"- **Profile {aname}**: {n_costs} cost files, {n_xml} XML models "
                f"(`Output/profile_{aname}/`)"
            )
