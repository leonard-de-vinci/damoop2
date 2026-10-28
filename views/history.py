"""History page — snapshot management and comparison."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.constants import (
    LAYOUT_BASE,
    LEGEND_BASE,
    METRIC_OPTIONS,
    PLOTLY_COLORS,
    fmt_metric,
)
from lib.data_loading import list_snapshots, load_snapshot


def render() -> None:
    snapshots = list_snapshots()

    st.markdown('<div class="section-heading">Saved Snapshots</div>', unsafe_allow_html=True)

    if not snapshots:
        st.info("No snapshots saved yet. Run the pipeline and save a snapshot from the Cost Analysis page.")
    else:
        for snap in snapshots:
            sc1, sc2, sc3 = st.columns([4, 1, 1])
            stem_parts = snap.stem.split("__", 1)
            ts_raw     = stem_parts[0] if stem_parts else snap.stem
            label      = stem_parts[1].replace("_", " ") if len(stem_parts) > 1 else snap.stem
            try:
                ts_fmt = datetime.strptime(ts_raw, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                ts_fmt = ts_raw

            sc1.markdown(f"**{label}**  \n`{ts_fmt}` &nbsp; — &nbsp; `{snap.name}`")

            sc2.download_button(
                "Download",
                data=snap.read_bytes(),
                file_name=snap.name,
                mime="text/csv",
                key=f"dl_{snap.name}",
            )

            if sc3.button("Delete", key=f"del_{snap.name}"):
                snap.unlink()
                load_snapshot.clear()
                st.rerun()

            st.divider()

    st.markdown('<div class="section-heading">Compare Two Snapshots</div>', unsafe_allow_html=True)

    if len(snapshots) < 2:
        st.info("Need at least two snapshots to compare.")
    else:
        snap_names = {p.stem: str(p) for p in snapshots}
        cmp1, cmp2 = st.columns(2)
        sel_a = cmp1.selectbox("Snapshot A", list(snap_names.keys()), key="cmp_a")
        sel_b = cmp2.selectbox("Snapshot B", list(snap_names.keys()), index=1, key="cmp_b")

        if sel_a != sel_b:
            df_a = load_snapshot(snap_names[sel_a])
            df_b = load_snapshot(snap_names[sel_b])
            df_both = pd.concat([df_a, df_b], ignore_index=True)

            cmp_metric = st.selectbox("Metric", METRIC_OPTIONS, format_func=fmt_metric, key="cmp_metric")
            cmp_vol    = st.selectbox("Data Volume",
                                      [int(v) for v in sorted(df_both["data_volume"].dropna().unique())],
                                      key="cmp_vol")

            df_plot = (df_both[df_both["data_volume"] == cmp_vol]
                       .groupby(["source", "xml_file", "servers"], as_index=False)[cmp_metric].mean())

            fig_cmp = px.line(df_plot, x="servers", y=cmp_metric, color="xml_file",
                              line_dash="source", markers=True,
                              color_discrete_sequence=PLOTLY_COLORS,
                              labels={"servers": "Server Count", cmp_metric: fmt_metric(cmp_metric),
                                      "xml_file": "Model", "source": "Snapshot"})
            fig_cmp.update_traces(line=dict(width=2), marker=dict(size=6))
            fig_cmp.update_layout(height=500, legend=dict(**LEGEND_BASE, orientation="h", y=-0.28), **LAYOUT_BASE)
            st.plotly_chart(fig_cmp, width="stretch")
