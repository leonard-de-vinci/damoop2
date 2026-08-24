"""Small shared UI helpers."""
from __future__ import annotations

import streamlit as st


def persistent_tabs(labels: list[str], qp_key: str) -> str:
    """Render pill-style tab selector and persist selection via URL query params."""
    saved = st.query_params.get(qp_key, labels[0])
    if saved not in labels:
        saved = labels[0]
    selected = st.pills(
        qp_key, labels, default=saved,
        key=f"pills_{qp_key}", label_visibility="collapsed",
    )
    if selected is None:
        selected = labels[0]
    if selected != saved:
        st.query_params[qp_key] = selected
    return selected
