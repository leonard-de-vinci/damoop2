"""Shared Plotly chart helpers."""
from __future__ import annotations

import plotly.graph_objects as go


def apply_log_axis_readability(fig: go.Figure, *, x: bool = False, y: bool = False) -> None:
    """Make Plotly log axes easier to read with enough ticks across all magnitudes."""
    axis_kwargs = dict(
        nticks=10,
        showgrid=False,
        minor=dict(showgrid=False),
        exponentformat="power",
        showexponent="all",
        separatethousands=True,
    )
    if x:
        fig.update_xaxes(**axis_kwargs)
    if y:
        fig.update_yaxes(**axis_kwargs)
