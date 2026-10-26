"""Signature viewer for the DaMoOp generated models. Run: streamlit run signature_viewer.py"""
import html

import streamlit as st

from lib.signatures import Nested, describe, load_models

ENTITY_COLORS = {"Customer": "#CDEFE3", "Order": "#FFE3B8", "Warehouse": "#D6E4FF"}
HEADER = "#1F2A44"


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def fields_table(fields: list, entity: str) -> str:
    color = ENTITY_COLORS.get(entity, "#EEEEEE")
    rows = []
    for f in fields:
        if isinstance(f, Nested):
            inner_color = ENTITY_COLORS.get(f.entity, "#EEEEEE")
            rows.append(
                f'<TR><TD BGCOLOR="{inner_color}" CELLPADDING="4">'
                f'<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" COLOR="#555555">'
                f'<TR><TD BGCOLOR="{inner_color}"><B>{esc(f.entity)}</B></TD></TR>'
                f'{fields_table(f.fields, f.entity)}</TABLE></TD></TR>'
            )
        else:
            rows.append(f'<TR><TD ALIGN="LEFT" BGCOLOR="{color}">{esc(f)}</TD></TR>')
    return "".join(rows)


def model_to_dot(model) -> str:
    lines = [
        "digraph G {",
        'graph [rankdir=LR, bgcolor="white", pad=0.3, nodesep=0.6, ranksep=1.0];',
        'node [shape=plaintext, fontname="Helvetica", fontsize=11];',
        'edge [fontname="Helvetica", fontsize=10, color="#555555", fontcolor="#333333"];',
    ]
    for c in model.collections:
        cols = []
        for i, part in enumerate(c.parts):
            label = f"part {i + 1}" if len(c.parts) > 1 else ""
            head = f'<TR><TD BGCOLOR="#E9ECF2"><FONT POINT-SIZE="9">{label}</FONT></TD></TR>' if label else ""
            cols.append(
                f'<TD VALIGN="TOP"><TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">'
                f"{head}{fields_table(part, c.name)}</TABLE></TD>"
            )
        split_note = f' <FONT POINT-SIZE="9">split in {len(c.parts)}</FONT>' if len(c.parts) > 1 else ""
        table = (
            f'<<TABLE BORDER="2" CELLBORDER="0" CELLSPACING="6" COLOR="{HEADER}">'
            f'<TR><TD COLSPAN="{len(c.parts)}" BGCOLOR="{HEADER}"><FONT COLOR="white"><B>{esc(c.name)}</B>{split_note}</FONT></TD></TR>'
            f'<TR>{"".join(cols)}</TR></TABLE>>'
        )
        lines.append(f'"{c.name}" [label={table}];')

    def owner(key):
        for c in model.collections:
            if any(key in [f for f in p if isinstance(f, str)] for p in c.parts):
                return c.name
        return None

    for r in model.references:
        a, b = owner(r.left), owner(r.right)
        if a and b and a != b:
            lines.append(f'"{a}" -> "{b}" [label="{esc(r.name)}\\n{esc(r.left)} → {esc(r.right)}"];')
    lines.append("}")
    return "\n".join(lines)


st.set_page_config(page_title="Signature viewer", layout="wide")
st.title("Signature viewer")


@st.cache_data
def get_models():
    return load_models()


models = get_models()
by_path = {m.path.replace("\\", "/"): m for m in models}

with st.sidebar:
    kind = st.radio("Filter", ["All", "Splits only", "Merges only", "Splits + merges"], index=0)
    pool = [
        m for m in models
        if kind == "All"
        or (kind == "Splits only" and m.n_splits and not m.n_merges)
        or (kind == "Merges only" and m.n_merges and not m.n_splits)
        or (kind == "Splits + merges" and m.n_splits and m.n_merges)
    ]
    st.caption(f"{len(pool)}/{len(models)}")
    idx = st.selectbox(
        "Model",
        [m.index for m in pool],
        format_func=lambda i: f"M{i}  ({models[i].n_splits} split, {models[i].n_merges} merge)",
    )
    compare = st.checkbox("Compare with parent")

model = models[idx]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", f"M{model.index}")
c2.metric("Collections", len(model.collections))
c3.metric("References", len(model.references))
c4.metric("Depth", len(model.steps))

st.markdown("`" + " / ".join(["M0"] + model.steps) + "`")

parent = by_path.get(model.parent_path) if model.parent_path else None
if compare and parent:
    left, right = st.columns(2)
    with left:
        st.subheader(f"Parent: M{parent.index}")
        st.graphviz_chart(model_to_dot(parent), use_container_width=True)
    with right:
        st.subheader(f"This model: M{model.index}")
        st.graphviz_chart(model_to_dot(model), use_container_width=True)
else:
    if compare and not parent:
        st.info("M0 has no parent.")
    st.graphviz_chart(model_to_dot(model), use_container_width=True)

with st.expander("Signature"):
    st.code(describe(model), language="text")