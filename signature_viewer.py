"""Signature viewer for the DaMoOp generated models. Run: streamlit run signature_viewer.py"""
import html

import streamlit as st

from lib.costs import costs_at, load_costs, model_key
from lib.signatures import Nested, describe, load_models
from lib.stats import counts_at, load_stats, model_stats

ENTITY_COLORS = {"Customer": "#CDEFE3", "Order": "#FFE3B8", "Warehouse": "#D6E4FF"}
HEADER = "#1F2A44"


def esc(s: str) -> str:
    return html.escape(s, quote=True)


class Drawing:
    """Builds the Graphviz source and remembers where each concept instance is drawn."""

    def __init__(self, model):
        self.model = model
        self.instances = []   # (node, port, entity, attributes, path of enclosing ports)
        self.n = 0

    def port(self):
        self.n += 1
        return f"p{self.n}"

    def fields_table(self, fields, entity, node, port, ancestors):
        color = ENTITY_COLORS.get(entity, "#EEEEEE")
        attrs = [f for f in fields if isinstance(f, str)]
        self.instances.append((node, port, entity, set(attrs), ancestors))
        rows = []
        for f in fields:
            if not isinstance(f, Nested):
                rows.append(f'<TR><TD ALIGN="LEFT" BGCOLOR="{color}">{esc(f)}</TD></TR>')
                continue
            inner = ENTITY_COLORS.get(f.entity, "#EEEEEE")
            p = self.port()
            if f.multiplicity == "list":
                title = f"[ {esc(f.entity)} ]"
                frame = f'<TABLE BORDER="3" CELLBORDER="0" CELLSPACING="0" COLOR="#444444" STYLE="rounded">'
            else:
                title = f"{{ {esc(f.entity)} }}"
                frame = f'<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" COLOR="#444444">'
            rows.append(
                f'<TR><TD PORT="{p}" BGCOLOR="{inner}" CELLPADDING="4">{frame}'
                f'<TR><TD BGCOLOR="{inner}"><B>{title}</B></TD></TR>'
                f'{self.fields_table(f.fields, f.entity, node, p, ancestors + [port])}</TABLE></TD></TR>'
            )
        return "".join(rows)

    def dot(self):
        out = [
            "digraph G {",
            'graph [rankdir=LR, bgcolor="white", pad=0.3, nodesep=0.6, ranksep=1.0];',
            'node [shape=plaintext, fontname="Helvetica", fontsize=11];',
            'edge [fontname="Helvetica", fontsize=10, color="#555555", fontcolor="#333333"];',
        ]
        for c in self.model.collections:
            cols = []
            for i, part in enumerate(c.parts):
                p = self.port()
                head = f'<TR><TD BGCOLOR="#E9ECF2"><FONT POINT-SIZE="9">part {i + 1}</FONT></TD></TR>' if len(c.parts) > 1 else ""
                cols.append(
                    f'<TD PORT="{p}" VALIGN="TOP"><TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">'
                    f"{head}{self.fields_table(part, c.name, c.name, p, [])}</TABLE></TD>"
                )
            note = f' <FONT POINT-SIZE="9">split in {len(c.parts)}</FONT>' if len(c.parts) > 1 else ""
            out.append(
                f'"{c.name}" [label=<<TABLE BORDER="2" CELLBORDER="0" CELLSPACING="6" COLOR="{HEADER}">'
                f'<TR><TD COLSPAN="{len(c.parts)}" BGCOLOR="{HEADER}"><FONT COLOR="white"><B>{esc(c.name)}</B>{note}</FONT></TD></TR>'
                f'<TR>{"".join(cols)}</TR></TABLE>>];'
            )
        out += self.links()
        out.append("}")
        return "\n".join(out)

    def links(self):
        edges, seen = [], set()
        for r in self.model.references:
            left = [i for i in self.instances if r.left in i[3]]
            right = [i for i in self.instances if r.right in i[3]]
            for a in left:
                for b in right:
                    if a[0] == b[0] and (a[1] in b[4] or b[1] in a[4] or a[1] == b[1]):
                        continue  # same instance, or one nested in the other
                    # one edge per pair of concept instances, even if the concept is split
                    pair = (a[0], a[2], tuple(a[4]), b[0], b[2], tuple(b[4]))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    edges.append(
                        f'"{a[0]}":{a[1]} -> "{b[0]}":{b[1]} '
                        f'[label="{esc(r.name)}\\n{esc(r.left)} → {esc(r.right)}"];'
                    )
        return edges


def model_to_dot(model) -> str:
    return Drawing(model).dot()


st.set_page_config(page_title="Signature viewer", layout="wide")
st.title("Signature viewer")


@st.cache_data
def get_models():
    return load_models()


models = get_models()
stats = st.cache_data(load_stats)()
costs = st.cache_data(load_costs)()
by_path = {m.path: m for m in models}
children = {}
for m in models:
    if m.parent_path:
        children.setdefault(m.parent_path, []).append(m)

if "sel" not in st.session_state:
    st.session_state.sel = 0


def goto(i):
    st.session_state.sel = i
    st.session_state.query = ""
    st.session_state.kind = "All"


with st.sidebar:
    query = st.text_input("Search", key="query", placeholder="M42, Customer, Merge_c_o_ID…")
    kind = st.radio("Filter", ["All", "Splits only", "Merges only", "Splits + merges"], key="kind")
    q = query.strip().lower().lstrip("m")
    pool = [
        m for m in models
        if (kind == "All"
            or (kind == "Splits only" and m.n_splits and not m.n_merges)
            or (kind == "Merges only" and m.n_merges and not m.n_splits)
            or (kind == "Splits + merges" and m.n_splits and m.n_merges))
        and (not q
             or (q.isdigit() and m.index == int(q))
             or (not q.isdigit() and (query.lower() in m.signature.lower() or query.lower() in m.path.lower())))
    ]
    st.caption(f"{len(pool)}/{len(models)}")
    if not pool:
        st.stop()
    ids = [m.index for m in pool]
    if st.session_state.sel not in ids:
        st.session_state.sel = ids[0]
    st.selectbox(
        "Model", ids, key="sel",
        format_func=lambda i: f"M{i}  ({models[i].n_splits} split, {models[i].n_merges} merge)",
    )
    compare = st.checkbox("Compare with parent")
    varying = [c for c, v in stats.volumes.items() if len(v) > 1]
    n_steps = max(len(v) for v in stats.volumes.values())
    step = st.select_slider(
        "Volume", range(n_steps),
        format_func=lambda k: ", ".join(f"{counts_at(stats, k)[c]:,.0f} {c}" for c in varying) or str(k),
    )
    servers = st.select_slider("Servers", stats.servers)

volume_key = counts_at(stats, step)[varying[0]] if varying else None
cost_table = costs_at(costs, volume_key, servers) if costs is not None else None

model = models[st.session_state.sel]
parent = by_path.get(model.parent_path) if model.parent_path else None
kids = children.get(model.path, [])
siblings = [m for m in children.get(model.parent_path, []) if m.index != model.index] if parent else []

rows = model_stats(model, stats, step)
volume = sum(r["Volume (GB)"] for r in rows)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Model", f"M{model.index}")
c2.metric("Collections", len(model.collections))
c3.metric("References", len(model.references))
c4.metric("Depth", len(model.steps))
if parent:
    pv = sum(r["Volume (GB)"] for r in model_stats(parent, stats, step))
    c5.metric("Volume", f"{volume:,.2f} GB", f"{volume - pv:+,.2f} GB vs parent", delta_color="inverse")
else:
    c5.metric("Volume", f"{volume:,.2f} GB")

if cost_table is None:
    st.caption("No costs yet: run the pipeline, then python -m scripts.export_costs")
else:
    here = cost_table.loc[model_key(model)]
    before = cost_table.loc[model_key(parent)] if parent else None
    ranks = cost_table.rank()
    for col, (name, unit) in zip(st.columns(3), [("Time", "s"), ("Carbon", "kg CO2e"), ("Financial", "€")]):
        delta = f"{(here[name] / before[name] - 1) * 100:+.1f}% vs parent" if before is not None else None
        col.metric(f"{name} · rank {ranks.loc[model_key(model), name]:.0f}/{len(cost_table)}",
                   f"{here[name]:,.4g} {unit}", delta, delta_color="inverse")

# breadcrumb: every ancestor is clickable
crumbs, p = [], model
while p is not None:
    crumbs.insert(0, p)
    p = by_path.get(p.parent_path) if p.parent_path else None
bc = st.columns(len(crumbs) + 1)
for col, m in zip(bc, crumbs):
    col.button(f"M{m.index}", key=f"bc{m.index}", on_click=goto, args=(m.index,),
               disabled=m.index == model.index, help=m.steps[-1] if m.steps else "root")
    

def nav_row(title, items):
    if not items:
        return
    st.caption(f"{title} ({len(items)})")
    cols = st.columns(min(len(items), 6))
    for k, m in enumerate(items):
        cols[k % len(cols)].button(f"M{m.index} · {m.steps[-1]}", key=f"{title}{m.index}", on_click=goto, args=(m.index,))


with st.expander("Children and siblings", expanded=bool(kids)):
    nav_row("Children", kids)
    nav_row("Siblings", siblings)
    if cost_table is not None and (kids or siblings or parent):
        here = cost_table.loc[model_key(model)]
        near = [("parent", parent)] * bool(parent) + [("child", m) for m in kids] + [("sibling", m) for m in siblings]
        st.dataframe(
            [{"Model": f"M{m.index}", "Relation": rel, "Step": m.steps[-1] if m.steps else "root",
              **{f"{k} vs M{model.index}": f"{(cost_table.loc[model_key(m), k] / here[k] - 1) * 100:+.1f}%"
                 for k in ("Time", "Carbon", "Financial")}}
             for rel, m in near],
            hide_index=True, use_container_width=True,
        )

if compare and parent:
    left, right = st.columns(2)
    with left:
        st.subheader(f"M{parent.index}")
        st.graphviz_chart(model_to_dot(parent), use_container_width=True)
    with right:
        st.subheader(f"M{model.index}")
        st.graphviz_chart(model_to_dot(model), use_container_width=True)
else:
    if compare:
        st.info("M0 has no parent.")
    st.graphviz_chart(model_to_dot(model), use_container_width=True)

with st.expander("Statistics", expanded=True):
    st.dataframe(rows, hide_index=True, use_container_width=True,
                 column_config={"Volume (GB)": st.column_config.NumberColumn(format="%.4f"),
                                "Documents": st.column_config.NumberColumn(format="%d")})

with st.expander("Signature"):
    st.code(model.signature, language="text")
    st.code(describe(model), language="text")