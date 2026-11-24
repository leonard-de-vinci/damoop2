"""Signature Viewer page: draw, navigate and compare the generated data models."""
import html

import streamlit as st

from lib.costs import costs_at, load_costs, model_key
from lib.signatures import Nested, build_context, describe, load_models
from lib.stats import counts_at, load_stats, model_stats

# one light colour per concept, given in schema order, so any input schema works
PALETTE = ["#D6E4FF", "#CDEFE3", "#FFE3B8", "#F6D5E5", "#E6DDF7", "#FFF3B0", "#D5F0F4", "#EADBC8"]
HEADER = "#1F2A44"
SAME_KEY = "#2F7D6D"
ENTITY_COLORS: dict = {}
PRIMARY_KEYS: dict = {}


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
        out += self.copy_links()
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


    def copy_links(self):
        """Dashed link between copies of the same concept, through its primary key.

        A merge can leave a concept in two places (e.g. Customer top-level and
        nested in Warehouse): the copies share the key, so the link stays visible.
        """
        edges = []
        for entity, pk in PRIMARY_KEYS.items():
            copies = {}
            for inst in self.instances:
                if inst[2] == entity and pk in inst[3]:
                    copies.setdefault((inst[0], tuple(inst[4])), inst)  # split parts of one copy count once
            groups = list(copies.values())
            if len(groups) < 2:
                continue
            main = next((g for g in groups if not g[4]), groups[0])  # prefer the top-level copy
            for g in groups:
                if g is main or (g[0] == main[0] and (main[1] in g[4] or g[1] in main[4])):
                    continue
                edges.append(
                    f'"{g[0]}":{g[1]} -> "{main[0]}":{main[1]} '
                    f'[style=dashed, color="{SAME_KEY}", fontcolor="{SAME_KEY}", arrowhead=none, '
                    f'label="same {esc(entity)}\n{esc(pk)}"];'
                )
        return edges


def model_to_dot(model) -> str:
    return Drawing(model).dot()


@st.cache_data
def get_models():
    return load_models()


@st.cache_data
def get_stats():
    return load_stats()


@st.cache_data
def get_costs():
    return load_costs()


def goto(i):
    st.session_state.sv_sel = i
    st.session_state.sv_query = ""
    st.session_state.sv_kind = "All"


def heading(text):
    st.markdown(f'<div class="section-heading">{esc(text)}</div>', unsafe_allow_html=True)


def short(x):
    """29321000 -> 29.32 M, keeps small values readable."""
    for div, unit in ((1e12, " T"), (1e9, " B"), (1e6, " M"), (1e4, " k")):
        if abs(x) >= div:
            return f"{x / div:,.2f}{unit}"
    return f"{x:,.4g}"


def size(b):
    for div, unit in ((1e12, "TB"), (1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if b >= div:
            return f"{b / div:,.2f} {unit}"
    return f"{b:,.0f} B"


def change(new, old):
    """Relative change; ×N when a model is ten times or more above its reference."""
    if not old:
        return "n/a"
    r = new / old
    return f"×{r:,.0f}" if r >= 10 else f"{(r - 1) * 100:+.1f}%"


def nav_row(title, items):
    if not items:
        return
    st.caption(f"{title} ({len(items)}), hover for the transformation step")
    cols = st.columns(min(len(items), 8))
    for k, m in enumerate(items):
        cols[k % len(cols)].button(f"M{m.index}", key=f"sv_{title}{m.index}", on_click=goto, args=(m.index,),
                                   help=m.steps[-1], use_container_width=True)


FILTERS = ["All", "Splits only", "Merges only", "Splits + merges"]


def render() -> None:
    models = get_models()
    ctx = build_context(models[0].signature)
    PRIMARY_KEYS.update(ctx.primary_keys)
    for i, c in enumerate(ctx.concepts):
        ENTITY_COLORS[c] = PALETTE[i % len(PALETTE)]
    stats = get_stats()
    costs = get_costs()
    by_path = {m.path: m for m in models}
    children = {}
    for m in models:
        if m.parent_path:
            children.setdefault(m.parent_path, []).append(m)
    if "sv_sel" not in st.session_state:
        st.session_state.sv_sel = 0

    # ---- controls -------------------------------------------------------
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 2, 3, 2], vertical_alignment="bottom")
        query = c1.text_input("Search", key="sv_query", placeholder="M42, w_c_ID_cv, Merge_c_o_ID…")
        kind = c2.selectbox("Filter", FILTERS, key="sv_kind")
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
        if not pool:
            st.warning("No model matches this search.")
            return
        ids = [m.index for m in pool]
        if st.session_state.sv_sel not in ids:
            st.session_state.sv_sel = ids[0]
        c3.selectbox(
            f"Model ({len(pool)}/{len(models)})", ids, key="sv_sel",
            format_func=lambda i: f"M{i}  ({models[i].n_splits} split, {models[i].n_merges} merge)",
        )
        compare = c4.checkbox("Compare with parent", key="sv_compare")

        varying = [c for c, v in stats.volumes.items() if len(v) > 1]
        n_steps = max(len(v) for v in stats.volumes.values())
        s1, s2 = st.columns(2)
        step = s1.select_slider(
            "Volume", range(n_steps), key="sv_step",
            format_func=lambda k: ", ".join(f"{counts_at(stats, k)[c]:,.0f} {c}" for c in varying) or str(k),
        )
        servers = s2.select_slider("Servers", stats.servers, key="sv_servers")

    volume_key = counts_at(stats, step)[varying[0]] if varying else None
    cost_table = costs_at(costs, volume_key, servers) if costs is not None else None

    model = models[st.session_state.sv_sel]
    parent = by_path.get(model.parent_path) if model.parent_path else None
    kids = children.get(model.path, [])
    siblings = [m for m in children.get(model.parent_path, []) if m.index != model.index] if parent else []

    # ---- path from M0 ---------------------------------------------------
    crumbs, p = [], model
    while p is not None:
        crumbs.insert(0, p)
        p = by_path.get(p.parent_path) if p.parent_path else None
    bc = st.columns(max(len(crumbs), 10))
    for col, m in zip(bc, crumbs):
        col.button(f"M{m.index}", key=f"sv_bc{m.index}", on_click=goto, args=(m.index,), use_container_width=True,
                   disabled=m.index == model.index, help=m.steps[-1] if m.steps else "base model")
    st.caption(" › ".join(["M0"] + model.steps) if model.steps else "M0, the base model")

    # ---- key figures ----------------------------------------------------
    rows = model_stats(model, stats, step)
    volume = sum(r["Volume (GB)"] for r in rows) * 1e9
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model", f"M{model.index}")
    c2.metric("Collections", len(model.collections))
    c3.metric("References", len(model.references))
    c4.metric("Depth", len(model.steps))
    if parent:
        pv = sum(r["Volume (GB)"] for r in model_stats(parent, stats, step)) * 1e9
        c5.metric("Volume", size(volume), f"{change(volume, pv)} vs parent", delta_color="inverse")
    else:
        c5.metric("Volume", size(volume))

    if cost_table is None:
        st.caption("No costs yet: run the pipeline, then python -m scripts.export_costs")
    else:
        here = cost_table.loc[model_key(model)]
        before = cost_table.loc[model_key(parent)] if parent else None
        ranks = cost_table.rank()
        for col, (name, unit) in zip(st.columns(3), [("Time", "s"), ("Carbon", "kg CO2e"), ("Financial", "€")]):
            delta = f"{change(here[name], before[name])} vs parent" if before is not None else None
            col.metric(f"{name} · rank {ranks.loc[model_key(model), name]:.0f}/{len(cost_table)}",
                       f"{short(here[name])} {unit}", delta, delta_color="inverse")

    # ---- drawing --------------------------------------------------------
    heading("Structure")
    if compare and parent:
        left, right = st.columns(2)
        with left:
            st.caption(f"Parent: M{parent.index}")
            st.graphviz_chart(model_to_dot(parent), use_container_width=True)
        with right:
            st.caption(f"M{model.index}")
            st.graphviz_chart(model_to_dot(model), use_container_width=True)
    else:
        if compare:
            st.info("M0 has no parent.")
        st.graphviz_chart(model_to_dot(model), use_container_width=True)
    st.caption("[ list ] thick rounded frame · { single } thin frame · dashed line: same concept, linked by its key")

    # ---- neighbours -----------------------------------------------------
    heading("Parent, children and siblings")
    nav_row("Children", kids)
    nav_row("Siblings", siblings)
    if cost_table is not None and (kids or siblings or parent):
        here = cost_table.loc[model_key(model)]
        near = [("parent", parent)] * bool(parent) + [("child", m) for m in kids] + [("sibling", m) for m in siblings]
        st.dataframe(
            [{"Model": f"M{m.index}", "Relation": rel, "Step": m.steps[-1] if m.steps else "base model",
              **{f"{k} vs M{model.index}": change(cost_table.loc[model_key(m), k], here[k])
                 for k in ("Time", "Carbon", "Financial")}}
             for rel, m in near],
            hide_index=True, use_container_width=True,
        )
    elif not (kids or siblings or parent):
        st.caption("No parent, children or siblings.")

    # ---- statistics -----------------------------------------------------
    heading("Statistics")
    st.dataframe(
        [{"Collection": r["Collection"], "Part": r["Part"], "Documents": f"{r['Documents']:,.0f}",
          "Doc size": size(r["Doc size (B)"]), "Volume": size(r["Volume (GB)"] * 1e9)} for r in rows],
        hide_index=True, use_container_width=True,
    )

    with st.expander("Signature"):
        st.code(model.signature, language="text")
        st.code(describe(model), language="text")
