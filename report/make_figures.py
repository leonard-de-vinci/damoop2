"""Render IEEE-column publication figures (PDF) from the exported CSVs.

Consumes files written by report/export_figure_data.py + report/regret_cases.csv:
  fig_regret_cdf.pdf        single col   — CDF of the 13,440 held-out regrets
  fig_feature_importance.pdf single col  — top-12 mean gain importance
  fig_pred_vs_actual.pdf    full width    — predicted vs true unit cost, 3 dims, log-log
  fig_winner_grid.pdf       full width    — composite cost of 4 models across the 6x7 grid
  fig_transformation_tree.pdf single col  — icicle of the 135-model tree at one op point

Spec: matplotlib, 8 pt fonts, no titles (captions live in LaTeX), PDF vector output,
figsize (3.46, 2.4) single column / (7.16, 2.6) full width.

Run: .venv/Scripts/python -m report.make_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from lib.pareto import build_tree_dataframe

REPORT = Path(__file__).resolve().parent
FIGS = REPORT / "figures"
FIGS.mkdir(exist_ok=True)
PARQUET = REPORT.parent / "Output" / "sweep_ml.parquet"

SINGLE = (3.46, 2.4)
FULL = (7.16, 2.6)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(FIGS / name, format="pdf")
    plt.close(fig)
    print(f"  -> figures/{name}")


def fig_regret_cdf():
    r = pd.read_csv(REPORT / "regret_cases.csv")["regret"].to_numpy(float)
    # Zero-regret share = fraction paying no extra raw cost (regret <= 0), identical
    # to Table IV (62.8%). Exact ==0 gives 61.1% and coincidentally equals top-1
    # accuracy (a different quantity); the gap is ~222 cases where the pick ties or
    # slightly beats the composite-best in raw mean cost (regret <= 0 but != exact 0).
    zshare = float((r <= 1e-9).mean())
    x = np.sort(r) * 100.0
    y = np.arange(1, len(x) + 1) / len(x)
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot(x, y, lw=1.4)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlim(0, x.max())
    ax.set_ylim(0, 1)
    ax.axhline(zshare, ls=":", lw=0.8, color="gray")
    ax.set_xlabel("Regret (%)")
    ax.set_ylabel("Cumulative fraction of cases")
    # Park the label in the upper-left, clear above the flat segment (curve sits
    # at ~0.63 there and does not climb into this corner until well to the right).
    ax.text(0.04, 0.90, f"{zshare:.1%} zero regret", transform=ax.transAxes,
            fontsize=6, va="center", ha="left")
    save(fig, "fig_regret_cdf.pdf")
    print(f"     regret CDF: zero-regret share {zshare:.4f} (tie-tolerant; matches Table IV)")


def fig_feature_importance():
    d = pd.read_csv(REPORT / "feature_importance.csv").head(12)[::-1]
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.barh(d["feature"], d["mean_gain_norm"], color="#3b6ea5")
    ax.set_xlabel("Mean gain importance")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_feature_importance.pdf")


def fig_pred_vs_actual():
    d = pd.read_csv(REPORT / "pred_vs_actual.csv")
    fig, axs = plt.subplots(1, 3, figsize=FULL)
    for ax, dim, title in zip(axs, ["time", "env", "fin"],
                              ["Time", "Environmental", "Financial"]):
        s = d[d["dim"] == dim]
        ax.scatter(s["true"], s["pred"], s=2, alpha=0.15, edgecolors="none",
                   color="#3b6ea5", rasterized=True)
        lo = min(s["true"].min(), s["pred"].min())
        hi = max(s["true"].max(), s["pred"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"True unit cost\n({title})")
        if ax is axs[0]:
            ax.set_ylabel("Predicted unit cost")
    save(fig, "fig_pred_vs_actual.pdf")


def _panel_labels(paths):
    """Disambiguate panel titles: many models share a leaf name (e.g. 12 distinct
    models end in Merge_w_c_ID_w_ID). When the leaf collides among the shown set,
    tag it with its originating branch + depth so no two panels look identical."""
    leaves = [p.split("/")[-1] for p in paths]
    labels = []
    for p, leaf in zip(paths, leaves):
        if leaves.count(leaf) > 1:
            segs = p.split("/")
            labels.append(f"{leaf}\n(via {segs[0]}, d{len(segs)})")
        else:
            labels.append(leaf)
    return labels


def fig_winner_grid():
    d = pd.read_csv(REPORT / "winner_grid.csv")
    # pick the 4 models that win the most cells at default workload
    win = d.loc[d.groupby(["data_volume", "servers"])["composite"].idxmin()]
    top = win["xml_file"].value_counts().head(4).index.tolist()
    labels = _panel_labels(top)
    vols = sorted(d["data_volume"].unique())
    srvs = sorted(d["servers"].unique())
    fig, axs = plt.subplots(1, len(top), figsize=FULL, sharey=True)
    vmin, vmax = d["composite"].min(), d["composite"].max()
    for ax, model, lab in zip(axs, top, labels):
        m = d[d["xml_file"] == model].pivot_table(
            index="servers", columns="data_volume", values="composite")
        m = m.reindex(index=srvs, columns=vols)
        im = ax.imshow(m.to_numpy(), aspect="auto", origin="lower",
                       cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(vols)))
        ax.set_xticklabels([f"{int(v):.0e}".replace("e+0", "e") for v in vols],
                           rotation=45, ha="right", fontsize=5)
        ax.set_yticks(range(len(srvs)))
        ax.set_yticklabels([int(s) for s in srvs], fontsize=5)
        ax.set_title(lab, fontsize=4.5)
        ax.set_xlabel("Volume")
        ax.grid(False)
    axs[0].set_ylabel("Servers")
    fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02, label="Composite")
    save(fig, "fig_winner_grid.pdf")


def fig_transformation_tree(profile="S000", volume=1_000_000, servers=100,
                            metric="total_time_cost", cutoff=100.0):
    """Icicle of the 135-model transformation tree at one operating point.

    Colour = log10(cost / root cost), diverging RdYlGn_r (green cheaper than the
    normalized root, red more expensive) — the dashboard's 'log(cost/root)' mode.
    Subtrees whose cost exceeds `cutoff`x the root (and all their descendants) are
    hidden for legibility, matching the dashboard's 'hide broken subtrees'.
    """
    df = pd.read_parquet(PARQUET, columns=["profile_id", "xml_file",
                                           "data_volume", "servers", metric])
    df = df[(df.profile_id == profile) & (df.data_volume == volume)
            & (df.servers == servers)]
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    agg = df.groupby("xml_file", as_index=False)[metric].mean()
    agg["model_full_path"] = agg["xml_file"]
    tree = build_tree_dataframe(agg, metric=metric, label_col="xml_file",
                                path_col="model_full_path")
    n_total = len(tree)

    root = tree[tree.depth == 0]
    root_cost = float(root["value"].iloc[0])
    root_id = root["id"].iloc[0]
    tree["cost_vs_root"] = tree["value"] / max(root_cost, 1e-12)

    # Hide broken subtrees: nodes > cutoff x root, plus every descendant.
    parent_map = dict(zip(tree["id"], tree["parent"]))
    hidden = set(tree.loc[tree.cost_vs_root > cutoff, "id"])
    changed = True
    while changed:
        changed = False
        for nid, par in parent_map.items():
            if par in hidden and nid not in hidden:
                hidden.add(nid); changed = True
    vis = tree[~tree["id"].isin(hidden)].copy()
    n_shown = len(vis)

    # Icicle layout: leaves share equal width; parents span their descendants.
    children: dict[str, list[str]] = {i: [] for i in vis["id"]}
    for _, r in vis.iterrows():
        if r["parent"] in children and r["parent"] != r["id"]:
            children[r["parent"]].append(r["id"])
    depth = dict(zip(vis["id"], vis["depth"]))
    cvr = dict(zip(vis["id"], vis["cost_vs_root"]))

    def n_leaves(nid):
        ch = children.get(nid, [])
        return 1 if not ch else sum(n_leaves(c) for c in ch)

    boxes = []  # (x0, x1, depth, color_val)

    def layout(nid, x0, x1):
        boxes.append((x0, x1, depth[nid], np.log10(max(cvr[nid], 1e-12))))
        ch = children.get(nid, [])
        if not ch:
            return
        total = sum(n_leaves(c) for c in ch)
        x = x0
        for c in sorted(ch, key=lambda k: k):  # stable order
            w = (x1 - x0) * n_leaves(c) / total
            layout(c, x, x + w)
            x += w

    layout(root_id, 0.0, 1.0)

    lim = max(abs(b[3]) for b in boxes) or 1.0
    norm = Normalize(-lim, lim)
    cmap = plt.cm.RdYlGn_r
    maxd = max(b[2] for b in boxes)

    fig, ax = plt.subplots(figsize=SINGLE)
    for x0, x1, d, cval in boxes:
        ax.add_patch(Rectangle((x0, d), x1 - x0, 1.0,
                               facecolor=cmap(norm(cval)),
                               edgecolor="white", linewidth=0.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(maxd + 1, 0)               # root on top
    ax.set_xticks([])
    ax.set_yticks(np.arange(maxd + 1) + 0.5)
    ax.set_yticklabels(range(maxd + 1))
    ax.set_ylabel("Transformation depth")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"$\log_{10}(\mathrm{cost}/\mathrm{root})$")
    save(fig, "fig_transformation_tree.pdf")
    shown_max_depth = int(vis["depth"].max())
    hidden_deep = int((tree.loc[tree["id"].isin(hidden), "depth"] >= 7).sum())
    print(f"     tree: {n_shown}/{n_total} candidates shown, depths 0-{shown_max_depth} "
          f"({n_total - n_shown} hidden beyond {cutoff:g}x root, incl. all {hidden_deep} "
          f"models at depths 7-9; profile {profile}, vol {volume:g}, srv {servers})")


def main():
    print("rendering figures ...")
    fig_regret_cdf()
    fig_feature_importance()
    fig_pred_vs_actual()
    fig_winner_grid()
    fig_transformation_tree()
    print("done.")


if __name__ == "__main__":
    main()
