"""Pareto frontier and transformation tree helpers for the cost analysis page."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_pareto_mask(values_x: pd.Series, values_y: pd.Series) -> pd.Series:
    """Boolean mask: True for points on the Pareto frontier (minimize both axes).

    A point is Pareto-optimal if no other point dominates it (has lower or equal
    in both dimensions and strictly lower in at least one).
    """
    n = len(values_x)
    if n == 0:
        return pd.Series([], dtype=bool, index=values_x.index)
    arr = np.column_stack([values_x.to_numpy(dtype=float), values_y.to_numpy(dtype=float)])
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (arr[j, 0] <= arr[i, 0]
                    and arr[j, 1] <= arr[i, 1]
                    and (arr[j, 0] < arr[i, 0] or arr[j, 1] < arr[i, 1])):
                mask[i] = False
                break
    return pd.Series(mask, index=values_x.index)


def split_path_segments(full_path: str) -> list[str]:
    """Split a model path into directory segments.

    Paths in this project come in two flavours:
      - Root model is a single .xml file at top level (e.g. ``TPC_C.xml``)
      - Every other model is a directory path of operations (e.g.
        ``Split_r_Warehouse_1/Merge_c_ID_c_o_ID``); the costs.txt and
        ``<last_segment>.xml`` live inside that directory.

    Returns the list of directory segments (operations applied, in order).
    For the root .xml the list is empty.
    """
    if not full_path:
        return []
    p = str(full_path).replace("\\", "/").strip("/")
    if not p:
        return []
    # If the entire path is a single .xml file (the root), it has no segments
    if "/" not in p and p.lower().endswith(".xml"):
        return []
    # Otherwise the path is purely directory segments (no trailing .xml)
    return [s for s in p.split("/") if s]


def parent_path(full_path: str, root_label: str) -> str | None:
    """Return the parent model's full_path, or None if this is the root.

    Example:
        A/B/C  ->  A/B
        A      ->  <root_label> (e.g. ``TPC_C.xml``)
        TPC_C.xml or anything matching ``root_label`` -> None
    """
    if str(full_path) == str(root_label):
        return None
    segs = split_path_segments(full_path)
    if not segs:
        return None
    if len(segs) == 1:
        return root_label
    return "/".join(segs[:-1])


def find_root_label(all_paths: list[str]) -> str:
    """Identify the root model — the single top-level .xml file, e.g. TPC_C.xml."""
    candidates = [p for p in all_paths if "/" not in str(p).replace("\\", "/") and str(p).lower().endswith(".xml")]
    if candidates:
        return sorted(candidates)[0]
    # Fallback: shortest path overall
    return min(all_paths, key=lambda p: (len(split_path_segments(p)), str(p))) if all_paths else ""


def build_tree_dataframe(
    df_models: pd.DataFrame,
    metric: str,
    label_col: str = "xml_file",
    path_col: str = "model_full_path",
) -> pd.DataFrame:
    """Build a flat tree DataFrame (treemap-ready) with parent pointers and delta-vs-parent.

    `df_models` should have one row per model (already aggregated by mean/median).
    Returns columns: id, label, parent, value, depth, delta_vs_parent (relative %),
    delta_abs (absolute), op (the merge/split operation that produced this node).
    """
    if df_models.empty:
        return pd.DataFrame()

    paths = df_models[path_col].astype(str).tolist()
    root_label = find_root_label(paths)

    by_path: dict[str, dict] = {}
    for _, r in df_models.iterrows():
        full = str(r[path_col])
        segs = split_path_segments(full)
        if not segs:
            op = "(base)"
            depth = 0
        else:
            op = segs[-1]
            depth = len(segs)
        by_path[full] = {
            "id": full,
            "label": str(r[label_col]),
            "value": float(r[metric]),
            "depth": depth,
            "op": op,
        }

    rows = []
    for full, rec in by_path.items():
        parent_full = parent_path(full, root_label)
        if parent_full is None or parent_full == full:
            rec["parent"] = ""
            rec["delta_vs_parent"] = 0.0
            rec["delta_abs"] = 0.0
        else:
            parent_rec = by_path.get(parent_full)
            rec["parent"] = parent_rec["id"] if parent_rec else ""
            if parent_rec and parent_rec["value"] > 0:
                rec["delta_abs"] = rec["value"] - parent_rec["value"]
                rec["delta_vs_parent"] = rec["delta_abs"] / parent_rec["value"]
            else:
                rec["delta_abs"] = 0.0
                rec["delta_vs_parent"] = 0.0
        rows.append(rec)

    return pd.DataFrame(rows)


def chain_to_root(
    df_tree: pd.DataFrame,
    leaf_id: str,
) -> pd.DataFrame:
    """Walk from a leaf node to the root, returning the chain in root->leaf order."""
    if df_tree.empty:
        return df_tree
    by_id = {row["id"]: row for _, row in df_tree.iterrows()}
    chain = []
    cur = leaf_id
    seen = set()
    while cur and cur in by_id and cur not in seen:
        seen.add(cur)
        chain.append(by_id[cur])
        cur = by_id[cur].get("parent") or ""
    chain.reverse()
    return pd.DataFrame(chain)
