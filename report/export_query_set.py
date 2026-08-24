"""Export the 20-query workload as a table for the paper.

Parses Input/queries.txt (workload) + Input/keys.txt (baseline selectivities) into
report/query_set.csv with, per query: id, tables touched, join arity (1/2/3),
where-column(s), baseline selectivity of the filter, selectivity class, result width
(# projected columns), row size (bytes), default frequency, SLA bound (s).

Run: .venv/Scripts/python -m report.export_query_set
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "Input" / "queries.txt"
KEYS = ROOT / "Input" / "keys.txt"


def load_selectivities() -> dict[str, tuple[int, str]]:
    """col -> (baseline_selectivity, role)  from keys.txt (`col,bytes,sel/Table,role`)."""
    sel: dict[str, tuple[int, str]] = {}
    for line in KEYS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        col, _bytes, sel_table, role = line.split(",")
        s = int(sel_table.split("/")[0])
        sel[col] = (s, role)
    return sel


def sel_class(value: int, roles: list[str]) -> str:
    if all(r in ("pk", "fk") for r in roles):
        return "point (key)"
    if value < 10:
        return "low"
    if value < 100:
        return "medium"
    return "high"


def main() -> None:
    sel_map = load_selectivities()
    rows = []
    for qi, raw in enumerate(QUERIES.read_text(encoding="utf-8").splitlines(), start=0):
        raw = raw.strip()
        if not raw or raw.startswith("query/"):
            continue
        spec, size, freq, max_time = raw.rsplit("/", 3)
        tok = spec.split(",")
        # projected columns: between 'select' and 'from'
        proj = tok[tok.index("select") + 1: tok.index("from")]
        # tables: every token after 'from'/'join' that is a known table name
        tables = [tok[i + 1] for i, t in enumerate(tok)
                  if t in ("from", "join") and i + 1 < len(tok)]
        tables = [t for t in tables if t in ("Warehouse", "Customer", "Order")]
        # where columns: everything after the last 'where'
        wi = len(tok) - 1 - tok[::-1].index("where")
        where_cols = [c for c in tok[wi + 1:] if c in sel_map]
        sels = [sel_map[c][0] for c in where_cols]
        roles = [sel_map[c][1] for c in where_cols]
        # combined baseline selectivity = product of distinct-value counts
        combined = 1
        for s in sels:
            combined *= s
        rows.append({
            "query_id": f"q{len([r for r in rows]) + 1}",
            "tables": "+".join(tables),
            "join_arity": len(tables),
            "where_cols": ",".join(where_cols),
            "where_selectivity_baseline": combined,
            "selectivity_class": sel_class(combined, roles),
            "result_width_cols": len(proj),
            "row_size_bytes": int(size),
            "default_frequency": int(freq),
            "sla_bound_s": float(max_time),
        })

    df = pd.DataFrame(rows)
    out = ROOT / "report" / "query_set.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n-> {out}")
    print(f"join-arity counts: {df['join_arity'].value_counts().sort_index().to_dict()}")
    print(f"sum default_frequency = {df['default_frequency'].sum()}")


if __name__ == "__main__":
    main()
