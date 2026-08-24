"""Patch a single profile directory from a parameter spec.

Reads profile_A as the template, patches selectivity values into keys.txt,
index flags into index.txt, and Customer/Order row counts into config.txt.
queries.txt and queries_dmg.txt are copied verbatim.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "Input" / "profiles" / "profile_A"

NON_KEY_COLUMNS: tuple[str, ...] = (
    "w_name", "w_city", "w_tax",
    "c_last", "balance", "c_credit",
    "o_carrier_id", "o_ol_cnt",
)

# Per-column baseline (is_indexed, fanout, is_secondary, depth) from profile_A.
# Restored verbatim when index_flag is True.
INDEX_BASELINE: dict[str, tuple[str, str, str, str]] = {
    "w_name":       ("yes", "8", "no",  "8"),
    "w_city":       ("yes", "6", "yes", "8"),
    "w_tax":        ("yes", "6", "no",  "6"),
    "c_last":       ("yes", "5", "yes", "8"),
    "balance":      ("yes", "8", "no",  "6"),
    "c_credit":     ("yes", "4", "no",  "4"),
    "o_carrier_id": ("yes", "6", "no",  "8"),
    "o_ol_cnt":     ("yes", "5", "no",  "5"),
}

UNINDEXED_TUPLE: tuple[str, str, str, str] = ("no", "0", "no", "0")

PROFILE_FILES: tuple[str, ...] = (
    "config.txt", "keys.txt", "index.txt", "queries.txt", "queries_dmg.txt",
)


def _patch_keys(text: str, selectivities: dict[str, int]) -> str:
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        eol = line[len(bare):]
        if not bare:
            out.append(line)
            continue
        parts = bare.split(",")
        if len(parts) != 4:
            out.append(line)
            continue
        name, bytes_, sel_table, role = parts
        if name in selectivities and role == "null" and "/" in sel_table:
            _, table = sel_table.split("/", 1)
            sel_table = f"{int(selectivities[name])}/{table}"
            out.append(",".join([name, bytes_, sel_table, role]) + eol)
        else:
            out.append(line)
    return "".join(out)


def _patch_index(text: str, index_flags: dict[str, bool]) -> str:
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        eol = line[len(bare):]
        if not bare:
            out.append(line)
            continue
        toks = bare.split(",")
        if len(toks) < 6 or (len(toks) - 1) % 5 != 0:
            out.append(line)
            continue
        table = toks[0]
        rest = toks[1:]
        patched: list[str] = [table]
        for i in range(0, len(rest), 5):
            name, is_idx, fan, is_sec, depth = rest[i:i + 5]
            if name in index_flags:
                if index_flags[name]:
                    base = INDEX_BASELINE.get(name)
                    if base is not None:
                        is_idx, fan, is_sec, depth = base
                else:
                    is_idx, fan, is_sec, depth = UNINDEXED_TUPLE
            patched.extend([name, is_idx, fan, is_sec, depth])
        out.append(",".join(patched) + eol)
    return "".join(out)


def _patch_config(text: str, customer_count: int, order_count: int) -> str:
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        eol = line[len(bare):]
        parts = bare.split("\t")
        if len(parts) >= 2 and parts[0] == "Customer":
            parts[1] = str(int(customer_count))
            out.append("\t".join(parts) + eol)
        elif len(parts) >= 2 and parts[0] == "Order":
            parts[1] = str(int(order_count))
            out.append("\t".join(parts) + eol)
        else:
            out.append(line)
    return "".join(out)


def generate_profile(spec: dict[str, Any], dest: Path) -> None:
    """Write a complete profile directory at `dest` based on `spec`.

    Required spec keys: selectivities, index_flags, customer_count, order_count.
    Any additional keys (profile_id, seed_idx, ...) are persisted to params.json.
    """
    dest.mkdir(parents=True, exist_ok=True)

    keys_src = (TEMPLATE_DIR / "keys.txt").read_text(encoding="utf-8", newline="")
    (dest / "keys.txt").write_bytes(
        _patch_keys(keys_src, spec["selectivities"]).encode("utf-8")
    )

    idx_src = (TEMPLATE_DIR / "index.txt").read_text(encoding="utf-8", newline="")
    (dest / "index.txt").write_bytes(
        _patch_index(idx_src, spec["index_flags"]).encode("utf-8")
    )

    cfg_src = (TEMPLATE_DIR / "config.txt").read_text(encoding="utf-8", newline="")
    (dest / "config.txt").write_bytes(
        _patch_config(cfg_src, spec["customer_count"], spec["order_count"]).encode("utf-8")
    )

    for fname in ("queries.txt", "queries_dmg.txt"):
        shutil.copy2(TEMPLATE_DIR / fname, dest / fname)

    (dest / "params.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
