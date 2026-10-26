"""
Parse DaMoOp model signatures (Output/signatures_TPC_C.txt) into Python objects.

File format: lines come in pairs
    <signature>
    <folder path>   e.g. Output/GeneratedDMs/Split_r_Warehouse_1/

Signature format, collections separated by "|":
    Warehouse{(w_ID,w_city);(w_ID,w_name)}   -> a collection; ";" separates the parts of a SPLIT
    Order{(o_ID,...,c_ID_cv(c_ID,c_last))}  -> "xxx_cv(...)" is an entity NESTED by a MERGE
    orders(c_ID,c_o_ID)                      -> a REFERENCE (join link) between two collections

Run `python -m lib.signatures` from the repo root to print a summary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SIGNATURES_FILE = Path("Output/signatures_TPC_C.txt")


@dataclass
class Nested:
    """An entity nested inside another one by a merge (written xxx_cv(...))."""
    key: str                      # e.g. "c_ID"
    entity: str                   # e.g. "Customer" (resolved from the key)
    fields: list                  # plain field names (str) and Nested objects


@dataclass
class Collection:
    """One stored collection. Several parts = the entity was split."""
    name: str
    parts: list                   # list of parts; each part is a list of str / Nested


@dataclass
class Reference:
    """A join link between two collections, e.g. orders(c_ID, c_o_ID)."""
    name: str
    left: str
    right: str


@dataclass
class Model:
    index: int                    # 0 = the original normalized model
    path: str                     # folder path from the file
    steps: list                   # e.g. ["Split_r_Warehouse_1"]
    collections: list = field(default_factory=list)
    references: list = field(default_factory=list)

    @property
    def parent_path(self) -> str | None:
        """Folder path of the model this one was derived from (None for the original)."""
        if not self.steps:
            return None
        return "Output/GeneratedDMs/" + "".join(s + "/" for s in self.steps[:-1])

    @property
    def n_splits(self) -> int:
        return sum(s.startswith("Split") for s in self.steps)

    @property
    def n_merges(self) -> int:
        return sum(s.startswith("Merge") for s in self.steps)


# ---------------------------------------------------------------- helpers

def split_top(text: str, sep: str) -> list[str]:
    """Split on `sep`, ignoring separators inside parentheses or braces."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "({":
            depth += 1
        elif ch in ")}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [s.strip() for s in out if s.strip()]


def parse_fields(text: str, key_to_entity: dict) -> list:
    """'a,b,c_ID_cv(c_ID,x)' -> ['a', 'b', Nested(...)]."""
    fields = []
    for tok in split_top(text, ","):
        m = re.fullmatch(r"(\w+?)_cv\((.*)\)", tok)
        if m:
            key = m.group(1)
            fields.append(Nested(key, key_to_entity.get(key, key), parse_fields(m.group(2), key_to_entity)))
        else:
            fields.append(tok)
    return fields


def parse_signature(sig: str, key_to_entity: dict) -> tuple[list, list]:
    collections, references = [], []
    for tok in split_top(sig, "|"):
        m = re.fullmatch(r"(\w+)\{(.*)\}", tok)
        if m:
            parts = [parse_fields(p.strip()[1:-1], key_to_entity) for p in split_top(m.group(2), ";")]
            collections.append(Collection(m.group(1), parts))
            continue
        m = re.fullmatch(r"(\w+)\((\w+),(\w+)\)", tok)
        if m:
            references.append(Reference(*m.groups()))
            continue
        raise ValueError(f"Unrecognized token: {tok!r}")
    return collections, references


# ---------------------------------------------------------------- main API

def load_models(path: Path | str = SIGNATURES_FILE) -> list[Model]:
    lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) % 2:
        raise ValueError("Expected pairs of lines (signature, folder path)")
    pairs = list(zip(lines[0::2], lines[1::2]))

    # The first model is the normalized one: its first field in each collection is the key.
    first_cols, _ = parse_signature(pairs[0][0], {})
    key_to_entity = {c.parts[0][0]: c.name for c in first_cols}

    models = []
    for i, (sig, folder) in enumerate(pairs):
        rel = folder.replace("\\", "/").split("GeneratedDMs/", 1)[-1]
        steps = [s for s in rel.split("/") if s]
        cols, refs = parse_signature(sig, key_to_entity)
        models.append(Model(i, folder, steps, cols, refs))
    return models


def describe(model: Model) -> str:
    """Short human-readable description, used by the viewer and the CLI."""
    def fmt(fields):
        return ", ".join(f if isinstance(f, str) else f"[{f.entity}: {fmt(f.fields)}]" for f in fields)
    lines = [f"Model {model.index}  ({model.n_splits} split, {model.n_merges} merge)  {'/'.join(model.steps) or 'original'}"]
    for c in model.collections:
        lines.append(f"  {c.name}: " + " | ".join(f"({fmt(p)})" for p in c.parts))
    for r in model.references:
        lines.append(f"  ref {r.name}: {r.left} -> {r.right}")
    return "\n".join(lines)


if __name__ == "__main__":
    models = load_models()
    print(f"{len(models)} models loaded")
    print(f"with splits: {sum(m.n_splits > 0 for m in models)}, with merges: {sum(m.n_merges > 0 for m in models)}")
    for m in models[:3] + models[-1:]:
        print()
        print(describe(m))