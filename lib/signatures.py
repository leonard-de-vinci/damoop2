"""Parser for DaMoOp model signatures."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from lib.schema import SCHEMA_FILE, load_schema

SIGNATURES_FILE = Path("Output/signatures_TPC_C.txt")


@dataclass
class Nested:
    """A concept nested in another one by a merge, written <key>_cv(...)."""
    key: str
    entity: str
    multiplicity: str             # "list" or "single"
    fields: list


@dataclass
class Collection:
    name: str
    parts: list                   # one part per row; several rows = split


@dataclass
class Reference:
    name: str
    left: str                     # key on the "one" side
    right: str                    # foreign key on the "many" side


@dataclass
class Model:
    index: int
    path: str
    signature: str
    steps: list
    collections: list = field(default_factory=list)
    references: list = field(default_factory=list)

    @property
    def parent_path(self) -> str | None:
        if not self.steps:
            return None
        return "Output/GeneratedDMs/" + "".join(s + "/" for s in self.steps[:-1])

    @property
    def n_splits(self) -> int:
        return sum(s.startswith("Split") for s in self.steps)

    @property
    def n_merges(self) -> int:
        return sum(s.startswith("Merge") for s in self.steps)


@dataclass
class Context:
    """What we know about the input schema, used to read every signature."""
    field_owner: dict             # attribute -> concept that owns it
    foreign_keys: set             # attributes on the "many" side of a reference
    primary_keys: dict = field(default_factory=dict)  # concept -> primary key
    concepts: list = field(default_factory=list)      # concept names, schema order


def split_top(text: str, sep: str) -> list[str]:
    """Split on sep, ignoring separators inside () or {}."""
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


def parse_fields(text: str, ctx: Context | None) -> list:
    fields = []
    for tok in split_top(text, ","):
        m = re.fullmatch(r"(\w+?)_cv\((.*)\)", tok)
        if not m:
            fields.append(tok)
            continue
        key = m.group(1)
        entity = ctx.field_owner.get(key, key) if ctx else key
        # The concept holding the foreign key is the "many" side:
        # nested in the other concept it becomes a list, otherwise a single document.
        multiplicity = "list" if ctx and key in ctx.foreign_keys else "single"
        fields.append(Nested(key, entity, multiplicity, parse_fields(m.group(2), ctx)))
    return fields


def parse_signature(sig: str, ctx: Context | None) -> tuple[list, list]:
    collections, references = [], []
    for tok in split_top(sig, "|"):
        m = re.fullmatch(r"(\w+)\{(.*)\}", tok)
        if m:
            parts = [parse_fields(p.strip()[1:-1], ctx) for p in split_top(m.group(2), ";")]
            collections.append(Collection(m.group(1), parts))
            continue
        m = re.fullmatch(r"(\w+)\((\w+),(\w+)\)", tok)
        if m:
            references.append(Reference(*m.groups()))
            continue
        raise ValueError(f"Unrecognised token: {tok!r}")
    return collections, references


def build_context(m0_signature: str, schema_path: Path | str = SCHEMA_FILE) -> Context:
    """Keys and links from the input schema; falls back to M0's signature line."""
    schema = load_schema(schema_path)
    if schema is not None:
        return Context(schema.owner, set(schema.foreign_keys) | {t for _, _, t in schema.references},
                       schema.primary_keys, schema.concepts)
    cols, refs = parse_signature(m0_signature, None)
    owner = {f: c.name for c in cols for p in c.parts for f in p if isinstance(f, str)}
    pks = {r.left: owner[r.left] for r in refs if r.left in owner}
    return Context(owner, {r.right for r in refs}, {c: k for k, c in pks.items()}, [c.name for c in cols])


def load_models(path: Path | str = SIGNATURES_FILE, schema_path: Path | str = SCHEMA_FILE) -> list[Model]:
    lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) % 2:
        raise ValueError("Expected pairs of lines (signature, folder path)")
    pairs = list(zip(lines[0::2], lines[1::2]))

    ctx = build_context(pairs[0][0], schema_path)

    models = []
    for i, (sig, folder) in enumerate(pairs):
        folder = folder.replace("\\", "/")
        rel = folder.split("GeneratedDMs/", 1)[-1]
        steps = [s for s in rel.split("/") if s]
        cols, refs = parse_signature(sig, ctx)
        models.append(Model(i, folder, sig, steps, cols, refs))
    return models


def describe(model: Model) -> str:
    def fmt(fields):
        out = []
        for f in fields:
            if isinstance(f, str):
                out.append(f)
            elif f.multiplicity == "list":
                out.append(f"[{f.entity}: {fmt(f.fields)}]")
            else:
                out.append(f"{{{f.entity}: {fmt(f.fields)}}}")
        return ", ".join(out)

    lines = [f"M{model.index}  ({model.n_splits} split, {model.n_merges} merge)  {'/'.join(model.steps) or 'M0'}"]
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