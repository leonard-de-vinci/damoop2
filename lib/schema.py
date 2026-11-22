"""Input schema (concepts, keys, references) read from Input/models/<base>.xml."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_FILE = Path("Input/models/TPC_C.xml")


@dataclass
class Schema:
    concepts: list = field(default_factory=list)      # concept names, in file order
    owner: dict = field(default_factory=dict)         # attribute -> concept
    primary_keys: dict = field(default_factory=dict)  # concept -> primary key attribute
    foreign_keys: set = field(default_factory=set)    # attributes marked fk
    references: list = field(default_factory=list)    # (name, source, target): source = pk, target = fk


def load_schema(path: Path | str = SCHEMA_FILE) -> Schema | None:
    path = Path(path)
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    s = Schema()
    for concept in root.iter("concepts"):
        name = concept.get("cName")
        s.concepts.append(name)
        for key in concept.iter("key"):
            attr, kind = key.get("kName"), key.get("kType")
            s.owner[attr] = name
            if kind == "Oid":
                s.primary_keys[name] = attr
            elif kind == "fk":
                s.foreign_keys.add(attr)
    for ref in root.iter("reference"):
        s.references.append((ref.get("rName"), ref.get("source"), ref.get("target")))
    return s
