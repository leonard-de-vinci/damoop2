"""Collection statistics from Input/config.txt and Input/keys.txt."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lib.signatures import Model

CONFIG_FILE = Path("Input/config.txt")
KEYS_FILE = Path("Input/keys.txt")


@dataclass
class Stats:
    servers: list            # server counts tested
    volumes: dict            # concept -> list of document counts (one per volume)
    sizes: dict              # attribute -> size in bytes


def load_stats(config: Path | str = CONFIG_FILE, keys: Path | str = KEYS_FILE) -> Stats:
    servers, volumes, section = [], {}, None
    for line in Path(config).read_text(encoding="utf-8").splitlines():
        cols = line.split()
        if not cols:
            continue
        if len(cols) == 1:
            section = cols[0]
        elif section == "Servers":
            servers = [int(x) for x in cols[1:]]
        elif section == "Data":
            nums = [int(x) for x in cols[1:]]
            if len(nums) == 2:  # a range: min to max, one volume per power of 10
                lo, hi = nums
                nums = []
                while lo <= hi:
                    nums.append(lo)
                    lo *= 10
            volumes[cols[0]] = nums

    sizes = {}
    for line in Path(keys).read_text(encoding="utf-8").splitlines():
        if line.strip():
            name, size, *_ = line.strip().split(",")
            sizes[name] = int(size)
    return Stats(servers, volumes, sizes)


def counts_at(stats: Stats, step: int) -> dict:
    """Document count per concept for a volume step.

    The concept given as a range sets the scale; the others grow in proportion,
    as CostCalculation.jar does.
    """
    ranged = [v for v in stats.volumes.values() if len(v) > 1]
    factor = ranged[0][min(step, len(ranged[0]) - 1)] / ranged[0][0] if ranged else 1
    return {c: v[min(step, len(v) - 1)] if len(v) > 1 else v[0] * factor
            for c, v in stats.volumes.items()}


def doc_size(fields: list, entity: str, counts: dict, sizes: dict) -> float:
    total = 0.0
    for f in fields:
        if isinstance(f, str):
            total += sizes.get(f, 0)
        else:
            # a list holds on average N(nested) / N(parent) documents, a single document holds one
            n = counts[f.entity] / counts[entity] if f.multiplicity == "list" else 1
            total += n * doc_size(f.fields, f.entity, counts, sizes)
    return total


def model_stats(model: Model, stats: Stats, step: int = 0) -> list[dict]:
    counts = counts_at(stats, step)
    rows = []
    for c in model.collections:
        for i, part in enumerate(c.parts):
            size = doc_size(part, c.name, counts, stats.sizes)
            rows.append({
                "Collection": c.name,
                "Part": i + 1,
                "Documents": counts[c.name],
                "Doc size (B)": round(size, 1),
                "Volume (GB)": counts[c.name] * size / 1e9,
            })
    return rows
