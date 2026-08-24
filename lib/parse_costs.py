from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


STAR_LINE_RE = re.compile(r"^\*{5,}\s*$")


def normalize_sig_paths(sig_file: Path) -> None:
    """Strip 'Output/' prefix and trailing '/' from path lines so Cost.jar writes correctly.

    The signatures file alternates: signature line, path line, signature line, path line…
    Old DMgeneration.jar wrote paths like 'Output/GeneratedDMs/SubDir/'.
    New Cost.java prepends 'Output/' itself, so the paths must be 'GeneratedDMs/SubDir'.
    """
    if not sig_file.exists():
        return
    lines = sig_file.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = False
    for i in range(1, len(lines), 2):  # path lines sit at odd indices (0-based)
        raw = lines[i]
        cleaned = raw.strip()
        if cleaned.startswith("Output/"):
            cleaned = cleaned[len("Output/"):]
        cleaned = cleaned.rstrip("/")
        new_line = cleaned + "\n"
        if new_line != raw:
            lines[i] = new_line
            changed = True
    if changed:
        sig_file.write_text("".join(lines), encoding="utf-8")


def _parse_key_value_line(line: str) -> tuple[str, str] | None:
    if ":" not in line:
        return None
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def parse_single_costs_file(costs_file: Path, gdm_root: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    xml_candidates = sorted(costs_file.parent.glob("*.xml"))
    xml_name = xml_candidates[0].name if xml_candidates else None
    xml_path = str(xml_candidates[0]) if xml_candidates else None

    # Build a unique model identifier using the relative path from the GeneratedDMs
    # root so that models at different nesting levels with the same XML filename
    # are not collapsed together (e.g. there are 12 distinct models all named
    # Merge_w_c_ID_w_ID.xml at different depths).
    if gdm_root is not None:
        try:
            rel = costs_file.parent.relative_to(gdm_root)
            rel_str = rel.as_posix()          # forward-slash separated, OS-agnostic
            model_id = rel_str if rel_str != "." else (xml_name or "TPC_C")
        except ValueError:
            model_id = xml_name
    else:
        model_id = xml_name

    lines = costs_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not STAR_LINE_RE.match(line):
            i += 1
            continue

        block: dict[str, Any] = {
            "costs_file": str(costs_file),
            "xml_file": model_id,   # unique path-based ID, not just the filename
            "xml_name": xml_name,   # short filename for display
            "xml_path": xml_path,
        }

        i += 1
        while i < len(lines):
            current = lines[i].strip()
            if STAR_LINE_RE.match(current):
                i += 1
                break

            parsed = _parse_key_value_line(current)
            if parsed:
                key, value = parsed
                if key == "Data Volume":
                    block["data_volume"] = _safe_float(value)
                elif key == "Data Model Size":
                    block["data_model_size"] = _safe_float(value)
                elif key == "#servers":
                    block["servers"] = _safe_float(value)
                else:
                    normalized = key.lower().replace(" ", "_").replace("#", "num_")
                    block[normalized] = value
            elif current:
                block["mode"] = current

            i += 1

        query_idx = 0
        while i < len(lines):
            current = lines[i].strip()

            if STAR_LINE_RE.match(current):
                break

            # ── Per-query block header ──────────────────────────────────
            if current.startswith("Query:"):
                query_idx += 1
                label_raw = current[len("Query:"):].strip()
                block[f"q{query_idx}_label"] = label_raw
                i += 1
                continue

            # ── Per-query cost lines (tab-separated: Key:\tvalue\t...) ──
            if query_idx > 0 and current.startswith("Time Cost:"):
                parts = current.split("\t")
                block[f"q{query_idx}_time_cost"] = _safe_float(parts[1]) if len(parts) > 1 else None
                i += 1
                continue

            if query_idx > 0 and current.startswith("Environmental Cost:"):
                parts = current.split("\t")
                block[f"q{query_idx}_env_cost"] = _safe_float(parts[1]) if len(parts) > 1 else None
                i += 1
                continue

            if query_idx > 0 and current.startswith("Financial Cost:"):
                parts = current.split("\t")
                block[f"q{query_idx}_fin_cost"] = _safe_float(parts[1]) if len(parts) > 1 else None
                i += 1
                continue

            # ── Total cost summary block ────────────────────────────────
            if current.startswith("All queries (Total Cost)"):
                query_idx = 0  # stop per-query capture; total section handled below
                i += 1

                while i < len(lines):
                    summary_line = lines[i].strip()
                    if not summary_line:
                        i += 1
                        break
                    if summary_line.startswith("Query:") or STAR_LINE_RE.match(summary_line):
                        break
                    if set(summary_line) == {"-"}:
                        i += 1
                        continue

                    summary_parsed = _parse_key_value_line(summary_line)
                    if summary_parsed:
                        key, value = summary_parsed
                        normalized_key = f"total_{key.lower().replace(' ', '_')}"
                        numeric = _safe_float(value)
                        block[normalized_key] = numeric if numeric is not None else value

                    i += 1

                continue

            i += 1

        rows.append(block)

    return rows


def build_costs_dataframe(root_dir: Path) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []

    # The GeneratedDMs folder is used as the anchor for unique model IDs so that
    # models with the same XML filename at different nesting levels are kept distinct.
    gdm_root = root_dir / "GeneratedDMs"
    if not gdm_root.exists():
        gdm_root = root_dir   # fallback

    for costs_file in root_dir.rglob("costs.txt"):
        all_rows.extend(parse_single_costs_file(costs_file, gdm_root))

    df = pd.DataFrame(all_rows)
    if not df.empty:
        ordered_columns = [
            "xml_file",
            "xml_name",
            "xml_path",
            "costs_file",
            "mode",
            "data_volume",
            "data_model_size",
            "servers",
            "total_time_cost",
            "total_environmental_cost",
            "total_financial_cost",
        ]
        existing = [column for column in ordered_columns if column in df.columns]
        remaining = [column for column in df.columns if column not in existing]
        df = df[existing + remaining]

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse all costs.txt files and attach sibling XML filename into a pandas DataFrame."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Output"),
        help="Root directory to search for costs.txt files (default: Output)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aggregated_costs_from_txt.csv"),
        help="Output CSV file path (default: aggregated_costs_from_txt.csv)",
    )
    args = parser.parse_args()

    df = build_costs_dataframe(args.root)
    df.to_csv(args.output, index=False)

    print(f"Rows parsed: {len(df)}")
    print(f"CSV written to: {args.output}")
    if not df.empty:
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
