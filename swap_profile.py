"""Profile swap helper — copies all input files from a profile directory into Input/.

Usage:
    python swap_profile.py TPC_C_ORIGINAL  # Activate TPC-C Original (paper's exact 5-query, 10-key setup)
    python swap_profile.py A               # Activate Profile A (extended baseline: paper infrastructure + 3 added attributes)
    python swap_profile.py B               # Activate Profile B (degraded selectivities)
    python swap_profile.py C               # Activate Profile C (sparse indexes)

After swapping, run the pipeline (Phase 1 + Phase 2) to generate costs for this
profile, then archive the Output/ folder before switching to the next profile.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "Input"
PROFILES_DIR = INPUT_DIR / "profiles"
OUTPUT_DIR = ROOT / "Output"

SWAPPED_FILES = ["config.txt", "keys.txt", "index.txt", "queries.txt", "queries_dmg.txt"]


def swap(profile: str) -> None:
    profile_dir = PROFILES_DIR / f"profile_{profile.upper()}"
    if not profile_dir.is_dir():
        print(f"ERROR: Profile directory not found: {profile_dir}")
        sys.exit(1)

    for fname in SWAPPED_FILES:
        src = profile_dir / fname
        dst = INPUT_DIR / fname
        if not src.exists():
            print(f"  WARNING: {src} does not exist, skipping.")
            continue
        shutil.copy2(src, dst)
        print(f"  Copied {src.name} -> Input/{fname}")

    print(f"\nProfile {profile.upper()} activated.")
    print("Next steps:")
    print("  1. Run pipeline (DMgeneration + CostCalculation)")
    print(f"  2. Archive output:  python swap_profile.py --archive {profile.upper()}")


def archive(profile: str) -> None:
    dest = OUTPUT_DIR / f"profile_{profile.upper()}"
    generated = OUTPUT_DIR / "GeneratedDMs"
    if not generated.exists():
        print(f"ERROR: No GeneratedDMs/ to archive.")
        sys.exit(1)
    if dest.exists():
        print(f"WARNING: {dest} already exists. Overwriting.")
        shutil.rmtree(dest)
    shutil.copytree(generated, dest)
    print(f"Archived Output/GeneratedDMs/ -> Output/profile_{profile.upper()}/")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] == "--archive" and len(sys.argv) >= 3:
        archive(sys.argv[2])
    else:
        swap(sys.argv[1])


if __name__ == "__main__":
    main()
