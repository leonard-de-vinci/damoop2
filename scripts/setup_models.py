"""Materialize the shipped model bundles from their committed `.xz` archives.

A fresh clone ships the trained models LZMA-compressed (so they fit under GitHub's
100 MB limit). The Streamlit Predict page decompresses them on demand automatically;
this script does the same up-front for CLI users and the paper-figure dev scripts.

    python -m scripts.setup_models              # decompress every Output/models/*.pkl.xz
    python -m scripts.setup_models --compress   # (dev) re-create the .xz after training

Idempotent: skips a plain `.pkl` that already exists / is newer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lib.model_io import compress, ensure_pkl

MODELS_DIR = Path(__file__).resolve().parent.parent / "Output" / "models"

# Bundles that are meant to travel WITH the repo (Predict page + legacy predictor).
SHIPPED = ["model_ranker_log.pkl", "cost_evolution.pkl"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compress/decompress shipped model bundles.")
    ap.add_argument("--compress", action="store_true",
                    help="(dev) write .pkl.xz from the plain .pkl instead of decompressing.")
    ap.add_argument("--preset", type=int, default=6, help="LZMA preset for --compress.")
    args = ap.parse_args()

    for name in SHIPPED:
        pkl = MODELS_DIR / name
        if args.compress:
            if not pkl.exists():
                print(f"  skip {name}: no plain .pkl to compress")
                continue
            xz = compress(pkl, preset=args.preset)
            print(f"  compressed {name} -> {xz.name} ({xz.stat().st_size/1e6:.1f} MB)")
        else:
            xz = pkl.with_name(pkl.name + ".xz")
            if pkl.exists() and (not xz.exists() or pkl.stat().st_mtime >= xz.stat().st_mtime):
                print(f"  ok {name} already present")
                continue
            ensure_pkl(pkl)
            print(f"  decompressed {name} ({pkl.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
