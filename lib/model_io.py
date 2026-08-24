"""Transparent load/compress for the shipped model bundles.

The winner ranker (`model_ranker_log.pkl`) is ~366 MB uncompressed — too big for
GitHub's 100 MB/file limit. We therefore commit it LZMA-compressed as
`model_ranker_log.pkl.xz` (~94 MB) so a fresh clone has a working Predict page with
no dataset regeneration or retraining. This module hides the compression:

  * `load_bundle(path)` — accepts a `.pkl` or `.pkl.xz` path and returns the unpickled
    bundle, decompressing (and caching the plain `.pkl` next to it) when only the
    `.xz` is present.
  * `ensure_pkl(path)` — materialize the plain `.pkl` from its `.xz` if needed.
  * `compress(path)` — write `<path>.xz` from a plain `.pkl` (dev-side, after training).

LZMA is lossless, so a round-tripped bundle is byte-identical to the original.
"""
from __future__ import annotations

import lzma
import pickle
import shutil
from pathlib import Path

XZ_SUFFIX = ".xz"


def _xz_for(pkl_path: Path) -> Path:
    return pkl_path.with_name(pkl_path.name + XZ_SUFFIX)


def ensure_pkl(pkl_path: Path) -> Path:
    """Return a usable plain-`.pkl` path, decompressing the committed `.xz` if the
    plain file is absent. The decompressed `.pkl` is cached on disk (git-ignored) so
    subsequent loads and the CLI dev scripts find it without re-decompressing."""
    pkl_path = Path(pkl_path)
    if pkl_path.exists():
        return pkl_path
    xz = _xz_for(pkl_path)
    if xz.exists():
        tmp = pkl_path.with_name(pkl_path.name + ".tmp")
        with lzma.open(xz, "rb") as fin, open(tmp, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        tmp.replace(pkl_path)          # atomic; avoids a half-written cache
        return pkl_path
    raise FileNotFoundError(f"Neither {pkl_path.name} nor {xz.name} found in {pkl_path.parent}")


def load_bundle(path: str | Path) -> dict:
    """Load a model bundle from a `.pkl` or `.pkl.xz` path."""
    path = Path(path)
    if path.suffix == XZ_SUFFIX:
        # Decompress-and-cache the sibling .pkl, then load it.
        path = ensure_pkl(path.with_name(path.name[: -len(XZ_SUFFIX)]))
    else:
        path = ensure_pkl(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def compress(pkl_path: Path, preset: int = 6) -> Path:
    """Write `<pkl_path>.xz` (LZMA) next to a plain `.pkl`. Returns the .xz path."""
    pkl_path = Path(pkl_path)
    xz = _xz_for(pkl_path)
    with open(pkl_path, "rb") as fin, lzma.open(xz, "wb", preset=preset) as fout:
        shutil.copyfileobj(fin, fout)
    return xz
