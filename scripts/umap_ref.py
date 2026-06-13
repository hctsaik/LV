"""Persistent UMAP reference frame（固定參考系）.

UMAP is non-parametric: refitting after ANY data change can rotate or
rearrange the whole layout. This module freezes a fitted reducer on disk
so layouts stay comparable across runs:

- first run fits UMAP, stores {keys(sha256), coords, reducer} as a pickle
- later runs reuse stored coords for known images (by content hash) and
  ``reducer.transform()`` new ones into the same space, then extend the
  reference so they are "known" next time
- the frame is rebuilt on demand, or automatically when the embedding
  dimensionality / component count changes or the pickle is unreadable

Trade-off (documented in the UI): ``transform()`` placements are
approximate compared to a fresh fit — rebuild when the dataset has
drifted far from the original reference.

Framework-free: no streamlit imports, unit-testable.
"""
from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import umap

_REF_VERSION = 1


def ref_path_for(folder: Path, model_name: str) -> Path:
    return Path(folder) / f"embeddings_{model_name}" / "umap_ref.pkl"


def load_ref(path: Path) -> dict | None:
    """Load a reference frame; any corruption/incompatibility → None."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            ref = pickle.load(f)
        if (ref.get("version") == _REF_VERSION and "reducer" in ref
                and len(ref.get("keys", [])) == len(ref.get("coords", []))):
            return ref
    except Exception:
        pass
    return None


def save_ref(path: Path, ref: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ref["version"] = _REF_VERSION
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(ref, f)
    tmp.replace(path)


def stable_umap(
    embeddings: np.ndarray,
    keys: Sequence[str],
    ref_path: Path,
    n_components: int,
    n_neighbors: int,
    rebuild: bool = False,
) -> tuple[np.ndarray, int, bool]:
    """Project ``embeddings`` (rows identified by content-hash ``keys``)
    in a persistent reference frame.

    Returns (coords, n_new_points, refitted). ``refitted`` is True when a
    fresh fit replaced the frame (first run, rebuild, or incompatibility).
    """
    embeddings = np.asarray(embeddings)
    keys = list(keys)
    ref = None if rebuild else load_ref(ref_path)
    if (ref is not None
            and ref["dim"] == embeddings.shape[1]
            and ref["coords"].shape[1] == n_components):
        known = {k: i for i, k in enumerate(ref["keys"])}
        old = [i for i, k in enumerate(keys) if k in known]
        new = [i for i, k in enumerate(keys) if k not in known]
        coords = np.zeros((len(keys), n_components))
        if old:
            coords[old] = ref["coords"][[known[keys[i]] for i in old]]
        if new:
            coords[new] = ref["reducer"].transform(embeddings[new])
            # extend the frame so these points are stable from now on
            ref["keys"] = list(ref["keys"]) + [keys[i] for i in new]
            ref["coords"] = np.vstack([ref["coords"], coords[new]])
            save_ref(Path(ref_path), ref)
        return coords, len(new), False

    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        random_state=42)
    coords = np.asarray(reducer.fit_transform(embeddings))
    save_ref(Path(ref_path), {
        "keys": keys, "coords": coords, "reducer": reducer,
        "dim": int(embeddings.shape[1]),
    })
    return coords, 0, True
