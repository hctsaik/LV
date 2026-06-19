from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

# `models` imports torch/torchvision (~6s) at module load. Imported LAZILY inside
# load_model / load_text_encoder so that `import _utils` (used for available_models /
# extract_embeddings, which need no torch) stays cheap and the LV UI shell starts fast.

# Model weights live in ``models/`` by default; the host platform (CIM) points
# this at a writable "model-house" via LV_MODELS_DIR so the vendored submodule
# stays thin (weights are not committed). Unset → unchanged local behaviour.
_DEFAULT_MODELS_DIR = Path(
    os.environ.get("LV_MODELS_DIR") or (Path(__file__).parent.parent / "models")
)


def available_models(models_dir: Path = _DEFAULT_MODELS_DIR) -> list[str]:
    """Return selectable model names found in models_dir. Recognises:
    - `<name>/<name>.pth`  → the per-model folder layout (see MODELS.md)
    - `<name>.pth`         → loose file (back-compat; e.g. a dropped-in resnet*.pth)
    - `chinese-clip*/`     → HF-style directory with a config.json
    """
    if not models_dir.exists():
        return []
    names = [p.stem for p in models_dir.glob("*.pth")]                       # loose
    names += [p.parent.name for p in models_dir.glob("*/*.pth")              # folder
              if p.stem == p.parent.name]
    names += [
        d.name for d in models_dir.iterdir()
        if d.is_dir() and d.name.startswith("chinese-clip")
        and (d / "config.json").exists()
    ]
    return sorted(set(names))


def supports_text_query(model_name: str) -> bool:
    """True when the model has a text tower in the same space as its image
    tower — i.e. text-to-image search (F7) is meaningful."""
    return model_name.startswith("chinese-clip")


def load_model(
    model_name: str, models_dir: Path = _DEFAULT_MODELS_DIR,
    keep_aspect: bool = False, target_res: int = 224, head: str = "cls",
) -> Callable[[Path], np.ndarray]:
    """Load a model by name. Returns embed_fn(path_or_PIL) -> np.ndarray.

    ``keep_aspect=True`` (object-crop path) preserves aspect ratio and resizes to
    multiples of 14 instead of squashing to a square — see ImagePreprocessor.
    ``target_res`` sets the resize budget (longest side); ``head`` ('cls'|'meanpool')
    picks the DINOv2 pooled output (both D=384). The returned embed_fn accepts a path
    OR an in-memory PIL image."""
    from models import (ChineseClipExtractor, Dinov2Extractor,
                        ImagePreprocessor, ResNetExtractor)
    preprocessor = ImagePreprocessor(size=target_res, keep_aspect=keep_aspect)

    if supports_text_query(model_name):
        model_dir = models_dir / model_name
        if not (model_dir / "config.json").exists():
            from model_manifest import explain
            raise FileNotFoundError(explain(
                model_name, feature="以文搜圖 (F7) / 影像 embedding",
                expected=model_dir / "config.json"))
        extractor = ChineseClipExtractor(model_dir)
    else:
        # Folder layout (models/<name>/<name>.pth) first, then loose file.
        pth_path = models_dir / model_name / f"{model_name}.pth"
        if not pth_path.exists():
            pth_path = models_dir / f"{model_name}.pth"
        if not pth_path.exists():
            from model_manifest import explain
            raise FileNotFoundError(
                explain(model_name,
                        feature="影像 embedding / 視覺化 / 相似搜尋",
                        expected=models_dir / model_name / f"{model_name}.pth")
                + f"\n  目前 models/ 內可用: {available_models(models_dir)}")
        if model_name.startswith("resnet"):
            extractor = ResNetExtractor(arch=model_name, pth_path=pth_path)
        elif model_name.startswith("dinov2"):
            extractor = Dinov2Extractor(model_name=model_name, pth_path=pth_path,
                                        head=head)
        else:
            raise ValueError(
                f"Unknown model type '{model_name}'. "
                f"Name must start with 'resnet', 'dinov2' or 'chinese-clip'."
            )

    def embed_fn(path: Path) -> np.ndarray:
        return extractor(preprocessor.preprocess(path))

    return embed_fn


def load_text_encoder(
    model_name: str, models_dir: Path = _DEFAULT_MODELS_DIR
) -> Callable[[str], np.ndarray]:
    """Text tower for a text-capable model. Returns text_fn(query) -> vector
    in the same space as that model's image embeddings."""
    if not supports_text_query(model_name):
        raise ValueError(f"Model '{model_name}' has no text tower.")
    model_dir = models_dir / model_name
    if not (model_dir / "config.json").exists():
        from model_manifest import explain
        raise FileNotFoundError(explain(
            model_name, feature="以文搜圖 (F7) · 文字塔",
            expected=model_dir / "config.json"))
    from models import ChineseClipTextEncoder
    return ChineseClipTextEncoder(model_dir)


def _cache_rows_for_keys(data, cache_keys: list[str]) -> np.ndarray | None:
    """Match a loaded cache against per-image content keys → row order.

    Returns the reordered embeddings, or None on any mismatch (treated as
    a cache miss). Duplicate keys (byte-identical images) can't be mapped
    by set — they only hit when the full key sequence matches exactly.
    """
    if "keys" not in data.files:
        return None  # legacy filename-validated cache → stale by definition
    cached = data["keys"].tolist()
    if cached == cache_keys:
        return data["embeddings"]
    if len(set(cached)) != len(cached) or len(set(cache_keys)) != len(cache_keys):
        return None
    if set(cached) != set(cache_keys):
        return None
    key_to_idx = {k: i for i, k in enumerate(cached)}
    return data["embeddings"][[key_to_idx[k] for k in cache_keys]]


def extract_embeddings(
    image_paths: list[Path],
    embed_fn: Callable[[Path], np.ndarray],
    cache_path: Path | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    cache_keys: list[str] | None = None,
) -> np.ndarray:
    """Extract embeddings for all images. Returns shape (N, D).

    If cache_path is given, loads from cache when the images match;
    otherwise extracts and saves to cache_path.

    cache_keys, when given, are per-image CONTENT keys (e.g. the manifest's
    sha256) used to validate and reorder the cache instead of bare
    filenames — a changed file with an unchanged name can then never serve
    stale embeddings. Caches written before keys existed are treated as
    stale when keys are provided. Without cache_keys the legacy
    filename-set validation applies (back-compat for CLI callers).

    progress_cb, when given, is called as progress_cb(done, total) after
    each image; on a cache hit it is called exactly once with (total, total).
    """
    n_total = len(image_paths)
    if cache_keys is not None and len(cache_keys) != n_total:
        raise ValueError(
            f"cache_keys length {len(cache_keys)} != image count {n_total}")
    # per-item identity key: content key (sha256) when given, else filename
    item_keys = ([str(k) for k in cache_keys] if cache_keys is not None
                 else [p.name for p in image_paths])

    # ── load existing cache → key→row (incremental: reuse what's already done) ──
    cached_emb: dict = {}
    cached_name: dict = {}
    if cache_path is not None and Path(cache_path).exists():
        try:
            with np.load(str(cache_path), allow_pickle=False) as data:  # close → Windows replace
                emb_all = data["embeddings"]
                if "keys" in data.files:
                    ks = [str(k) for k in data["keys"].tolist()]
                elif "filenames" in data.files:
                    ks = [str(k) for k in data["filenames"].tolist()]
                else:
                    ks = []
                fns = ([str(k) for k in data["filenames"].tolist()]
                       if "filenames" in data.files else ks)
                for j, k in enumerate(ks):
                    cached_emb[k] = emb_all[j]
                    cached_name[k] = fns[j] if j < len(fns) else k
        except (OSError, ValueError, KeyError):
            cached_emb, cached_name = {}, {}

    if item_keys and all(k in cached_emb for k in item_keys):  # full hit
        print(f"  [cache] {cache_path}")
        if progress_cb is not None:
            progress_cb(n_total, n_total)
        return np.stack([cached_emb[k] for k in item_keys])

    def _flush() -> None:  # checkpoint: atomic, .npz-suffixed temp (np.savez appends otherwise)
        if cache_path is None or not cached_emb:
            return
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        ks = list(cached_emb.keys())
        arr = np.stack([cached_emb[k] for k in ks])
        fns = np.array([cached_name.get(k, k) for k in ks])
        tmp = cp.with_name(cp.stem + ".part.npz")
        np.savez(str(tmp), embeddings=arr, keys=np.array(ks), filenames=fns)
        tmp.replace(cp)

    # ── compute only the missing items, checkpoint every 200 ──
    done = sum(1 for k in item_keys if k in cached_emb)
    if progress_cb is not None:
        progress_cb(done, n_total)
    since = 0
    for i, p in enumerate(image_paths):
        k = item_keys[i]
        if k in cached_emb:
            continue
        cached_emb[k] = np.asarray(embed_fn(p))
        cached_name[k] = p.name
        done += 1
        since += 1
        if progress_cb is not None:
            progress_cb(done, n_total)
        if since >= 200:
            _flush()
            since = 0
    if since:
        _flush()
    return np.stack([cached_emb[k] for k in item_keys])


