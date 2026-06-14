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
    """Return model names found in models_dir: *.pth stems plus HF-style
    chinese-clip* directories (downloaded via scripts/download_chinese_clip.py)."""
    if not models_dir.exists():
        return []
    names = [p.stem for p in models_dir.glob("*.pth")]
    names += [
        d.name for d in models_dir.iterdir()
        if d.is_dir() and d.name.startswith("chinese-clip")
        and (d / "config.json").exists()
    ]
    return sorted(names)


def supports_text_query(model_name: str) -> bool:
    """True when the model has a text tower in the same space as its image
    tower — i.e. text-to-image search (F7) is meaningful."""
    return model_name.startswith("chinese-clip")


def load_model(
    model_name: str, models_dir: Path = _DEFAULT_MODELS_DIR
) -> Callable[[Path], np.ndarray]:
    """Load a model by name. Returns embed_fn(path) -> np.ndarray."""
    from models import (ChineseClipExtractor, Dinov2Extractor,
                        ImagePreprocessor, ResNetExtractor)
    preprocessor = ImagePreprocessor()

    if supports_text_query(model_name):
        model_dir = models_dir / model_name
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"Chinese-CLIP weights not found: {model_dir}\n"
                "Run scripts/download_chinese_clip.py first."
            )
        extractor = ChineseClipExtractor(model_dir)
    else:
        pth_path = models_dir / f"{model_name}.pth"
        if not pth_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {pth_path}\n"
                f"Available: {available_models(models_dir)}"
            )
        if model_name.startswith("resnet"):
            extractor = ResNetExtractor(arch=model_name, pth_path=pth_path)
        elif model_name.startswith("dinov2"):
            extractor = Dinov2Extractor(model_name=model_name, pth_path=pth_path)
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
        raise FileNotFoundError(
            f"Chinese-CLIP weights not found: {model_dir}\n"
            "Run scripts/download_chinese_clip.py first."
        )
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
    if cache_path is not None and cache_path.exists():
        data = np.load(str(cache_path), allow_pickle=False)
        cached_rows = None
        if cache_keys is not None:
            cached_rows = _cache_rows_for_keys(data, cache_keys)
        else:
            cached_names = data["filenames"].tolist()
            current_names = [p.name for p in image_paths]
            if set(cached_names) == set(current_names):
                name_to_idx = {name: i for i, name in enumerate(cached_names)}
                cached_rows = data["embeddings"][[name_to_idx[p.name]
                                                  for p in image_paths]]
        if cached_rows is not None:
            print(f"  [cache] {cache_path}")
            if progress_cb is not None:
                progress_cb(n_total, n_total)
            return cached_rows

    emb_list = []
    for i, p in enumerate(tqdm(image_paths, desc="Extracting embeddings")):
        emb_list.append(embed_fn(p))
        if progress_cb is not None:
            progress_cb(i + 1, n_total)
    embeddings = np.stack(emb_list)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            embeddings=embeddings,
            filenames=np.array([p.name for p in image_paths]),
        )
        if cache_keys is not None:
            arrays["keys"] = np.array(cache_keys)
        np.savez(str(cache_path), **arrays)
        print(f"  [cache] Saved → {cache_path}")

    return embeddings


