from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from models import Dinov2Extractor, ImagePreprocessor, ResNetExtractor

_DEFAULT_MODELS_DIR = Path(__file__).parent.parent / "models"


def available_models(models_dir: Path = _DEFAULT_MODELS_DIR) -> list[str]:
    """Return model names found in models_dir (*.pth stems)."""
    if not models_dir.exists():
        return []
    return sorted(p.stem for p in models_dir.glob("*.pth"))


def load_model(
    model_name: str, models_dir: Path = _DEFAULT_MODELS_DIR
) -> Callable[[Path], np.ndarray]:
    """Load a model from models_dir/<model_name>.pth. Returns embed_fn(path) -> np.ndarray."""
    pth_path = models_dir / f"{model_name}.pth"
    if not pth_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {pth_path}\n"
            f"Available: {available_models(models_dir)}"
        )

    preprocessor = ImagePreprocessor()

    if model_name.startswith("resnet"):
        extractor = ResNetExtractor(arch=model_name, pth_path=pth_path)
    elif model_name.startswith("dinov2"):
        extractor = Dinov2Extractor(model_name=model_name, pth_path=pth_path)
    else:
        raise ValueError(
            f"Unknown model type '{model_name}'. "
            f"Name must start with 'resnet' or 'dinov2'."
        )

    def embed_fn(path: Path) -> np.ndarray:
        return extractor(preprocessor.preprocess(path))

    return embed_fn


def extract_embeddings(
    image_paths: list[Path],
    embed_fn: Callable[[Path], np.ndarray],
    cache_path: Path | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Extract embeddings for all images. Returns shape (N, D).

    If cache_path is given, loads from cache when filenames match;
    otherwise extracts and saves to cache_path.

    progress_cb, when given, is called as progress_cb(done, total) after
    each image; on a cache hit it is called exactly once with (total, total).
    """
    n_total = len(image_paths)
    if cache_path is not None and cache_path.exists():
        data = np.load(str(cache_path), allow_pickle=False)
        cached_names = data["filenames"].tolist()
        current_names = [p.name for p in image_paths]
        if set(cached_names) == set(current_names):
            print(f"  [cache] {cache_path}")
            if progress_cb is not None:
                progress_cb(n_total, n_total)
            name_to_idx = {name: i for i, name in enumerate(cached_names)}
            indices = [name_to_idx[p.name] for p in image_paths]
            return data["embeddings"][indices]

    emb_list = []
    for i, p in enumerate(tqdm(image_paths, desc="Extracting embeddings")):
        emb_list.append(embed_fn(p))
        if progress_cb is not None:
            progress_cb(i + 1, n_total)
    embeddings = np.stack(emb_list)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(cache_path),
            embeddings=embeddings,
            filenames=np.array([p.name for p in image_paths]),
        )
        print(f"  [cache] Saved → {cache_path}")

    return embeddings


