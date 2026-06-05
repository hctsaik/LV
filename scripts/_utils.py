from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.services.model.clip import Clip
from retrieval.services.model.dinov2 import Dinov2
from retrieval.services.utils.image_process import ImagePreprocessor

MULTIMODAL_REGISTRY = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip2_base": "google/siglip2-base-patch16-224",
}


def load_model(model_name: str) -> Callable[[Path], np.ndarray]:
    """Return embed_fn(image_path) -> np.ndarray."""
    preprocessor = ImagePreprocessor()
    if model_name == "dinov2":
        model = Dinov2(model_size="small")

        def embed_fn(path: Path) -> np.ndarray:
            return model(preprocessor.preprocess(path))
    else:
        hf_name = MULTIMODAL_REGISTRY.get(model_name)
        if hf_name is None:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose: dinov2, siglip2_base, clip"
            )
        model = Clip(model_name=hf_name)

        def embed_fn(path: Path) -> np.ndarray:
            return model.extract_image_features(preprocessor.preprocess(path))

    return embed_fn


def extract_embeddings(
    image_paths: list[Path], embed_fn: Callable[[Path], np.ndarray]
) -> np.ndarray:
    """Extract embeddings for all images. Returns shape (N, D)."""
    return np.stack([
        embed_fn(p) for p in tqdm(image_paths, desc="Extracting embeddings")
    ])


def save_figure(plotly_fig, mpl_fig, output_dir: Path, name: str) -> None:
    """Save Plotly figure as .html and Matplotlib figure as .png."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plotly_fig.write_html(str(output_dir / f"{name}.html"))
    mpl_fig.savefig(str(output_dir / f"{name}.png"), dpi=150, bbox_inches="tight")
