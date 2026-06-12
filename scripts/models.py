from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

ImageInput = Union[str, Path, Image.Image]

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class ImagePreprocessor:
    def __init__(self, size: int = 224) -> None:
        self.size = size

    def preprocess(self, image: ImageInput) -> Image.Image:
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                return img.convert("RGB").resize((self.size, self.size))
        if not isinstance(image, Image.Image):
            raise TypeError("Image must be a PIL Image, str, or Path")
        return image.convert("RGB").resize((self.size, self.size))


class ResNetExtractor:
    """ResNet feature extractor — removes FC head, outputs global avg pool features."""

    def __init__(self, arch: str, pth_path: Path) -> None:
        self.device = _DEVICE
        model = getattr(tvm, arch)(weights=None)
        state_dict = torch.load(str(pth_path), map_location=self.device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        self.model = nn.Sequential(*list(model.children())[:-1]).to(self.device)
        self.model.eval()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def __call__(self, image: Any) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor).flatten(1)
        return feat.squeeze(0).cpu().numpy()


_CHINESE_CLIP_CACHE: dict[str, tuple] = {}
_OPENCC_T2S = None


def normalize_zh_query(text: str) -> str:
    """Traditional→Simplified normalization for text queries.

    Chinese-CLIP's training corpus is predominantly Simplified; converting
    Traditional queries (the primary input here) closes that gap. ASCII /
    English passes through untouched, so mixed and English-only queries
    are unaffected. Falls back to the raw text if opencc is unavailable.
    """
    global _OPENCC_T2S
    if _OPENCC_T2S is None:
        try:
            from opencc import OpenCC
            _OPENCC_T2S = OpenCC("t2s")
        except ImportError:
            _OPENCC_T2S = False
    if _OPENCC_T2S:
        return _OPENCC_T2S.convert(text)
    return text


def _load_chinese_clip(model_dir: Path):
    """Load (model, processor) once per directory — the image extractor and
    the text encoder share the same ~700MB tower pair."""
    key = str(model_dir)
    if key not in _CHINESE_CLIP_CACHE:
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        model = ChineseCLIPModel.from_pretrained(key, local_files_only=True)
        model = model.to(_DEVICE)
        model.eval()
        processor = ChineseCLIPProcessor.from_pretrained(key, local_files_only=True)
        _CHINESE_CLIP_CACHE[key] = (model, processor)
    return _CHINESE_CLIP_CACHE[key]


class ChineseClipExtractor:
    """Chinese-CLIP image tower — embeddings live in the SAME space as the
    text tower, which is what makes text-to-image search (F7) possible.
    Not interchangeable with ResNet/DINOv2 embeddings."""

    def __init__(self, model_dir: Path) -> None:
        self.device = _DEVICE
        self.model, self.processor = _load_chinese_clip(Path(model_dir))

    def __call__(self, image: Any) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.model.get_image_features(**inputs)
        return feat.squeeze(0).cpu().numpy()


class ChineseClipTextEncoder:
    """Chinese-CLIP text tower — encodes a (Chinese) query string into the
    shared text-image space."""

    def __init__(self, model_dir: Path) -> None:
        self.device = _DEVICE
        self.model, self.processor = _load_chinese_clip(Path(model_dir))

    def __call__(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[normalize_zh_query(text)], padding=True,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.model.get_text_features(**inputs)
        return feat.squeeze(0).cpu().numpy()


_DINOV2_HUB_DIR = Path(__file__).parent / "dinov2_hub"


class Dinov2Extractor:
    """DINOv2 feature extractor — architecture and weights both loaded locally."""

    def __init__(self, model_name: str, pth_path: Path) -> None:
        self.device = _DEVICE
        model = torch.hub.load(
            str(_DINOV2_HUB_DIR), model_name,
            source="local", pretrained=False,
        )
        state_dict = torch.load(str(pth_path), map_location=self.device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        self.model = model.to(self.device)
        self.model.eval()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def __call__(self, image: Any) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)  # CLS token, shape (1, D)
        return feat.squeeze(0).cpu().numpy()
