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
