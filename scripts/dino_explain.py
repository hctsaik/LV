"""DINO CLS→patch attention（重建 softmax，不是 fused kernel 的真實 weights）。

單張、按需、request-local hook。不改 vendored attention.py、不寫 cache。
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_LOCK = threading.Lock()
_EXPLAINER_CACHE: dict[str, Any] = {}

SCHEMA_VERSION = 1
METHODS = ("last_layer_head_mean", "single_head", "rollout")
MAX_PATCHES_DEFAULT = 32 * 32
MAX_PATCHES_ROLLOUT = 16 * 16

WARNINGS = (
    "DINO Attention 不是瑕疵分割，也不等於模型的因果理由。",
    "亮度是該圖相對值，不能直接跨圖比較。",
    "patch grid 是低解析度空間資訊，邊界不是 pixel-accurate segmentation。",
    "本圖是重建的 CLS→patch softmax，不是 SDPA / xFormers fused kernel 的真實 weights。",
    "無 register token 的最後一層 CLS 常點亮高-norm / 邊角 token，亮區不是缺陷。",
)


class ExplainError(ValueError):
    """使用者可讀的解釋失敗（壞圖、非 DINO、解析度超限、不相容模型）。"""


@dataclass
class ExplainRequest:
    image: Any
    source_id: str
    scope: str = "whole_image"
    model_id: str = "dinov2_vits14"
    checkpoint_path: str | Path | None = None
    keep_aspect: bool = False
    target_res: int = 224
    pooling: str = "cls"
    method: str = "last_layer_head_mean"
    head_index: int | None = None
    original_size: tuple[int, int] | None = None
    crop_bbox_px: tuple[int, int, int, int] | None = None
    pad: float | None = None
    max_patches: int = MAX_PATCHES_DEFAULT


@dataclass
class AttentionResult:
    raw_patch_scores: np.ndarray
    display_patch_scores: np.ndarray
    grid: tuple[int, int]
    patch_size: int
    n_register: int
    n_cls: int
    cls_mass: float
    register_mass: float
    patch_mass: float
    method: str
    head_index: int | None
    model_id: str
    checkpoint_id: str
    input_size: tuple[int, int]
    original_size: tuple[int, int] | None
    crop_bbox_px: tuple[int, int, int, int] | None
    keep_aspect: bool
    target_res: int
    pooling: str
    scope: str
    source_id: str
    input_image: Image.Image
    timing_s: float
    warnings: tuple[str, ...] = WARNINGS
    schema_version: int = SCHEMA_VERSION
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["raw_patch_scores"] = self.raw_patch_scores
        d["display_patch_scores"] = self.display_patch_scores
        d["input_image"] = self.input_image
        return d


def checkpoint_identity(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        st = os.stat(p)
        return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return p.name


def max_norm(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float32)
    m = float(s.max()) if s.size else 0.0
    if m <= 0:
        return np.zeros_like(s, dtype=np.float32)
    return s / m


def attn_weights_from_qkv(qkv, num_heads: int):
    """qkv (B, N, 3C) → softmax attention (B, heads, N, N)."""
    import torch

    if qkv.ndim != 3:
        raise ExplainError(f"qkv 維度應為 (B,N,3C)，得到 {tuple(qkv.shape)}")
    b, n, three_c = qkv.shape
    if three_c % (3 * num_heads) != 0:
        raise ExplainError(f"qkv last dim {three_c} 無法分成 3×{num_heads} heads")
    head_dim = three_c // (3 * num_heads)
    qkv = qkv.reshape(b, n, 3, num_heads, head_dim)
    q, k, _v = qkv.unbind(2)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    scale = head_dim ** -0.5
    return torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)


def slice_cls_to_patches(attn, n_register: int = 0):
    """attn (B,H,N,N) → CLS query 對 patch keys (B,H,P)。"""
    start = 1 + int(n_register)
    if attn.shape[-1] <= start:
        raise ExplainError("token 數不足以切出 patch（檢查 register token 計數）")
    return attn[:, :, 0, start:]


def iter_transformer_blocks(model):
    blocks = []
    raw = getattr(model, "blocks", None)
    if raw is None:
        raise ExplainError("模型沒有 .blocks，無法取 attention")
    for item in raw:
        if hasattr(item, "attn") and hasattr(item.attn, "qkv"):
            blocks.append(item)
            continue
        inner = list(item) if hasattr(item, "__iter__") and not hasattr(item, "attn") else []
        for ch in inner:
            if hasattr(ch, "attn") and hasattr(ch.attn, "qkv"):
                blocks.append(ch)
    if not blocks:
        raise ExplainError("找不到帶 attn.qkv 的 transformer block")
    return blocks


def _resolve_checkpoint(model_id: str, checkpoint_path) -> Path:
    from _utils import _DEFAULT_MODELS_DIR

    if checkpoint_path:
        p = Path(checkpoint_path)
        if p.exists():
            return p
    pth = _DEFAULT_MODELS_DIR / model_id / f"{model_id}.pth"
    if pth.exists():
        return pth
    loose = _DEFAULT_MODELS_DIR / f"{model_id}.pth"
    if loose.exists():
        return loose
    raise ExplainError(f"找不到 DINO checkpoint：{model_id}")


def get_explainer(model_id: str, checkpoint_path=None, *, head: str = "cls"):
    """Request-local 專用 extractor（快取權重，不與 Visualize 活實例共用 hook）。"""
    if not str(model_id).startswith("dinov2"):
        raise ExplainError(f"可解釋性只支援本機 DINOv2，目前是 {model_id}")
    pth = _resolve_checkpoint(model_id, checkpoint_path)
    key = f"{model_id}|{pth.resolve()}|{head}"
    with _LOCK:
        ext = _EXPLAINER_CACHE.get(key)
        if ext is None:
            from models import Dinov2Extractor
            ext = Dinov2Extractor(model_name=model_id, pth_path=pth, head=head)
            _EXPLAINER_CACHE[key] = ext
        return ext, pth


def _pil_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        from safe_io import safe_open_image
        im = safe_open_image(image)
        if im is None:
            raise ExplainError("無法讀取影像（壞檔或格式錯誤）")
        return im.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ExplainError("影像陣列維度不正確")
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def _preprocess(req: ExplainRequest, model, explainer):
    from models import ImagePreprocessor

    img = _pil_rgb(req.image)
    patch = int(getattr(model, "patch_size", 14) or 14)
    proc = ImagePreprocessor(
        size=int(req.target_res), keep_aspect=bool(req.keep_aspect), patch=patch,
    ).preprocess(img)
    tensor = explainer.transform(proc).unsqueeze(0).to(explainer.device)
    gh = int(tensor.shape[-2]) // patch
    gw = int(tensor.shape[-1]) // patch
    if gh < 1 or gw < 1:
        raise ExplainError("前處理後影像小於一個 patch")
    n_patches = gh * gw
    limit = int(req.max_patches)
    if req.method == "rollout":
        limit = min(limit, MAX_PATCHES_ROLLOUT)
    if n_patches > limit:
        raise ExplainError(
            f"解析度過大：{gh}×{gw}={n_patches} patches（上限 {limit}）。"
            "請降低 target_res，不會靜默降級。"
        )
    return img, proc, tensor, patch, (gh, gw)


def _hook_qkv(modules, store: dict, *, all_layers: bool):
    handles = []

    def _capture(idx):
        def _fn(_m, _inp, out):
            store.setdefault("qkv", {})[idx] = out
        return _fn

    for i, blk in enumerate(modules):
        if all_layers or i == len(modules) - 1:
            handles.append(blk.attn.qkv.register_forward_hook(_capture(i)))
    return handles


def _scores_from_attn(attn, n_register: int, grid: tuple[int, int],
                      method: str, head_index: int | None):
    import torch

    b, h, n, _ = attn.shape
    cls_row = attn[:, :, 0, :]
    start = 1 + int(n_register)
    patch = slice_cls_to_patches(attn, n_register)
    if method == "single_head":
        if head_index is None or not (0 <= int(head_index) < h):
            raise ExplainError(f"single_head 需要 0..{h - 1} 的 head_index")
        scores = patch[0, int(head_index)]
    else:
        scores = patch[0].mean(dim=0)
        head_index = None
    gh, gw = grid
    if scores.numel() != gh * gw:
        raise ExplainError(
            f"patch 數 {int(scores.numel())} 與 grid {gh}×{gw} 不符（可能轉置或 register 切錯）"
        )
    raw = scores.detach().float().cpu().numpy().reshape(gh, gw)
    cls_mass = float(cls_row[0].mean(dim=0)[0].detach().cpu())
    if n_register:
        register_mass = float(cls_row[0].mean(dim=0)[1:start].sum().detach().cpu())
    else:
        register_mass = 0.0
    patch_mass = float(raw.sum())
    return raw, head_index, cls_mass, register_mass, patch_mass


def _rollout_maps(qkv_by_layer: dict, num_heads: int, n_register: int,
                  grid: tuple[int, int]):
    import torch

    layers = [qkv_by_layer[i] for i in sorted(qkv_by_layer)]
    if not layers:
        raise ExplainError("rollout 沒有抓到任何層的 qkv")
    mats = []
    eye = None
    for qkv in layers:
        attn = attn_weights_from_qkv(qkv, num_heads).mean(dim=1)[0]
        n = attn.shape[0]
        if eye is None:
            eye = torch.eye(n, device=attn.device, dtype=attn.dtype)
        mats.append(0.5 * eye + 0.5 * attn)
    r = mats[0]
    for m in mats[1:]:
        r = m @ r
    start = 1 + int(n_register)
    scores = r[0, start:]
    gh, gw = grid
    if scores.numel() != gh * gw:
        raise ExplainError(f"rollout patch 數 {int(scores.numel())} ≠ {gh}×{gw}")
    raw = scores.detach().float().cpu().numpy().reshape(gh, gw)
    cls_mass = float(r[0, 0].detach().cpu())
    register_mass = float(r[0, 1:start].sum().detach().cpu()) if n_register else 0.0
    return raw, cls_mass, register_mass, float(raw.sum())


def attention_map(request: ExplainRequest, *, explainer=None) -> AttentionResult:
    import torch

    if request.method not in METHODS:
        raise ExplainError(f"未知 method：{request.method}")
    t0 = time.perf_counter()
    pth = None
    if explainer is None:
        explainer, pth = get_explainer(
            request.model_id, request.checkpoint_path, head=request.pooling or "cls")
    else:
        if request.checkpoint_path:
            pth = Path(request.checkpoint_path)
    model = explainer.model
    model.eval()
    _img, proc, tensor, patch, grid = _preprocess(request, model, explainer)
    n_register = int(getattr(model, "num_register_tokens", 0) or 0)
    blocks = iter_transformer_blocks(model)
    last = blocks[-1]
    num_heads = int(
        getattr(last.attn, "num_heads", 0)
        or getattr(model, "num_heads", 0)
        or 0
    )
    if num_heads <= 0:
        raise ExplainError("無法推斷 attention head 數")

    captured: dict = {}
    handles = []
    all_layers = request.method == "rollout"
    try:
        with _LOCK:
            handles = _hook_qkv(blocks, captured, all_layers=all_layers)
            with torch.no_grad():
                _ = model.forward_features(tensor) if hasattr(model, "forward_features") else model(tensor)
    except ExplainError:
        raise
    except Exception as exc:
        raise ExplainError(f"DINO forward 失敗：{exc}") from exc
    finally:
        for h in handles:
            h.remove()

    qkv_map = captured.get("qkv") or {}
    if request.method == "rollout":
        raw, cls_mass, register_mass, patch_mass = _rollout_maps(
            qkv_map, num_heads, n_register, grid)
        used_head = None
    else:
        last_i = max(qkv_map) if qkv_map else None
        if last_i is None:
            raise ExplainError("hook 沒有抓到 qkv（模型結構與預期不符）")
        attn = attn_weights_from_qkv(qkv_map[last_i], num_heads)
        if not torch.isfinite(attn).all():
            raise ExplainError("attention 含 NaN/Inf")
        raw, used_head, cls_mass, register_mass, patch_mass = _scores_from_attn(
            attn, n_register, grid, request.method, request.head_index)

    display = max_norm(raw)
    warnings = list(WARNINGS)
    if request.method == "rollout":
        warnings.append("Attention rollout 是跨層啟發式乘積，不是更高階的因果解釋。")
    orig = request.original_size or _pil_rgb(request.image).size
    return AttentionResult(
        raw_patch_scores=raw.astype(np.float32),
        display_patch_scores=display.astype(np.float32),
        grid=grid,
        patch_size=patch,
        n_register=n_register,
        n_cls=1,
        cls_mass=cls_mass,
        register_mass=register_mass,
        patch_mass=patch_mass,
        method=request.method,
        head_index=used_head,
        model_id=request.model_id,
        checkpoint_id=checkpoint_identity(pth),
        input_size=proc.size,
        original_size=orig,
        crop_bbox_px=request.crop_bbox_px,
        keep_aspect=bool(request.keep_aspect),
        target_res=int(request.target_res),
        pooling=request.pooling,
        scope=request.scope,
        source_id=request.source_id,
        input_image=proc,
        timing_s=time.perf_counter() - t0,
        warnings=tuple(warnings),
        extras={"token_layout": "[CLS]+[R]+[patches]", "reconstructed": True},
    )


def recipe_key(req: ExplainRequest) -> tuple:
    return (
        req.source_id, req.scope, req.model_id, req.method, req.head_index,
        req.keep_aspect, req.target_res, req.pooling, req.pad,
    )


def source_fingerprint(image: Image.Image) -> str:
    arr = np.asarray(image.convert("RGB"))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]
