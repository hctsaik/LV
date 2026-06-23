"""patch_features:YOLO 物件框 → DINOv2 patch-grid 特徵(+ grid 形狀)並快取。

設計:3_Architect_Design/01_patch_features.md。復用 discover_yolo_objects/crop_bbox/
ImagePreprocessor 與 .lv_cache 慣例。extractor 可注入(測試免真模型)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# DINOv2 變體的輸出維度(vits=384、vitb=768…)。壞圖 fallback 用它,才不會在
# vstack 不同模型維度時炸(vitb=768 與寫死的 384 不相容)。未知名稱退 384。
_DINOV2_DIM = {"dinov2_vits14": 384, "dinov2_vitb14": 768,
               "dinov2_vitl14": 1024, "dinov2_vitg14": 1536}


def model_dim(model: str) -> int:
    return _DINOV2_DIM.get(model, 384)


def _l2n_rows(feats: np.ndarray) -> np.ndarray:
    feats = np.asarray(feats, dtype=np.float32)
    return feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)


def _real_extractor(model_name: str, target_res: int):
    """預設真實 DINOv2 patch extractor(凍結;回 (patchtokens (N,384), (gh,gw)))。"""
    import torch
    from torch.nn.functional import normalize

    from _utils import _DEFAULT_MODELS_DIR
    from models import Dinov2Extractor

    models_dir = _DEFAULT_MODELS_DIR
    pth = models_dir / model_name / f"{model_name}.pth"
    if not pth.exists():
        pth = models_dir / f"{model_name}.pth"
    ext = Dinov2Extractor(model_name=model_name, pth_path=pth, head="meanpool")

    def extractor(proc):
        t = ext.transform(proc).unsqueeze(0).to(ext.device)
        with torch.no_grad():
            patch = normalize(ext.model.forward_features(t)["x_norm_patchtokens"], dim=-1)
        feats = patch.squeeze(0).cpu().numpy().astype(np.float32)
        return feats, (t.shape[2] // 14, t.shape[3] // 14)

    return extractor


def extract_patch_grid(crop, extractor=None, *, target_res: int = 224,
                       model: str = "dinov2_vits14") -> dict:
    """單一 crop → {"feats": (P,384) L2 正規化, "grid": (gh,gw)},P==gh*gw。"""
    from models import ImagePreprocessor
    proc = ImagePreprocessor(size=target_res, keep_aspect=True).preprocess(crop)
    if extractor is None:
        extractor = _real_extractor(model, target_res)
    feats, grid = extractor(proc)
    return {"feats": _l2n_rows(feats), "grid": (int(grid[0]), int(grid[1]))}


def embed_objects_patch(meta, model: str = "dinov2_vits14", *, target_res: int = 224,
                        pad: float = 0.12, cache_dir=None, extractor=None,
                        progress=None) -> list[dict]:
    """逐物件回傳 PatchFeat(順序同 meta)。cache_dir 給定時逐物件 .npz(float16)快取。"""
    from interaction import crop_bbox
    from object_eval import _adaptive_pad_px
    from safe_io import safe_open_image

    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    out: list[dict] = [None] * len(meta)
    real_ext = None
    cur_ip, cur = None, None
    total = len(meta)
    for i, m in enumerate(meta):
        key = f"{Path(str(m['image_path'])).stem}__{m['obj_index']}"
        cpath = (cache_dir / f"{key}.npz") if cache_dir else None
        if cpath and cpath.exists():
            try:
                with np.load(str(cpath)) as d:
                    out[i] = {"feats": d["feats"].astype(np.float32),
                              "grid": (int(d["grid"][0]), int(d["grid"][1]))}
                if progress:
                    progress(i + 1, total)
                continue
            except (OSError, ValueError, KeyError):
                pass

        ip = str(m["image_path"])
        if ip != cur_ip:
            cur = safe_open_image(ip)   # 壞圖回 None → 補零保索引對齊(out[i] 仍填)
            cur_ip = ip

        if cur is None:
            pf = {"feats": np.zeros((1, model_dim(model)), dtype=np.float32),
                  "grid": (1, 1)}
        else:
            iw, ih = cur.size
            b = m["bbox"]
            crop = crop_bbox(cur, *b, pad_px=_adaptive_pad_px(b, iw, ih, pad))
            ext = extractor
            if ext is None:
                if real_ext is None:
                    real_ext = _real_extractor(model, target_res)
                ext = real_ext
            pf = extract_patch_grid(crop, extractor=ext, target_res=target_res)

        out[i] = pf
        if cpath:
            np.savez(str(cpath), feats=pf["feats"].astype(np.float16),
                     grid=np.array(pf["grid"], dtype=np.int32))
        if progress:
            progress(i + 1, total)
    return out
