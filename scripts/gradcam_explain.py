"""Token-aware Grad-CAM。CLS target 鉤最後一層 pre-attention；meanpool 鉤被池化的 patch。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from dino_explain import (
    ExplainError,
    ExplainRequest,
    _pil_rgb,
    _preprocess,
    checkpoint_identity,
    get_explainer,
    iter_transformer_blocks,
)
from head_adapter import (
    FrozenLogisticHead,
    from_sklearn_head,
    head_can_gradcam,
    is_forbidden_target,
)

WARNINGS = (
    "Token Grad-CAM 解釋 target 相對最強競爭類別的差異，不代表整體品質或機台判定。",
    "正常 / Unknown 是 cascade 結果，不是分類頭類別，不能做 Grad-CAM。",
    "patch grid 不是 pixel-accurate segmentation。",
    "紅＝支持 target；藍＝支持 reference；強度是該圖相對值，不能跨圖比較。",
    "這是最後一層 token 的局部代理；必須搭配遮蔽忠實度與 OOF 品質判讀。",
)


class GradCAMError(ExplainError):
    pass


@dataclass
class GradCAMResult:
    raw_patch_scores: np.ndarray
    display_patch_scores: np.ndarray
    grid: tuple[int, int]
    patch_size: int
    target_class: str
    target_score: float
    target_index: int
    reference_class: str
    reference_index: int
    target_mode: str
    predicted_class: str
    predicted_probability: float
    runner_up_class: str
    runner_up_probability: float
    probability_margin: float
    target_probability: float
    reference_probability: float
    is_predicted_target: bool
    positive_mass: float
    negative_mass: float
    faithfulness: dict
    head_diagnostics: dict
    sample_oof: dict | None
    classes: list
    model_id: str
    checkpoint_id: str
    pooling: str
    keep_aspect: bool
    target_res: int
    scope: str
    source_id: str
    input_size: tuple[int, int]
    original_size: tuple[int, int] | None
    input_image: Image.Image
    timing_s: float
    warnings: tuple[str, ...] = WARNINGS
    schema_version: int = 2

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def signed_norm(scores: np.ndarray) -> np.ndarray:
    """Symmetric [-1, 1] normalization that never invents positive evidence."""
    s = np.asarray(scores, dtype=np.float32)
    scale = float(np.abs(s).max()) if s.size else 0.0
    if scale <= 0:
        return np.zeros_like(s, dtype=np.float32)
    return (s / scale).astype(np.float32)


def _target_index(classes: list, target: str | int | None) -> tuple[int, str]:
    if target is None:
        raise GradCAMError("target=None 必須在模型預測後解析")
    if is_forbidden_target(target):
        raise GradCAMError(f"「{target}」不是可微分分類 target，拒開 Grad-CAM")
    if isinstance(target, int):
        if not (0 <= target < len(classes)):
            raise GradCAMError(f"target index {target} 超出 classes")
        lab = str(classes[target])
        if is_forbidden_target(lab):
            raise GradCAMError(f"「{lab}」不是可微分分類 target")
        return target, lab
    names = [str(c) for c in classes]
    if str(target) not in names:
        raise GradCAMError(f"target「{target}」不在 head.classes={names}")
    return names.index(str(target)), str(target)


def _pooled_embedding(feats, pooling: str):
    import torch.nn.functional as F
    if pooling == "meanpool":
        patch = F.normalize(feats["x_norm_patchtokens"], dim=-1)
        return patch.mean(dim=1)
    return feats["x_norm_clstoken"]


def _occlusion_audit(model, tensor, adapter, *, pooling: str, grid: tuple[int, int],
                     patch: int, signed_scores: np.ndarray, target_idx: int,
                     reference_idx: int, original_score: float,
                     fraction: float = 0.10) -> dict:
    """Batched top/bottom/random patch deletion faithfulness probe."""
    import torch

    flat = np.asarray(signed_scores, dtype=np.float32).reshape(-1)
    if flat.size == 0 or float(flat.max()) <= 0:
        return {
            "available": False, "status": "no_positive_evidence",
            "reason": "CAM 沒有正向 target 證據，拒絕製造熱點。",
        }
    n = max(1, int(np.ceil(flat.size * float(fraction))))
    order = np.argsort(flat)
    rng = np.random.default_rng(0)
    picks = {
        "top": order[-n:],
        "bottom": order[:n],
        "random": rng.choice(flat.size, size=n, replace=False),
    }
    baseline = tensor.mean(dim=(-2, -1), keepdim=True)
    gh, gw = grid
    variants = []
    for indices in picks.values():
        x = tensor.detach().clone()
        for flat_i in indices:
            yy, xx = divmod(int(flat_i), int(gw))
            y0, x0 = yy * patch, xx * patch
            x[:, :, y0:y0 + patch, x0:x0 + patch] = baseline
        variants.append(x)
    batch = torch.cat(variants, dim=0)
    with torch.no_grad():
        feats = model.forward_features(batch)
        emb = _pooled_embedding(feats, pooling)
        logits = adapter.logits_torch(emb)
        values = adapter.contrastive_score_torch(
            logits, target_idx, reference_idx).detach().float().cpu().numpy()
    occluded = {name: float(values[i]) for i, name in enumerate(picks)}
    drops = {name: float(original_score - value) for name, value in occluded.items()}
    supported = drops["top"] > max(0.0, drops["random"]) and drops["top"] > drops["bottom"]
    return {
        "available": True,
        "method": "mean_color_patch_deletion",
        "fraction": float(fraction),
        "n_patches": int(n),
        "original_margin": float(original_score),
        "occluded_margins": occluded,
        "margin_drops": drops,
        "top_minus_random_drop": float(drops["top"] - drops["random"]),
        "status": "supported" if supported else "weak",
    }


def token_gradcam_map(
    request: ExplainRequest,
    head: dict,
    *,
    target_class: str | int | None,
    explainer=None,
    adapter: FrozenLogisticHead | None = None,
    run_faithfulness: bool = True,
) -> GradCAMResult:
    import torch
    import torch.nn.functional as F

    # Reject a legacy / internally inconsistent artifact before loading a
    # foundation model.  The complete input recipe is checked after the model
    # is available, so embedding_dim is not merely copied from the head.
    ok, reason = head_can_gradcam(head, None)
    if not ok:
        raise GradCAMError(reason)

    t0 = time.perf_counter()
    pth = None
    if explainer is None:
        explainer, pth = get_explainer(
            request.model_id, request.checkpoint_path, head=request.pooling or "cls")
    elif request.checkpoint_path:
        pth = request.checkpoint_path
    model = explainer.model
    model.eval()
    model_dim = getattr(model, "embed_dim", None)
    recipe = {
        "model_id": request.model_id,
        "keep_aspect": bool(request.keep_aspect),
        "target_res": int(request.target_res),
        "pooling": request.pooling,
        "scope": request.scope,
        "l2norm": head.get("l2norm", True),
        "pad": request.pad,
    }
    if pth is not None:
        recipe["checkpoint_id"] = checkpoint_identity(pth)
    if model_dim not in (None, ""):
        recipe["embedding_dim"] = int(model_dim)
    ok, reason = head_can_gradcam(head, recipe)
    if not ok:
        raise GradCAMError(reason)
    adapter = adapter or from_sklearn_head(head)

    _img, proc, tensor, patch, grid = _preprocess(request, model, explainer)
    n_register = int(getattr(model, "num_register_tokens", 0) or 0)
    start = 1 + n_register
    pooling = (request.pooling or "cls").lower()
    blocks = iter_transformer_blocks(model)
    last = blocks[-1]
    acts: dict[str, torch.Tensor] = {}

    def _pre_hook(_m, _inp, out):
        out.retain_grad()
        acts["pre"] = out
        return out

    def _final_hook(_m, _inp, out):
        out.retain_grad()
        acts["final"] = out
        return out

    handle = last.norm1.register_forward_hook(_pre_hook)
    handle_f = None
    if hasattr(model, "norm") and pooling == "meanpool":
        handle_f = model.norm.register_forward_hook(_final_hook)
    try:
        tensor = tensor.detach().requires_grad_(False)
        feats = model.forward_features(tensor) if hasattr(model, "forward_features") else None
        if feats is None:
            raise GradCAMError("模型沒有 forward_features，無法做 token Grad-CAM")
        if pooling == "meanpool":
            full = acts.get("final")
            if full is None:
                raise GradCAMError("沒有抓到 final-norm token activation")
            patch_tok = full[:, start:, :]
            patch_n = F.normalize(patch_tok, dim=-1)
            emb = patch_n.mean(dim=1)
        else:
            emb = feats["x_norm_clstoken"]
            if acts.get("pre") is None:
                raise GradCAMError("沒有抓到 last-block norm1 activation")
        logits = adapter.logits_torch(emb)
        probs = adapter.probabilities_torch(logits)
        predicted_idx = int(probs[0].argmax().detach().cpu())
        if target_class is None:
            idx, lab = predicted_idx, str(adapter.classes[predicted_idx])
        else:
            idx, lab = _target_index(adapter.classes, target_class)
        rival_logits = logits[0].detach().clone()
        rival_logits[idx] = -torch.inf
        reference_idx = int(rival_logits.argmax().cpu())
        reference_lab = str(adapter.classes[reference_idx])
        score = adapter.contrastive_score_torch(logits, idx, reference_idx)
        if score.ndim == 1:
            score = score[0]
        if score.grad_fn is None:
            raise GradCAMError("target 對計算圖無梯度（不可做 Grad-CAM）")
        score.backward()
        if pooling == "meanpool":
            gfull = acts["final"].grad
            if gfull is None:
                raise GradCAMError("meanpool patch token 梯度為 None")
            activation = acts["final"].detach()[:, start:, :]
            gradient = gfull[:, start:, :]
        else:
            g = acts["pre"].grad
            if g is None:
                raise GradCAMError("pre-attention token 梯度為 None")
            activation = acts["pre"].detach()[:, start:, :]
            gradient = g[:, start:, :]
        # Classical Grad-CAM: average the target gradient over spatial tokens
        # per channel, then use those channel weights to score each token.
        channel_weights = gradient.mean(dim=1, keepdim=True)
        cam = (activation * channel_weights).sum(dim=-1)[0]
        if not torch.isfinite(cam).all():
            raise GradCAMError("Grad-CAM 含 NaN/Inf")
        if float(cam.abs().sum()) == 0.0:
            raise GradCAMError("指定 target 的梯度為零，拒出圖")
        gh, gw = grid
        if cam.numel() != gh * gw:
            raise GradCAMError(f"CAM 長度 {int(cam.numel())} ≠ grid {gh}×{gw}")
        raw = cam.detach().float().cpu().numpy().reshape(gh, gw).astype(np.float32)
        target_score = float(score.detach().cpu())
        prob_np = probs[0].detach().float().cpu().numpy()
        ranking = np.argsort(prob_np)[::-1]
        runner_idx = int(ranking[1])
    except GradCAMError:
        raise
    except Exception as exc:
        raise GradCAMError(f"Grad-CAM 失敗：{exc}") from exc
    finally:
        handle.remove()
        if handle_f is not None:
            handle_f.remove()
        model.zero_grad(set_to_none=True)

    faithfulness = (
        _occlusion_audit(
            model, tensor, adapter, pooling=pooling, grid=grid, patch=patch,
            signed_scores=raw, target_idx=idx, reference_idx=reference_idx,
            original_score=target_score,
        ) if run_faithfulness else {"available": False, "status": "not_run"}
    )
    diagnostics = dict(head.get("explain_diagnostics") or {})
    sample_oof = dict((diagnostics.get("samples") or {}).get(request.source_id) or {}) or None
    diagnostics.pop("samples", None)

    return GradCAMResult(
        raw_patch_scores=raw,
        display_patch_scores=signed_norm(raw),
        grid=grid,
        patch_size=patch,
        target_class=lab,
        target_score=target_score,
        target_index=idx,
        reference_class=reference_lab,
        reference_index=reference_idx,
        target_mode="contrastive_margin",
        predicted_class=str(adapter.classes[predicted_idx]),
        predicted_probability=float(prob_np[predicted_idx]),
        runner_up_class=str(adapter.classes[runner_idx]),
        runner_up_probability=float(prob_np[runner_idx]),
        probability_margin=float(prob_np[predicted_idx] - prob_np[runner_idx]),
        target_probability=float(prob_np[idx]),
        reference_probability=float(prob_np[reference_idx]),
        is_predicted_target=bool(idx == predicted_idx),
        positive_mass=float(np.clip(raw, 0, None).sum()),
        negative_mass=float(np.clip(-raw, 0, None).sum()),
        faithfulness=faithfulness,
        head_diagnostics=diagnostics,
        sample_oof=sample_oof,
        classes=list(adapter.classes),
        model_id=request.model_id,
        checkpoint_id=checkpoint_identity(pth),
        pooling=pooling,
        keep_aspect=bool(request.keep_aspect),
        target_res=int(request.target_res),
        scope=request.scope,
        source_id=request.source_id,
        input_size=proc.size,
        original_size=request.original_size or _pil_rgb(request.image).size,
        input_image=proc,
        timing_s=time.perf_counter() - t0,
        warnings=WARNINGS,
    )
