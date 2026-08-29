"""驗收:dino_explain —— reconstructed CLS→patch attention。"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest
import torch
from PIL import Image

from _tiny_vit import TinyExplainer, TinyViT
from dino_explain import (
    ExplainError,
    ExplainRequest,
    attn_weights_from_qkv,
    attention_map,
    iter_transformer_blocks,
    max_norm,
    slice_cls_to_patches,
)


def _rgb(h=32, w=32, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), "RGB")


def _req(**kw) -> ExplainRequest:
    base = dict(
        image=_rgb(), source_id="t0", model_id="tiny",
        keep_aspect=False, target_res=32, pooling="cls",
    )
    base.update(kw)
    return ExplainRequest(**base)


def test_attn_from_qkv_softmax_and_heads():
    b, n, h, d = 1, 5, 2, 4
    qkv = torch.randn(b, n, 3 * h * d)
    attn = attn_weights_from_qkv(qkv, h)
    assert attn.shape == (1, 2, 5, 5)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(1, 2, 5), atol=1e-5)
    assert torch.isfinite(attn).all()


def test_slice_skips_cls_and_registers():
    attn = torch.zeros(1, 2, 8, 8)
    attn[:, :, 0, :] = torch.arange(8).float()
    p = slice_cls_to_patches(attn, n_register=3)
    assert p.shape == (1, 2, 4)
    assert torch.equal(p[0, 0], torch.tensor([4., 5., 6., 7.]))


def test_max_norm_not_minmax():
    s = np.array([[0.2, 0.4], [0.4, 0.8]], dtype=np.float32)
    d = max_norm(s)
    assert d.max() == pytest.approx(1.0)
    assert d.min() == pytest.approx(0.25)  # 0.2/0.8，不是 0
    assert max_norm(np.zeros((2, 2))).sum() == 0


def test_last_layer_head_mean_grid_square():
    model = TinyViT(n_register=0)
    expl = TinyExplainer(model)
    r = attention_map(_req(), explainer=expl)
    assert r.grid == (4, 4)
    assert r.raw_patch_scores.shape == (4, 4)
    assert r.display_patch_scores.shape == (4, 4)
    assert r.n_register == 0
    assert np.isfinite(r.raw_patch_scores).all()
    assert r.raw_patch_scores.sum() == pytest.approx(r.patch_mass, rel=1e-5)
    assert r.cls_mass + r.register_mass + r.patch_mass == pytest.approx(
        r.cls_mass + r.patch_mass, abs=1e-5)


def test_non_square_grid_not_transposed():
    model = TinyViT(n_register=0)
    expl = TinyExplainer(model)
    img = _rgb(h=16, w=32)
    r = attention_map(
        _req(image=img, keep_aspect=True, target_res=32), explainer=expl)
    gh, gw = r.grid
    assert (gh, gw) == (r.input_image.size[1] // 8, r.input_image.size[0] // 8)
    assert r.raw_patch_scores.shape == (gh, gw)
    assert gh != gw


def test_register_token_slicing():
    model = TinyViT(n_register=4)
    expl = TinyExplainer(model)
    r = attention_map(_req(), explainer=expl)
    assert r.n_register == 4
    assert r.raw_patch_scores.shape == (4, 4)
    assert r.register_mass >= 0


def test_single_head_and_head_mean_differ_or_equal_but_valid():
    torch.manual_seed(0)
    model = TinyViT(heads=2)
    expl = TinyExplainer(model)
    mean = attention_map(_req(), explainer=expl)
    h0 = attention_map(_req(method="single_head", head_index=0), explainer=expl)
    h1 = attention_map(_req(method="single_head", head_index=1), explainer=expl)
    assert h0.head_index == 0 and h1.head_index == 1
    assert mean.raw_patch_scores.shape == h0.raw_patch_scores.shape
    mid = (h0.raw_patch_scores + h1.raw_patch_scores) / 2
    assert np.allclose(mean.raw_patch_scores, mid, atol=1e-5)


def test_single_head_out_of_range():
    expl = TinyExplainer(TinyViT(heads=2))
    with pytest.raises(ExplainError):
        attention_map(_req(method="single_head", head_index=9), explainer=expl)


def test_rollout_shape():
    expl = TinyExplainer(TinyViT(depth=2))
    r = attention_map(_req(method="rollout"), explainer=expl)
    assert r.grid == (4, 4)
    assert any("rollout" in w.lower() for w in r.warnings)


def test_resolution_guard_refuses():
    expl = TinyExplainer(TinyViT())
    with pytest.raises(ExplainError, match="解析度過大"):
        attention_map(_req(max_patches=4), explainer=expl)


def test_unknown_method():
    expl = TinyExplainer(TinyViT())
    with pytest.raises(ExplainError):
        attention_map(_req(method="nope"), explainer=expl)


def test_hook_removed_after_success_and_exception(monkeypatch):
    model = TinyViT()
    expl = TinyExplainer(model)
    last = iter_transformer_blocks(model)[-1]
    n_before = len(last.attn.qkv._forward_hooks)
    attention_map(_req(), explainer=expl)
    assert len(last.attn.qkv._forward_hooks) == n_before

    def boom(*_a, **_k):
        raise RuntimeError("fwd-boom")

    monkeypatch.setattr(model, "forward_features", boom)
    with pytest.raises(ExplainError, match="fwd-boom"):
        attention_map(_req(), explainer=expl)
    assert len(last.attn.qkv._forward_hooks) == n_before


def test_cls_embedding_unchanged_by_hook():
    model = TinyViT()
    expl = TinyExplainer(model)
    img = _rgb()
    from models import ImagePreprocessor
    proc = ImagePreprocessor(size=32, keep_aspect=False, patch=8).preprocess(img)
    t = expl.transform(proc).unsqueeze(0)
    with torch.no_grad():
        before = model(t).clone()
    attention_map(_req(image=img), explainer=expl)
    with torch.no_grad():
        after = model(t)
    assert torch.allclose(before, after, atol=1e-6)


def test_warnings_are_first_class():
    r = attention_map(_req(), explainer=TinyExplainer(TinyViT()))
    joined = " ".join(r.warnings)
    assert "不是瑕疵" in joined
    assert "重建" in joined


@pytest.mark.skipif(
    not (Path("models/dinov2_vits14/dinov2_vits14.pth").exists()
         or Path("models/dinov2_vits14.pth").exists()),
    reason="本機無 dinov2_vits14",
)
def test_real_vits14_224_is_16x16():
    img = _rgb(224, 224, seed=1)
    r = attention_map(ExplainRequest(
        image=img, source_id="real", model_id="dinov2_vits14",
        keep_aspect=False, target_res=224,
    ))
    assert r.grid == (16, 16)
    assert r.raw_patch_scores.shape == (16, 16)
    assert r.n_register == 0
    assert r.patch_size == 14
