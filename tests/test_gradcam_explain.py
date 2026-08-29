"""驗收:gradcam_explain —— token Grad-CAM、禁 target、缺 provenance 拒圖。"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest
from PIL import Image

from _tiny_vit import TinyExplainer, TinyViT
from dino_explain import ExplainRequest
from dino_head import train_head
from gradcam_explain import GradCAMError, signed_norm, token_gradcam_map
from head_adapter import attach_provenance


def _rgb(h=32, w=32) -> Image.Image:
    rng = np.random.default_rng(3)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), "RGB")


def _head_for(expl: TinyExplainer, n=20):
    model = expl.model.eval()
    rng = np.random.default_rng(4)
    X, y = [], []
    for i in range(n):
        arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        t = expl.transform(img).unsqueeze(0)
        with torch_no_grad():
            feat = model(t).detach().cpu().numpy().reshape(-1)
        X.append(feat)
        y.append("pos" if i % 2 == 0 else "neg")
    X = np.stack(X).astype(np.float32)
    # 若幾乎不可分，沿第一維拉開標籤
    X[: n // 2, 0] += 3.0
    X[n // 2:, 0] -= 3.0
    y = np.array(["pos"] * (n // 2) + ["neg"] * (n - n // 2))
    head = train_head(X, y)
    return attach_provenance(
        head,
        model_id="tiny", checkpoint_id="tiny", embedding_dim=int(X.shape[1]),
        keep_aspect=False, target_res=32, pooling="cls", l2norm=True,
        scope="whole_image", pad=None,
    )


def torch_no_grad():
    import torch
    return torch.no_grad()


def _req(img=None, pooling="cls"):
    return ExplainRequest(
        image=img or _rgb(), source_id="g0", model_id="tiny",
        keep_aspect=False, target_res=32, pooling=pooling,
    )


def test_gradcam_shape_and_nonzero_cls():
    expl = TinyExplainer(TinyViT())
    head = _head_for(expl)
    r = token_gradcam_map(_req(), head, target_class="pos", explainer=expl)
    assert r.grid == (4, 4)
    assert r.raw_patch_scores.shape == (4, 4)
    assert r.target_class == "pos"
    assert float(np.abs(r.raw_patch_scores).sum()) > 0
    assert np.isfinite(r.raw_patch_scores).all()
    assert float(np.abs(r.display_patch_scores).max()) == pytest.approx(1.0)
    assert r.reference_class == "neg"
    assert r.target_mode == "contrastive_margin"
    assert r.target_score == pytest.approx(
        np.log(r.target_probability / r.reference_probability), abs=1e-4)
    assert r.faithfulness["status"] in {
        "supported", "weak", "no_positive_evidence",
    }


def test_gradcam_meanpool_shape():
    expl = TinyExplainer(TinyViT())
    # 用 cls 訓的頭 dim 仍=16，meanpool embedding 同維
    head = _head_for(expl)
    head = attach_provenance(head, pooling="meanpool")
    r = token_gradcam_map(
        _req(pooling="meanpool"), head, target_class="neg", explainer=expl)
    assert r.raw_patch_scores.shape == (4, 4)
    assert r.pooling == "meanpool"


def test_rejects_normal_and_unknown():
    expl = TinyExplainer(TinyViT())
    head = _head_for(expl)
    with pytest.raises(GradCAMError, match="正常"):
        token_gradcam_map(_req(), head, target_class="正常", explainer=expl)
    with pytest.raises(GradCAMError, match="Unknown"):
        token_gradcam_map(_req(), head, target_class="Unknown", explainer=expl)


def test_legacy_head_rejected():
    expl = TinyExplainer(TinyViT())
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 16)).astype(np.float32)
    y = np.array(["a"] * 10 + ["b"] * 10)
    raw = train_head(X, y)
    with pytest.raises(GradCAMError, match="provenance"):
        token_gradcam_map(_req(), raw, target_class="a", explainer=expl)


def test_unknown_class_rejected():
    expl = TinyExplainer(TinyViT())
    head = _head_for(expl)
    with pytest.raises(GradCAMError):
        token_gradcam_map(_req(), head, target_class="not-a-class", explainer=expl)


def test_none_target_explains_actual_top1():
    expl = TinyExplainer(TinyViT())
    head = _head_for(expl)
    r = token_gradcam_map(
        _req(), head, target_class=None, explainer=expl, run_faithfulness=False)
    assert r.target_class == r.predicted_class
    assert r.is_predicted_target is True
    assert r.predicted_probability >= r.runner_up_probability
    assert r.probability_margin >= 0.0


def test_signed_normalization_never_turns_negative_only_cam_positive():
    display = signed_norm(np.array([[-4.0, -2.0, 0.0]], dtype=np.float32))
    assert display.min() == pytest.approx(-1.0)
    assert display.max() == pytest.approx(0.0)
