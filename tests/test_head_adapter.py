"""驗收:head_adapter —— 與 dino_head.predict_head 對齊；legacy 拒 Grad-CAM。"""
from __future__ import annotations

import numpy as np
import pytest

from dino_head import fit_temperature, predict_head, train_head
from head_adapter import (
    adapter_matches_predict_head,
    attach_provenance,
    from_sklearn_head,
    head_can_gradcam,
    is_forbidden_target,
    missing_provenance_keys,
)

D = 16
CLASSES = ["bubble", "scratch"]


def _data(n=40, seed=1, k=2):
    rng = np.random.default_rng(seed)
    X, y = [], []
    names = CLASSES if k == 2 else ["bubble", "scratch", "particle"]
    for c, name in enumerate(names):
        center = np.zeros(D, dtype=np.float32)
        center[c] = 5.0
        X.append(center + rng.normal(0, 0.3, (n, D)).astype(np.float32))
        y += [name] * n
    return np.vstack(X), np.array(y)


def test_binary_matches_predict_head():
    X, y = _data(k=2)
    head = train_head(X, y)
    assert adapter_matches_predict_head(head, X)
    ad = from_sklearn_head(head)
    p, c, pr = ad.predict_numpy(X)
    p2, c2, pr2 = predict_head(head, X)
    assert list(p) == list(p2)
    assert np.allclose(pr, pr2, atol=1e-5)


def test_multiclass_matches_predict_head():
    X, y = _data(k=3)
    head = train_head(X, y)
    assert adapter_matches_predict_head(head, X)


def test_temperature_matches():
    X, y = _data(k=3)
    head = fit_temperature(train_head(X, y), X, y)
    assert "temperature" in head
    assert adapter_matches_predict_head(head, X, atol=1e-4)


def test_l2norm_off():
    X, y = _data(k=2)
    head = train_head(X, y, l2norm=False)
    assert adapter_matches_predict_head(head, X)


def test_legacy_missing_provenance_blocks_gradcam():
    X, y = _data()
    head = train_head(X, y)
    ok, reason = head_can_gradcam(head)
    assert ok is False
    assert "provenance" in reason
    assert missing_provenance_keys(head)


def test_attached_provenance_allows_when_recipe_matches():
    X, y = _data()
    head = attach_provenance(
        train_head(X, y),
        model_id="dinov2_vits14", checkpoint_id="abc", embedding_dim=D,
        keep_aspect=False, target_res=224, pooling="cls", l2norm=True,
        scope="whole_image", pad=None,
    )
    ok, reason = head_can_gradcam(head)
    assert ok, reason
    ok2, reason2 = head_can_gradcam(head, {
        "model_id": "dinov2_vits14", "checkpoint_id": "abc",
        "keep_aspect": False, "target_res": 224, "pooling": "cls",
        "embedding_dim": D, "scope": "whole_image", "l2norm": True,
        "pad": None,
    })
    assert ok2, reason2
    ok3, _ = head_can_gradcam(head, {
        "model_id": "dinov2_vits14", "checkpoint_id": "OTHER",
        "keep_aspect": False, "target_res": 224, "pooling": "cls",
    })
    assert ok3 is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("scope", "object_crop"), ("l2norm", False), ("pad", 0.12)],
)
def test_provenance_rejects_input_recipe_mismatch(field, value):
    X, y = _data()
    head = attach_provenance(
        train_head(X, y),
        model_id="dinov2_vits14", checkpoint_id="abc", embedding_dim=D,
        keep_aspect=False, target_res=224, pooling="cls", l2norm=True,
        scope="whole_image", pad=None,
    )
    recipe = {
        "model_id": "dinov2_vits14", "checkpoint_id": "abc",
        "embedding_dim": D, "keep_aspect": False, "target_res": 224,
        "pooling": "cls", "l2norm": True, "scope": "whole_image",
        "pad": None,
    }
    recipe[field] = value
    ok, reason = head_can_gradcam(head, recipe)
    assert not ok
    assert field in reason


def test_provenance_rejects_crop_pad_when_request_is_whole_image():
    """``pad=None`` is a real whole-image value, not an omitted constraint."""
    X, y = _data()
    head = attach_provenance(
        train_head(X, y),
        model_id="dinov2_vits14", checkpoint_id="abc", embedding_dim=D,
        keep_aspect=True, target_res=224, pooling="cls", l2norm=True,
        scope="whole_image", pad=0.12,
    )
    ok, reason = head_can_gradcam(head, {
        "model_id": "dinov2_vits14", "checkpoint_id": "abc",
        "embedding_dim": D, "keep_aspect": True, "target_res": 224,
        "pooling": "cls", "l2norm": True, "scope": "whole_image",
        "pad": None,
    })
    assert not ok
    assert "pad" in reason


def test_forbidden_targets():
    assert is_forbidden_target("正常")
    assert is_forbidden_target("Unknown")
    assert not is_forbidden_target("bubble")


def test_contrastive_score_uses_selected_or_strongest_reference():
    import torch

    X, y = _data(k=3)
    adapter = from_sklearn_head(train_head(X, y))
    logits = torch.tensor([[1.5, 0.2, 1.1]], dtype=torch.float32)
    selected = adapter.contrastive_score_torch(logits, 0, 1)
    strongest = adapter.contrastive_score_torch(logits, 0)
    assert float(selected[0]) == pytest.approx(1.3)
    assert float(strongest[0]) == pytest.approx(0.4)


def test_adapter_probabilities_sum_to_one():
    import torch

    X, y = _data(k=3)
    adapter = from_sklearn_head(train_head(X, y))
    logits = adapter.logits_torch(torch.tensor(X[:3]))
    probs = adapter.probabilities_torch(logits)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-6)
