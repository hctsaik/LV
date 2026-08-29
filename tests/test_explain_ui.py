"""Focused contracts for the Streamlit-independent parts of explain_ui."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from explain_ui import (
    _explainability_capabilities,
    _gradcam_guidance,
    _gradcam_preflight,
    _normal_bank_guide_state,
    _source_id,
    _train_session_head,
)


def test_anomaly_object_source_ids_do_not_collapse_same_image():
    base = {"path": "same.jpg", "image_path": "same.jpg", "bbox": (0.5, 0.5, 0.2, 0.2)}
    assert _source_id({**base, "obj_index": 0}) != _source_id({**base, "obj_index": 1})


def test_session_head_preflight_matches_its_visualize_input(tmp_path):
    """A session head must remain usable without a hidden global UI recipe."""
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8)).save(image_path)
    records = [
        {"path": str(image_path), "label": "scratch"},
        {"path": str(image_path), "label": "bubble"},
    ] * 8
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(len(records), 384)).astype(np.float32)
    prep = {
        "keep_aspect": False,
        "target_res": 224,
        "pooling": "cls",
        "pad": None,
        "scope": "whole_image",
    }
    head = _train_session_head(records, embeddings, "dinov2_vits14", prep)
    assert head["explain_diagnostics"]["available"] is True
    assert head["explain_diagnostics"]["kind"] == "sample_stratified_oof"
    ok, reason = _gradcam_preflight(records[0], "dinov2_vits14", "whole_image", head)
    assert ok, reason


def test_shared_panel_reports_context_capabilities(tmp_path):
    """Visualize can offer Grad-CAM setup without claiming it is ready."""
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8)).save(image_path)
    record = {"path": str(image_path), "label": "scratch"}
    capabilities = _explainability_capabilities(
        record=record,
        model_name="dinov2_vits14",
        scope="whole_image",
        classifier_head=None,
        embedding_recipe=None,
        session_head_can_train=True,
        anomaly_map_available=False,
    )

    assert capabilities["attention"]["state"] == "ready"
    assert capabilities["gradcam"]["state"] == "setup"
    assert "session head" in capabilities["gradcam"]["reason"]
    assert capabilities["anomaly"]["state"] == "unavailable"


def test_shared_panel_only_enables_anomaly_heatmap_with_context_bank(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8)).save(image_path)
    record = {"path": str(image_path), "label": "scratch"}
    capabilities = _explainability_capabilities(
        record=record,
        model_name="dinov2_vits14",
        scope="whole_image",
        classifier_head=None,
        embedding_recipe=None,
        session_head_can_train=False,
        anomaly_map_available=True,
    )

    assert capabilities["anomaly"]["state"] == "ready"
    assert capabilities["anomaly"]["reason"] == "可用"


def test_guidance_blocks_action_when_decision_and_evidence_are_unstable():
    result = SimpleNamespace(
        is_predicted_target=True,
        probability_margin=0.002,
        predicted_class="poke",
        head_diagnostics={"available": True, "balanced_accuracy": 0.49},
        sample_oof={"predicted": "scratch"},
        faithfulness={"available": True, "status": "weak"},
    )
    guidance = _gradcam_guidance(result)
    titles = {item["title"] for item in guidance}
    assert "top-1 與 runner-up 太接近" in titles
    assert "分類頭的 OOF 泛化偏弱" in titles
    assert "此圖的 OOF 判斷不穩定" in titles
    assert "熱圖尚未通過遮蔽忠實度" in titles


def test_normal_bank_guide_starts_at_patch_build():
    guide = _normal_bank_guide_state({})
    assert guide["stage"] == "build"
    assert guide["step"] == "① 建模 / 載入模型"
    assert "建立 patch Normal Bank" in guide["button"]


def test_normal_bank_guide_advances_to_apply_after_patch_model():
    guide = _normal_bank_guide_state({
        "anomaly_model": {
            "meta": {"score_mode": "patch"},
            "bank_vectors": np.ones((2, 4), dtype=np.float32),
        },
    })
    assert guide["stage"] == "apply"
    assert guide["step"] == "② 套用偵測"


def test_normal_bank_guide_advances_to_inspector_after_apply():
    guide = _normal_bank_guide_state({
        "anomaly_model": {
            "meta": {"score_mode": "patch"},
            "bank_vectors": np.ones((2, 4), dtype=np.float32),
        },
        "anomaly_apply_result": {"bank": object()},
    })
    assert guide["stage"] == "inspect"
    assert "選圖查看熱圖" in guide["button"]


def test_object_level_model_does_not_claim_patch_bank_exists():
    guide = _normal_bank_guide_state({
        "anomaly_model": {
            "meta": {"score_mode": "object"},
            "bank_vectors": np.ones((2, 4), dtype=np.float32),
        },
    })
    assert guide["stage"] == "build"
