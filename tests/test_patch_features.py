"""驗收:patch_features(設計 01_patch_features.md)。先紅:scripts/patch_features.py 未實作。
用 fake extractor → 不需真模型 / 權重。"""
import numpy as np
import pytest
from PIL import Image

from patch_features import extract_patch_grid, embed_objects_patch
from interaction import discover_yolo_objects
from object_eval import list_images, classes_for


def test_square_crop_grid_and_shape(color_patch_extractor):  # AC1 + AC2
    ext = color_patch_extractor()
    crop = Image.new("RGB", (224, 224), (0, 170, 0))  # 224=16*14 → grid 16x16
    pf = extract_patch_grid(crop, extractor=ext, target_res=224)
    assert pf["grid"] == (16, 16)
    assert pf["feats"].shape == (256, 384)
    norms = np.linalg.norm(pf["feats"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)         # AC2 per-token L2


def test_grid_product_equals_P_invariant(synthetic_yolo_dataset, color_patch_extractor):  # AC3
    ext = color_patch_extractor()
    root = synthetic_yolo_dataset["root"]
    meta = discover_yolo_objects(list_images(root), classes_for(root))
    feats = embed_objects_patch(meta, target_res=224, extractor=ext)
    assert len(feats) == len(meta)
    for pf in feats:
        gh, gw = pf["grid"]
        assert pf["feats"].shape[0] == gh * gw       # 不變量
        assert pf["feats"].shape[1] == 384


def test_cache_roundtrip_no_recompute(synthetic_yolo_dataset, color_patch_extractor, tmp_path):  # AC4
    base = color_patch_extractor()
    calls = {"n": 0}

    def counting(crop):
        calls["n"] += 1
        return base(crop)

    root = synthetic_yolo_dataset["root"]
    meta = discover_yolo_objects(list_images(root), classes_for(root))
    cache = tmp_path / "patchcache"
    first = embed_objects_patch(meta, target_res=224, extractor=counting, cache_dir=cache)
    assert calls["n"] > 0
    calls["n"] = 0
    second = embed_objects_patch(meta, target_res=224, extractor=counting, cache_dir=cache)
    assert calls["n"] == 0                            # 全命中,不重算
    for a, b in zip(first, second):
        assert a["grid"] == b["grid"]
        assert np.allclose(a["feats"], b["feats"], atol=1e-3)  # float16 容差


def test_no_dataset_pollution(synthetic_yolo_dataset, color_patch_extractor, tmp_path):  # AC5
    ext = color_patch_extractor()
    root = synthetic_yolo_dataset["root"]
    before = set(root.rglob("*"))
    meta = discover_yolo_objects(list_images(root), classes_for(root))
    embed_objects_patch(meta, target_res=224, extractor=ext, cache_dir=tmp_path / "c")
    assert set(root.rglob("*")) == before            # 資料集目錄零新增 [[no-dataset-writes]]


def test_bad_image_yields_zero_grid(color_patch_extractor, tmp_path):  # AC6(邊界,推導)
    ext = color_patch_extractor()
    meta = [{"image_path": tmp_path / "does_not_exist.jpg",
             "bbox": (0.5, 0.5, 0.8, 0.8), "obj_index": 0,
             "label": "good", "class_id": 0, "score": None}]
    feats = embed_objects_patch(meta, target_res=224, extractor=ext)
    assert feats[0]["grid"] == (1, 1)
    assert feats[0]["feats"].shape == (1, 384)
    assert np.allclose(feats[0]["feats"], 0.0)
