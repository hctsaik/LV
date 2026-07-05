"""驗收:13 sample_bank(few-shot 樣本集持久化 + 模型綁定 + build)。

設計:3_Architect_Design/13_sample_bank.md。持久化/append/相容用合成向量(不觸真模型);
build 用小 YOLO 資料夾 + 注入 embed_fn。延遲匯入 sample_bank(先紅=failure 非 collection error)。
"""
import numpy as np
import pytest
from pathlib import Path
from PIL import Image


def _sb():
    import sample_bank
    return sample_bank


def _prov(labels):
    return [{"image_path": f"/d/{i}.jpg", "bbox": [0.5, 0.5, 0.4, 0.6], "label": labels[i]}
            for i in range(len(labels))]


def _bank(vectors, labels):
    return {"vectors": np.asarray(vectors, dtype=np.float32), "labels": np.array(labels),
            "provenance": _prov(labels), "model": "m", "target_res": 224}


def test_ac_b1_save_load_roundtrip(tmp_path):  # AC-B1
    sb = _sb()
    v = np.arange(12, dtype=np.float32).reshape(3, 4)
    labels = ["a", "a", "b"]
    bank = _bank(v, labels)
    sb.save_sample_bank(tmp_path / "sb", bank)
    got = sb.load_sample_bank(tmp_path / "sb")
    assert np.allclose(got["vectors"], v, atol=1e-6)
    assert list(got["labels"]) == labels
    assert got["provenance"] == _prov(labels)
    assert got["model"] == "m" and int(got["target_res"]) == 224


def test_ac_b2_append(tmp_path):  # AC-B2:接尾 N→N+k
    sb = _sb()
    sb.save_sample_bank(tmp_path / "sb", _bank(np.zeros((3, 4)), ["a", "a", "b"]))
    newv = np.ones((2, 4), dtype=np.float32)
    newp = [{"image_path": "/n/0.jpg", "bbox": [0.5, 0.5, 0.2, 0.2], "label": "c"},
            {"image_path": "/n/1.jpg", "bbox": [0.3, 0.3, 0.2, 0.2], "label": "c"}]
    sb.append_sample(tmp_path / "sb", vectors=newv, labels=["c", "c"], provenance=newp)
    got = sb.load_sample_bank(tmp_path / "sb")
    assert got["vectors"].shape == (5, 4)
    assert list(got["labels"])[3:] == ["c", "c"]
    assert got["provenance"][3:] == newp


def test_ac_b3_model_compat(tmp_path):  # AC-B3
    sb = _sb()
    bank = _bank(np.zeros((2, 4)), ["a", "b"])
    sb.assert_model_compatible(bank, model="m", target_res=224)   # 不 raise
    with pytest.raises(ValueError):
        sb.assert_model_compatible(bank, model="other", target_res=224)
    with pytest.raises(ValueError):
        sb.assert_model_compatible(bank, model="m", target_res=336)


def test_ac_b4_load_missing(tmp_path):  # AC-B4
    sb = _sb()
    with pytest.raises((FileNotFoundError, ValueError)):
        sb.load_sample_bank(tmp_path / "nope")


def test_ac_b5_append_dim_mismatch(tmp_path):  # AC-B5
    sb = _sb()
    sb.save_sample_bank(tmp_path / "sb", _bank(np.zeros((2, 4)), ["a", "b"]))
    with pytest.raises(ValueError):
        sb.append_sample(tmp_path / "sb", vectors=np.zeros((1, 5), dtype=np.float32),
                         labels=["c"], provenance=[{"image_path": "/n.jpg", "bbox": [0.5, 0.5, 0.2, 0.2], "label": "c"}])


# ── build(注入 embed_fn)──────────────────────────────────────────────────────
def _yolo_samples(tmp_path):
    root = tmp_path / "samples"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")
    for i, cls in enumerate([0, 1]):
        Image.new("RGB", (64, 64), (200, 120, 0) if cls == 0 else (0, 120, 200)).save(
            root / "images" / f"o{i}.jpg", quality=92)
        (root / "labels" / f"o{i}.txt").write_text(f"{cls} 0.5 0.5 0.8 0.8\n", encoding="utf-8")
    return root


def _fake_embed():
    def emb(crop):
        arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
        return np.array([arr[..., 0].mean() / 255.0, arr[..., 2].mean() / 255.0, 1.0, 0.0],
                        dtype=np.float32)
    return emb


def test_ac_b6_build_injected(tmp_path):  # AC-B6
    sb = _sb()
    bank = sb.build_sample_bank([_yolo_samples(tmp_path)], model="dinov2_vits14",
                                target_res=224, embed_fn=_fake_embed())
    assert bank["vectors"].shape[0] == 2
    assert set(bank["labels"]) == {"scratch", "stain"}
    assert len(bank["provenance"]) == 2
    assert "image_path" in bank["provenance"][0] and "bbox" in bank["provenance"][0]
    assert bank["model"] == "dinov2_vits14" and int(bank["target_res"]) == 224


def test_ac_b7_build_no_objects(tmp_path):  # AC-B7:無 labels/ → raise
    sb = _sb()
    root = tmp_path / "empty"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (64, 64), (0, 0, 0)).save(root / "images" / "o.jpg", quality=92)
    with pytest.raises(ValueError):
        sb.build_sample_bank([root], model="m", target_res=224, embed_fn=_fake_embed())


def test_ac_b8_load_feeds_multi_ref(tmp_path):  # AC-B8 衍生:往返後可直接餵 multi_ref
    import similarity
    sb = _sb()
    sb.save_sample_bank(tmp_path / "sb", _bank([[1, 0, 0, 0], [0, 1, 0, 0]], ["a", "b"]))
    got = sb.load_sample_bank(tmp_path / "sb")
    labs, sims = similarity.multi_ref_similarity([[1, 0, 0, 0]], got["vectors"], got["labels"])
    assert list(labs) == ["a"] and abs(float(np.asarray(sims)[0]) - 1.0) < 1e-6


# ── M13 Task7:訓頭導流門檻 training_head_ready(純計數,無 I/O)───────────────
def test_ac_sthr_1_two_classes_reach(tmp_path):  # AC-STHR-1:≥2 類 × 每類 ≥8 → ready
    sb = _sb()
    assert sb.training_head_ready(["a"] * 8 + ["b"] * 8) == {
        "ready": True, "per_class": {"a": 8, "b": 8}, "ready_classes": ["a", "b"]}


def test_ac_sthr_2_one_class_short(tmp_path):  # AC-STHR-2:一類差 1 顆 → 未達標,只 a 達標
    sb = _sb()
    r = sb.training_head_ready(["a"] * 8 + ["b"] * 7)
    assert r["ready"] is False
    assert r["ready_classes"] == ["a"]
    assert r["per_class"] == {"a": 8, "b": 7}


def test_ac_sthr_3_single_class(tmp_path):  # AC-STHR-3:只 1 類(數量多也不夠)→ 未達標
    sb = _sb()
    r = sb.training_head_ready(["a"] * 20)
    assert r["ready"] is False
    assert r["ready_classes"] == ["a"]


def test_ac_sthr_4_empty(tmp_path):  # AC-STHR-4:空 → 全空、未達標
    sb = _sb()
    assert sb.training_head_ready([]) == {"ready": False, "per_class": {}, "ready_classes": []}


def test_ac_sthr_5_threshold_param(tmp_path):  # AC-STHR-5 邊界:門檻可調 + 排序不變量
    sb = _sb()
    assert sb.training_head_ready(["a"] * 3 + ["b"] * 3, min_per_class=3)["ready"] is True
    assert sb.training_head_ready(["a"] * 3 + ["b"] * 3)["ready"] is False  # 預設 8 → 不達標
    # 衍生不變量:per_class / ready_classes 皆依類名排序(與輸入順序無關)
    r = sb.training_head_ready(["b"] * 8 + ["a"] * 8)
    assert list(r["per_class"].keys()) == ["a", "b"] and r["ready_classes"] == ["a", "b"]
