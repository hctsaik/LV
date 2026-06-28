"""驗收:anomaly_bank_store —— bank profile 目錄序列化 + 相容鍵 + few-shot 內容定址 + 白名單防呆。
先紅:scripts/anomaly_bank_store.py 尚未實作。

綜合設計裁決:meta.json 最後寫當完整性哨兵;硬鍵 model/target_res/dim 不符拒載;
few-shot 主鍵 (image_stem, 量化 bbox, label) — obj_index 來自 label 行序、重標註會變,只當 tie-break;
寫硬碟正向白名單防呆,絕不寫使用者資料集。
"""
import json

import numpy as np
import pytest

from anomaly_bank_store import (assert_safe_bank_dir, check_compat,
                                confirmed_to_fewshot, load_bank,
                                match_fewshot_to_indices, save_bank)

META = {"model": "dinov2_vits14", "target_res": 224, "patch_dim": 384, "obj_dim": 384}


def _unit(n, d, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _records():
    return [
        {"image_path": "/ds/img_001.jpg", "bbox": [0.10, 0.10, 0.20, 0.20], "obj_index": 0, "label": "vase"},
        {"image_path": "/ds/img_001.jpg", "bbox": [0.50, 0.50, 0.10, 0.10], "obj_index": 1, "label": "bowl"},
        {"image_path": "/ds/img_002.jpg", "bbox": [0.30, 0.30, 0.20, 0.20], "obj_index": 0, "label": "vase"},
    ]


def test_vectors_roundtrip_cosine(tmp_path):  # AC1:fp32 存讀 cosine 一致(<1e-5)
    v = _unit(500, 384)
    save_bank(tmp_path / "bk", vectors=v, meta=META)
    loaded = load_bank(tmp_path / "bk")
    assert loaded["vectors"].shape == v.shape
    assert np.all(np.abs((loaded["vectors"] * v).sum(1) - 1.0) < 1e-5)


def test_projection_only_no_bank(tmp_path):  # AC2:object 模式 — 只存 projection、無 bank.npz
    import anomaly_project
    basis = anomaly_project.fit_projector(_unit(200, 384))
    save_bank(tmp_path / "bk", projection=basis, meta={**META, "patch_dim": None})
    loaded = load_bank(tmp_path / "bk")
    assert loaded["vectors"] is None
    assert loaded["projection"] is not None
    assert np.allclose(loaded["projection"]["mean"], basis["mean"], atol=1e-4)


def test_load_without_meta_raises(tmp_path):  # AC3:無 meta.json(不完整)→ raise
    d = tmp_path / "bk"
    d.mkdir()
    np.savez_compressed(str(d / "bank.npz"), vectors=_unit(10, 384))
    with pytest.raises((FileNotFoundError, ValueError)):
        load_bank(d)


def test_meta_written_last_as_sentinel(tmp_path):  # AC3b:meta 最後寫且含 schema_version
    save_bank(tmp_path / "bk", vectors=_unit(10, 384), meta=META)
    meta = json.loads((tmp_path / "bk" / "meta.json").read_text(encoding="utf-8"))
    assert "schema_version" in meta and meta["model"] == "dinov2_vits14"


def test_check_compat_hard_keys(tmp_path):  # AC4:硬鍵相符 pass、不符 raise
    save_bank(tmp_path / "bk", vectors=_unit(10, 384), meta=META)
    meta = load_bank(tmp_path / "bk")["meta"]
    check_compat(meta, model="dinov2_vits14", target_res=224, patch_dim=384, obj_dim=384)  # 不 raise
    with pytest.raises(ValueError):
        check_compat(meta, model="dinov2_vitb14", target_res=224)        # 換模型
    with pytest.raises(ValueError):
        check_compat(meta, model="dinov2_vits14", target_res=336)        # 換解析度
    with pytest.raises(ValueError):
        check_compat(meta, model="dinov2_vits14", target_res=224, obj_dim=768)  # 維度不符


def test_empty_bank_rejected(tmp_path):  # AC5:空 bank 拒存
    with pytest.raises(ValueError):
        save_bank(tmp_path / "bk", vectors=np.zeros((0, 384), np.float32), meta=META)


def test_fewshot_roundtrip(tmp_path):  # AC6:confirmed→fewshot→match 還原回相同索引
    recs = _records()
    confirmed = {0: "good", 2: "bad"}
    fs = confirmed_to_fewshot(confirmed, recs)
    save_bank(tmp_path / "bk", vectors=_unit(10, 384), meta=META, fewshot=fs)
    fs2 = load_bank(tmp_path / "bk")["fewshot"]
    back, unmatched = match_fewshot_to_indices(fs2, recs)
    assert back == {0: "good", 2: "bad"}
    assert unmatched == 0


def test_match_survives_reordering(tmp_path):  # AC7:行序被打亂(obj_index 變)但 bbox 在 → 救回對齊
    recs = _records()
    fs = confirmed_to_fewshot({0: "good", 1: "bad"}, recs)
    shuffled = list(reversed(recs))  # index 0↔2 對調
    back, unmatched = match_fewshot_to_indices(fs, shuffled)
    # 原 index0(img_001/vase/bbox0.1) 在 shuffled 變 index2;原 index1 仍 img_001/bowl
    assert unmatched == 0
    assert back[2] == "good"          # img_001 vase
    assert back[1] == "bad"           # img_001 bowl(reversed 後仍 index1)


def test_match_rejects_bbox_mismatch():  # AC8:同 stem 同 label 但 bbox 差很多 → 未配,不靜默貼錯
    recs = _records()
    fs = confirmed_to_fewshot({0: "good"}, recs)
    fs[0]["bbox"] = [0.9, 0.9, 0.05, 0.05]  # 移到完全不同位置
    back, unmatched = match_fewshot_to_indices(fs, recs)
    assert unmatched == 1 and back == {}


def test_resave_clears_stale_files(tmp_path):  # AC10(對抗 review):重存清舊檔,不殘留被新 meta 收編
    import anomaly_project
    d = tmp_path / "bk"
    save_bank(d, vectors=_unit(10, 384), meta=META)            # 先 patch 模式(有 bank.npz)
    assert (d / "bank.npz").exists()
    basis = anomaly_project.fit_projector(_unit(50, 384))
    save_bank(d, projection=basis, meta={**META, "patch_dim": None, "score_mode": "object"})  # object 重存
    assert not (d / "bank.npz").exists(), "object 重存應清掉舊 bank.npz(否則被新 meta 收編 → mismatch)"
    loaded = load_bank(d)
    assert loaded["vectors"] is None and loaded["meta"].get("score_mode") == "object"


def test_no_tmp_leftover(tmp_path):  # AC11(對抗 review):atomic 寫不留 _tmp_ 殘檔
    import anomaly_project
    save_bank(tmp_path / "bk", vectors=_unit(10, 384),
              projection=anomaly_project.fit_projector(_unit(50, 384)), meta=META, fewshot=[])
    assert not any(p.name.startswith("_tmp_") for p in (tmp_path / "bk").iterdir())


def test_safe_dir_rejects_flat_yolo(tmp_path, monkeypatch):  # AC12(對抗 review):扁平 YOLO 也擋
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    flat = tmp_path / "flat_ds"
    flat.mkdir()
    (flat / "a.jpg").write_bytes(b"x")
    (flat / "a.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
    with pytest.raises(ValueError):
        assert_safe_bank_dir(flat)                              # 同層影像+.txt → 拒(即使 selected_folders 空)


def test_safe_bank_dir_whitelist(tmp_path, monkeypatch):  # AC9:白名單防呆
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    assert_safe_bank_dir(tmp_path / "cache" / "anomaly_bank" / "x")      # .lv_cache 下 → 安全
    assert_safe_bank_dir(tmp_path / "clean_dir")                          # 乾淨目錄 → 安全
    ds = tmp_path / "dataset"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir()
    with pytest.raises(ValueError):
        assert_safe_bank_dir(ds)                                         # 含 images/labels → 拒
    with pytest.raises(ValueError):
        assert_safe_bank_dir(ds, selected_folders=[str(ds)])            # 就是已選資料夾 → 拒
