"""驗收:anomaly_external_bank —— run_pipeline 加 external_bank/external_ref 注入路徑。
先紅:run_pipeline 尚未支援這兩個參數。

綜合設計裁決:掛載外部 bank(patch)時短路、不重建 bank;object 模式對外部 ref 算離群;
新資料夾 patch 特徵用「傳入的 model/target_res」(=bank meta 的鎖死值),非 GUI 現值。
用注入 fake extractor/embed_fn → 不需真模型。
"""
from pathlib import Path

import numpy as np

from anomaly_score import MemoryBank
from anomaly_tool import run_pipeline
from interaction import discover_yolo_objects
from object_eval import classes_for, list_images

A = np.zeros(384, np.float32); A[0] = 1.0      # 正常方向
B = np.zeros(384, np.float32); B[1] = 1.0      # 缺陷方向


def _patch_ext(crop):
    arr = np.asarray(crop.convert("RGB")).astype(float)
    red = float((arr[..., 0] > 1.2 * (arr[..., 1] + 1)).mean()) > 0.01
    w, h = crop.size
    gh, gw = max(1, h // 14), max(1, w // 14)
    return np.repeat((B if red else A)[None, :], gh * gw, axis=0).astype(np.float32), (gh, gw)


def _emb(crop):
    t, _ = _patch_ext(crop)
    v = t.mean(0)
    return (v / (np.linalg.norm(v) or 1.0)).astype(np.float32)


def _ds(ds):
    ip, cn = list_images(ds["root"]), classes_for(ds["root"])
    meta = discover_yolo_objects(ip, cn)
    defect = {i for i, m in enumerate(meta)
              if Path(str(m["image_path"])).stem.startswith("defect")}
    return ip, cn, meta, defect


def test_external_bank_skips_build(synthetic_yolo_dataset, monkeypatch):  # AC1:短路、不重建 bank
    ip, cn, meta, _ = _ds(synthetic_yolo_dataset)
    import anomaly_score
    called = {"n": 0}
    orig = anomaly_score.build_memory_bank
    monkeypatch.setattr(anomaly_score, "build_memory_bank",
                        lambda *a, **k: (called.__setitem__("n", called["n"] + 1) or orig(*a, **k)))
    ext = MemoryBank(np.repeat(A[None, :], 200, axis=0))
    r = run_pipeline(ip, cn, mode="one_stage", score_mode="patch",
                     external_bank=ext, extractor=_patch_ext, embed_fn=_emb)
    assert called["n"] == 0, "掛載 external_bank 時不應重建 memory bank"
    assert len(r["records"]) == len(meta)
    assert r["bank"] is ext                                  # 回傳掛載的 bank


def test_external_bank_scores_defects_top(synthetic_yolo_dataset):  # AC2:用外部 bank 評分,缺陷排前
    ip, cn, meta, defect = _ds(synthetic_yolo_dataset)
    ext = MemoryBank(np.repeat(A[None, :], 500, axis=0))     # 外部「正常」bank = A patch
    r = run_pipeline(ip, cn, mode="one_stage", score_mode="patch",
                     external_bank=ext, extractor=_patch_ext, embed_fn=_emb)
    assert set(r["ranking"][:len(defect)]) == defect          # 含 B patch 的缺陷對 A bank 距離大→前段


def test_external_ref_object_mode(synthetic_yolo_dataset):  # AC3:object 模式對外部 ref 算離群
    ip, cn, meta, defect = _ds(synthetic_yolo_dataset)
    ref = np.repeat(A[None, :], 50, axis=0)                   # 舊 good obj_emb = A 方向
    r = run_pipeline(ip, cn, mode="one_stage", score_mode="object",
                     external_ref=ref, extractor=_patch_ext, embed_fn=_emb)
    assert set(r["ranking"][:len(defect)]) == defect          # 缺陷 obj_emb=B 對 A ref 離群→前段


def test_external_patch_uses_given_model_res(synthetic_yolo_dataset, monkeypatch):  # AC4:cache 用傳入 model/res
    ip, cn, *_ = _ds(synthetic_yolo_dataset)
    seen = {}
    import patch_features
    orig = patch_features.embed_objects_patch

    def spy(meta, model, *, target_res, **k):
        seen["model"], seen["res"] = model, target_res
        return orig(meta, model, target_res=target_res, **k)

    monkeypatch.setattr(patch_features, "embed_objects_patch", spy)
    ext = MemoryBank(np.repeat(A[None, :], 200, axis=0))
    run_pipeline(ip, cn, mode="one_stage", score_mode="patch", external_bank=ext,
                 model="dinov2_vitb14", target_res=336, extractor=_patch_ext, embed_fn=_emb)
    assert seen["model"] == "dinov2_vitb14" and seen["res"] == 336  # 用 bank 鎖死值,非預設


def test_signature_backward_compatible(synthetic_yolo_dataset):  # AC5:不傳 external → 既有行為
    ip, cn, *_ = _ds(synthetic_yolo_dataset)
    r = run_pipeline(ip, cn, mode="one_stage", score_mode="patch",
                     extractor=_patch_ext, embed_fn=_emb)
    assert "records" in r and r["bank"] is not None           # 走既有建 bank 路徑
