"""驗收:anomaly_tool 核心 run_pipeline(設計 06_anomaly_tool.md,單元部分)。
先紅:scripts/anomaly_tool.py 未實作。用注入 fake → 不需真模型。
約定:records 依 discover_yolo_objects 順序;ranking 為 records 索引,分數由高到低。"""
import numpy as np
import pytest
from pathlib import Path
from sklearn.metrics import roc_auc_score

from anomaly_tool import run_pipeline
from interaction import discover_yolo_objects
from object_eval import list_images, classes_for

REQUIRED = {"path", "image_path", "bbox", "obj_index", "label", "split", "score", "verdict"}


def _ds(ds):
    root = ds["root"]
    ip, cn = list_images(root), classes_for(root)
    meta = discover_yolo_objects(ip, cn)
    defect_idx = {i for i, m in enumerate(meta)
                  if Path(str(m["image_path"])).stem.startswith("defect")}
    normal_idx = [i for i in range(len(meta)) if i not in defect_idx]
    return ip, cn, meta, defect_idx, normal_idx


def test_schema_and_original_path(synthetic_yolo_dataset, color_patch_extractor, color_object_embed):  # AC1
    ip, cn, meta, _, _ = _ds(synthetic_yolo_dataset)
    r = run_pipeline(ip, cn, mode="one_stage", score_mode="patch",
                     extractor=color_patch_extractor(), embed_fn=color_object_embed)
    assert len(r["records"]) == len(meta)
    for rec in r["records"]:
        assert REQUIRED <= set(rec)
        assert rec["path"] == rec["image_path"]      # 匯出原圖 [[object-export-crop-bug]]


def test_subtle_defects_top_and_auroc(synthetic_yolo_dataset, color_patch_extractor, color_object_embed):  # AC2 + AC3
    ip, cn, meta, defect_idx, normal_idx = _ds(synthetic_yolo_dataset)
    confirmed = {i: "good" for i in normal_idx[:8]}  # few-shot 乾淨正常集
    r = run_pipeline(ip, cn, mode="two_stage", score_mode="patch",
                     confirmed=confirmed, extractor=color_patch_extractor(),
                     embed_fn=color_object_embed)
    assert set(r["ranking"][:len(defect_idx)]) == defect_idx           # AC2 細微瑕疵全在前段
    scores = np.array([rec["score"] for rec in r["records"]])
    y = np.array([1 if i in defect_idx else 0 for i in range(len(meta))])
    assert roc_auc_score(y, scores) >= 0.85                            # AC3 餵 G1


def test_bank_pollution_drops_score(synthetic_yolo_dataset, color_patch_extractor, color_object_embed):  # AC4(反向驗證設計修正)
    ip, cn, meta, defect_idx, normal_idx = _ds(synthetic_yolo_dataset)
    clean = {i: "good" for i in normal_idx[:8]}
    d = sorted(defect_idx)[0]
    s_clean = run_pipeline(ip, cn, mode="two_stage", score_mode="patch", confirmed=clean,
                           extractor=color_patch_extractor(),
                           embed_fn=color_object_embed)["records"][d]["score"]
    polluted = dict(clean); polluted[d] = "good"      # 把一個瑕疵誤標進乾淨集 → 污染 bank
    s_pol = run_pipeline(ip, cn, mode="two_stage", score_mode="patch", confirmed=polluted,
                         extractor=color_patch_extractor(),
                         embed_fn=color_object_embed)["records"][d]["score"]
    assert s_pol < 0.5 * s_clean


def test_object_mode_and_one_stage_gross(synthetic_yolo_dataset):  # AC5
    ip, cn, meta, defect_idx, _ = _ds(synthetic_yolo_dataset)
    A = np.zeros(384, np.float32); A[0] = 1.0
    B = np.zeros(384, np.float32); B[1] = 1.0

    def gross_ext(crop):  # 任一格紅 → 整個物件全 B(物件級可分,明顯缺陷)
        arr = np.asarray(crop.convert("RGB")).astype(float)
        red = float((arr[..., 0] > 1.2 * (arr[..., 1] + 1)).mean()) > 0.01
        w, h = crop.size; gh, gw = max(1, h // 14), max(1, w // 14)
        return np.repeat((B if red else A)[None, :], gh * gw, axis=0).astype(np.float32), (gh, gw)

    def gross_emb(crop):
        t, _ = gross_ext(crop); v = t.mean(0)
        return (v / (np.linalg.norm(v) or 1.0)).astype(np.float32)

    r_obj = run_pipeline(ip, cn, mode="one_stage", score_mode="object",
                         extractor=gross_ext, embed_fn=gross_emb)
    assert all("verdict" in rec for rec in r_obj["records"])           # object 模式 schema
    r = run_pipeline(ip, cn, mode="one_stage", score_mode="patch",
                     extractor=gross_ext, embed_fn=gross_emb)
    assert set(r["ranking"][:len(defect_idx)]) == defect_idx           # 明顯缺陷 1-stage 仍排前段


def test_no_dataset_pollution(synthetic_yolo_dataset, color_patch_extractor, color_object_embed, tmp_path):  # AC6
    ip, cn, *_ = _ds(synthetic_yolo_dataset)
    root = synthetic_yolo_dataset["root"]
    before = set(root.rglob("*"))
    run_pipeline(ip, cn, mode="one_stage", score_mode="patch",
                 extractor=color_patch_extractor(), embed_fn=color_object_embed,
                 cache_dir=tmp_path / "cache")
    assert set(root.rglob("*")) == before
