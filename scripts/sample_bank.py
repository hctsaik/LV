"""13 sample_bank:few-shot 樣本集(帶 provenance + 綁定凍結模型),存 .lv_cache 供海掃比對。

設計:3_Architect_Design/13_sample_bank.md(M13)。復用 anomaly_tool._object_embeddings、
anomaly_bank_store._atomic_*。與 anomaly bank 不同:存**物件級** embedding + 從第一天帶 provenance。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_SCHEMA = 1


def build_sample_bank(sample_dirs, *, model, target_res, class_names=None,
                      object_source="yolo", embed_fn=None, extractor=None) -> dict:
    """樣本資料夾 → discover objects → _object_embeddings → bank dict。無物件 → ValueError。
    embed_fn 可注入(測試)。回 {vectors,labels,provenance,model,target_res}。"""
    from anomaly_tool import _object_embeddings
    from interaction import discover_whole_images, discover_yolo_objects
    from object_eval import classes_for, list_images
    meta = []
    cn = class_names
    for d in sample_dirs:
        d = Path(d)
        imgs = list_images(d)
        cn = cn or classes_for(d)
        meta.extend(discover_whole_images(imgs) if object_source == "whole_image"
                    else discover_yolo_objects(imgs, cn))
    if not meta:
        raise ValueError("樣本資料夾找不到物件(YOLO 需 images/ + labels/;或改整張影像來源)")
    vectors = np.asarray(_object_embeddings(meta, model, embed_fn), dtype=np.float32)
    labels = np.array([str(m.get("label") or "—") for m in meta])
    provenance = [{"image_path": str(m["image_path"]),
                   "bbox": [float(x) for x in m["bbox"]], "label": str(m.get("label") or "—")}
                  for m in meta]
    return {"vectors": vectors, "labels": labels, "provenance": provenance,
            "model": str(model), "target_res": int(target_res)}


def save_sample_bank(bank_dir, bank) -> None:
    """emb.npz(vectors+labels)+ meta.json(model/target_res/provenance)。原子寫。"""
    from anomaly_bank_store import _atomic_npz, _atomic_text
    d = Path(bank_dir)
    d.mkdir(parents=True, exist_ok=True)
    _atomic_npz(d / "emb.npz", vectors=np.asarray(bank["vectors"], dtype=np.float32),
                labels=np.asarray(bank["labels"]))
    meta = {"schema_version": _SCHEMA, "model": str(bank["model"]),
            "target_res": int(bank["target_res"]), "provenance": list(bank.get("provenance") or [])}
    _atomic_text(d / "meta.json", json.dumps(meta, ensure_ascii=False))


def load_sample_bank(bank_dir) -> dict:
    """回 {vectors,labels,provenance,model,target_res}。缺 meta.json → FileNotFoundError。"""
    d = Path(bank_dir)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"樣本集不完整(缺 meta.json):{d}")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    npz = np.load(d / "emb.npz", allow_pickle=False)
    return {"vectors": npz["vectors"].astype(np.float32), "labels": npz["labels"],
            "provenance": meta.get("provenance") or [],
            "model": meta.get("model"), "target_res": int(meta.get("target_res"))}


def append_sample(bank_dir, *, vectors, labels, provenance) -> dict:
    """把新物件接到既有樣本集尾端(N→N+k),原子覆寫,回更新後 bank。維度/長度不符 → ValueError。"""
    bank = load_sample_bank(bank_dir)
    nv = np.asarray(vectors, dtype=np.float32)
    if nv.ndim != 2 or nv.shape[1] != bank["vectors"].shape[1]:
        raise ValueError(f"append 向量維度 {nv.shape} 與樣本集 D={bank['vectors'].shape[1]} 不符")
    if len(labels) != nv.shape[0] or len(provenance) != nv.shape[0]:
        raise ValueError("append 的 labels/provenance 長度需等於向量數")
    merged = {"vectors": np.concatenate([bank["vectors"], nv], axis=0),
              "labels": np.array(list(bank["labels"]) + [str(l) for l in labels]),
              "provenance": list(bank["provenance"]) + list(provenance),
              "model": bank["model"], "target_res": bank["target_res"]}
    save_sample_bank(bank_dir, merged)
    return merged


def training_head_ready(labels, *, min_per_class=8, min_classes=2) -> dict:
    """M13 Task7 導流門檻:數每類樣本數,判斷是否足以訓一個分種類頭(≥min_classes 類、各 ≥min_per_class)。
    純計數、無 I/O、不改樣本集。回 {ready, per_class(依類名排序), ready_classes(排序)}。"""
    from collections import Counter
    counts = Counter(str(l) for l in labels)
    per_class = {c: int(counts[c]) for c in sorted(counts)}
    ready_classes = [c for c in sorted(per_class) if per_class[c] >= int(min_per_class)]
    return {"ready": len(ready_classes) >= int(min_classes),
            "per_class": per_class, "ready_classes": ready_classes}


def assert_model_compatible(bank, *, model, target_res) -> None:
    """樣本集綁的模型/解析度與現用不符 → ValueError(不可跨 embedding 空間比對)。"""
    if str(bank.get("model")) != str(model):
        raise ValueError(f"樣本集綁定模型 {bank.get('model')} 與現用 {model} 不符;請用同一模型或重建樣本集")
    if int(bank.get("target_res")) != int(target_res):
        raise ValueError(f"樣本集解析度 {bank.get('target_res')} 與現用 {target_res} 不符")
