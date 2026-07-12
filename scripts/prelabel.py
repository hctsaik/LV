"""11 prelabel:凍結分類頭的閘控判定 → 可接受的預標決定 → 安全 YOLO 標註檔。

設計:3_Architect_Design/11_prelabel.md。人仍是最終裁決:模型只填、只建議。
復用 dino_head 的 gated_predict/predict_head 與 anomaly_bank_store 的原子寫;
不重新發明閘控/原子寫。C6:絕不寫使用者來源資料集(assert_safe_prelabel_dir 守門)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_NON_CLASS = ("正常", "Unknown")   # 非「已知類別」判定 → 不 accept、不輸出 YOLO(留白給人)


def prelabel_records(records, obj_emb, scores, head, *, anomaly_threshold,
                     min_conf=0.5, target_classes=None) -> list[dict]:
    """每物件回 {"item", "pred", "conf", "accept"}。
    pred = gated_predict(...)（正常/Unknown/已知類別;min_conf 內含於閘控);
    conf = predict_head 最大類別機率;
    accept = pred 為已知類別 AND (target_classes is None OR pred in target_classes)。
    head=None → ValueError(不靜默降級/亂猜);長度不符 → ValueError;空輸入 → []。"""
    if head is None:
        raise ValueError("預標需要分類頭(head);此模型無 head,不做預標(不亂猜)。")
    emb = np.asarray(obj_emb, dtype=np.float32)
    n = len(records)
    if emb.shape[0] != n or len(scores) != n:
        raise ValueError(f"長度不符:records={n}, obj_emb={emb.shape[0]}, scores={len(scores)}")
    if n == 0:
        return []
    from dino_head import gated_predict, predict_head
    preds = gated_predict(head, emb, scores, anomaly_threshold=float(anomaly_threshold),
                          min_conf=float(min_conf))
    _, conf, _ = predict_head(head, emb)
    tset = None if target_classes is None else set(target_classes)
    out = []
    for i in range(n):
        pred = preds[i]
        is_known = pred not in _NON_CLASS
        accept = bool(is_known and (tset is None or pred in tset))
        out.append({"item": i, "pred": pred, "conf": float(conf[i]), "accept": accept})
    return out


def to_yolo_lines(records, decisions, *, class_names) -> dict:
    """依 image_path 分組回 {image_path: ["<cls_id> cx cy w h", …]}。
    只納入 accept 且 pred 為已知類別的物件;cls_id = class_names.index(pred)
    (標籤空間以 head 類別為準;pred 不在 class_names → ValueError)。座標 6 位小數。"""
    cls_index = {c: i for i, c in enumerate(class_names)}
    out: dict = {}
    for d in decisions:
        if not d.get("accept"):
            continue
        pred = d["pred"]
        if pred in _NON_CLASS:                      # 雙保險:正常/Unknown 絕不輸出
            continue
        if pred not in cls_index:
            raise ValueError(f"預測類別 '{pred}' 不在輸出類別空間 {list(class_names)}")
        rec = records[d["item"]]
        cx, cy, w, h = rec["bbox"]
        line = f"{cls_index[pred]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        out.setdefault(rec["image_path"], []).append(line)
    return out


def _under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_safe_prelabel_dir(out_dir, source_dirs=(), allow_images=False) -> None:
    """C6:out_dir 含 images/ → raise(像資料集);與任一 source_dir 有等於/祖先/後代關係 → raise。
    允許已含 labels/ 的目錄(支援重複匯出)—— 這是與 assert_safe_bank_dir 的關鍵差異。
    allow_images=True(M14c:明確要建 standalone 資料集)→ 跳過 images/ 啟發式,**來源關係檢查仍執行**。"""
    p = Path(out_dir).resolve()
    if not allow_images and (p / "images").exists():
        raise ValueError(f"輸出目錄含 images/,看起來是資料集,拒寫(C6):{p}")
    for f in source_dirs:
        fp = Path(f).resolve()
        if p == fp or _under(p, fp) or _under(fp, p):
            raise ValueError(f"輸出目錄不得等於/位於/包含來源資料集(C6):{p} vs {fp}")


def export_prelabels(lines_by_image, out_dir, *, class_names=None, source_dirs=(), allow_images=False) -> dict:
    """把 to_yolo_lines 結果原子寫到 out_dir/labels/<stem>.txt(+可選 classes.txt)。
    回 {"written": 檔數, "objects": 總行數, "out_dir": str}。C6 安全檢查在寫任何檔之前。
    allow_images=True:允許 out_dir 已含 images/(standalone 資料集匯出;來源關係檢查仍守)。"""
    from anomaly_bank_store import _atomic_text
    assert_safe_prelabel_dir(out_dir, source_dirs=source_dirs, allow_images=allow_images)
    out = Path(out_dir)
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)   # 同時建 out 與 out/labels
    written = 0
    objects = 0
    for image_path, lines in lines_by_image.items():
        stem = Path(image_path).stem
        text = "\n".join(lines) + ("\n" if lines else "")
        _atomic_text(labels_dir / f"{stem}.txt", text)
        written += 1
        objects += len(lines)
    if class_names is not None:
        _atomic_text(out / "classes.txt", "\n".join(class_names) + "\n")
    return {"written": written, "objects": objects, "out_dir": str(out)}
