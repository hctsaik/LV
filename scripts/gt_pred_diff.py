"""GT × 模型預測比對挖錯(FN / FP / 類別混淆 / 低信心)+ 預測框預標 seed。

設計:3_Architect_Design/10_gt_pred_diff.md

讓「模型自己」參與主動學習迴圈:同時載入 GT 與模型預測(YOLO 6 欄 txt,或 13 號模組
``heatmap_to_boxes`` 由熱力圖產生的同格式檔),產出四種可送標佇列 —— 漏抓 / 誤抓 /
類別混淆 / 低信心 —— 讓人優先看模型「錯得最兇」的圖,而不是隨機翻。

純函式、決定論、不 import streamlit。
"""
from __future__ import annotations

from pathlib import Path

from interaction import parse_yolo_boxes, parse_yolo_boxes_conf, yolo_label_path_for
from label_formats import annotation_boxes_for_image

# score 權重(排序用,釘死於設計):漏抓與類別混淆最痛(2),誤抓次之(1),低信心只是提示(0.5)
_W_FN, _W_FP, _W_CONFUSED, _W_LOW_CONF = 2.0, 1.0, 2.0, 0.5


def iou_cxcywh(a: tuple, b: tuple) -> float:
    """兩個 ``(cx, cy, w, h)`` 正規化框的 IoU。無交集或任一框退化(w/h<=0)回 ``0.0``。"""
    acx, acy, aw, ah = a[:4]
    bcx, bcy, bw, bh = b[:4]
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax1, ax2 = acx - aw / 2, acx + aw / 2
    ay1, ay2 = acy - ah / 2, acy + ah / 2
    bx1, bx2 = bcx - bw / 2, bcx + bw / 2
    by1, by2 = bcy - bh / 2, bcy + bh / 2

    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    # 角點是 cx±w/2 算出來的,浮點下 (cx+w/2)−(cx−w/2) 不精確等於 w:兩個相同的框
    # 會得到 0.9999999999999997 而非 1.0。量化到 1e-12(遠高於 float 雜訊 ~1e-16、
    # 遠低於任何有意義的 IoU 差異)讓「重合=1.0」這個數學事實在浮點下也成立。
    return min(max(round(inter / union, 12), 0.0), 1.0)


def match_boxes(gt: list[tuple], pred: list[tuple], iou_thr: float = 0.5) -> dict:
    """貪婪匹配 GT 與預測 → ``{"tp", "fn", "fp", "confused"}``。

    ``gt``   = ``[(cid, cx, cy, w, h), …]``(``parse_yolo_boxes`` 形)
    ``pred`` = ``[(cid, cx, cy, w, h, conf|None), …]``(``parse_yolo_boxes_conf`` 形)

    預測依 conf 由高到低處理(``None`` 視為 1.0;同 conf 依原順序),每個預測取 IoU 最高且
    ≥ ``iou_thr`` 的**未被占用** GT。位置對上但類別不同 → 進 ``confused``(不是漏抓也不是誤抓,
    是「找到了但叫錯名字」,該送標的理由不同)。決定論:不依賴 set/dict 迭代順序。
    """
    def _conf(p):
        return p[5] if len(p) > 5 and p[5] is not None else 1.0

    # stable sort → 同 conf 時保留原始索引序
    order = sorted(range(len(pred)), key=lambda i: -_conf(pred[i]))

    claimed: dict[int, None] = {}          # gi → 已被占用(用 dict 保序,不用 set)
    tp: list[tuple[int, int, float]] = []
    confused: list[tuple[int, int, float]] = []
    matched_pred: dict[int, None] = {}

    for pi in order:
        p = pred[pi]
        best_gi, best_iou = -1, 0.0
        for gi, g in enumerate(gt):
            if gi in claimed:
                continue
            v = iou_cxcywh(g[1:5], p[1:5])
            if v >= iou_thr and v > best_iou:
                best_gi, best_iou = gi, v
        if best_gi < 0:
            continue
        claimed[best_gi] = None
        matched_pred[pi] = None
        (tp if gt[best_gi][0] == p[0] else confused).append((best_gi, pi, best_iou))

    tp.sort(key=lambda t: t[1])            # 依 pi 升冪
    confused.sort(key=lambda t: t[1])
    return {
        "tp": tp,
        "fn": [gi for gi in range(len(gt)) if gi not in claimed],
        "fp": [pi for pi in range(len(pred)) if pi not in matched_pred],
        "confused": confused,
    }


def image_error_report(gt, pred, *, iou_thr: float = 0.5, low_conf: float = 0.5) -> dict:
    """``match_boxes`` + 低信心佇列 + 排序用 ``score``(權重釘死於設計)。

    ``low_conf`` 含**已匹配**的預測 —— 「猜對了但沒把握」本身就值得補標。
    """
    r = match_boxes(gt, pred, iou_thr=iou_thr)
    r["low_conf"] = [
        pi for pi, p in enumerate(pred)
        if len(p) > 5 and p[5] is not None and p[5] < low_conf
    ]
    r["score"] = (_W_FN * len(r["fn"]) + _W_FP * len(r["fp"])
                  + _W_CONFUSED * len(r["confused"]) + _W_LOW_CONF * len(r["low_conf"]))
    return r


def pred_label_path_for(image_path: Path, pred_root: Path) -> Path:
    """預測資料夾支援兩種佈局:``pred_root/labels/<stem>.txt``(YOLO 慣例)或
    ``pred_root/<stem>.txt``(平鋪)。只回路徑,不檢查存在。"""
    pred_root = Path(pred_root)
    stem = Path(image_path).stem
    labels_dir = pred_root / "labels"
    if labels_dir.is_dir():
        return labels_dir / f"{stem}.txt"
    return pred_root / f"{stem}.txt"


def _gt_boxes_for(image_path: Path) -> list[tuple]:
    """GT 框:先走 YOLO txt;txt **不存在**才回退四格式標註(COCO/VOC/LabelMe/NDJSON)。

    空的 txt 是「有標註來源、此圖無框」,不是「沒標註」→ 不觸發回退(與 YOLO 慣例一致)。
    回退來源可能沒有數字類別 id(名稱制格式),此時視為類別 0 —— 單類語義下仍可比對位置。
    """
    gt_path = yolo_label_path_for(image_path)
    if gt_path.exists():
        return parse_yolo_boxes(gt_path)
    rows = annotation_boxes_for_image(Path(image_path))
    if not rows:
        return []
    return [(int(r[0]) if r[0] is not None else 0, r[1], r[2], r[3], r[4]) for r in rows]


def scan_dataset(records: list[dict], pred_root: Path,
                 *, iou_thr: float = 0.5, low_conf: float = 0.5) -> list[dict]:
    """對每筆記錄比對 GT 與預測 → ``[{"path", "report", "n_gt", "n_pred"}, …]``,依 ``score`` 降冪。

    GT 與預測**兩邊皆空**的圖不進列表(無事可看)。
    """
    out: list[dict] = []
    for rec in records:
        path = Path(rec["path"])
        gt = _gt_boxes_for(path)
        pred = parse_yolo_boxes_conf(pred_label_path_for(path, pred_root))
        if not gt and not pred:
            continue
        out.append({
            "path": str(path),
            "report": image_error_report(gt, pred, iou_thr=iou_thr, low_conf=low_conf),
            "n_gt": len(gt),
            "n_pred": len(pred),
        })
    out.sort(key=lambda r: -r["report"]["score"])   # stable → 同分保留輸入序
    return out
