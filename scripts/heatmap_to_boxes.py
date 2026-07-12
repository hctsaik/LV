"""anomaly 熱力圖(patch 分數圖)→ YOLO 6 欄預標框。

設計:3_Architect_Design/13_heatmap_to_boxes.md

回答「很強的 embedding / anomaly DINO 能不能像 YOLO 一樣做預標?」——**能,但要經過
定位橋接**:DINOv2 embedding 本身只會分類/檢索,不會畫框;但 anomaly DINO 的 patch
分數圖帶空間資訊。本模組把 score map → 閾值 → 連通區域 → YOLO 6 欄框(class cx cy w h conf),
輸出成與模型預測完全同形的「預測資料夾」,直通 gt_pred_diff 管線,零特例。

誠實界線:框粒度受 patch 網格限制、單一類別(``defect``)、只適用瑕疵型資料;
一般多類物件偵測請用真的偵測器。

純函式、決定論、不 import streamlit。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

# 8 連通(含對角):對角相接的熱點屬同一個瑕疵區。
_CONNECTIVITY_8 = np.ones((3, 3), dtype=int)

DEFECT_CLASS_NAME = "defect"


def boxes_from_scoremap(
    pmap: np.ndarray,
    *,
    thr: float | None = None,
    quantile: float = 0.98,
    min_area_frac: float = 0.001,
) -> list[tuple[float, float, float, float, float]]:
    """HxW patch 分數圖 → ``[(cx, cy, w, h, conf), …]``,全部 0–1 正規化(相對 pmap 尺寸)。

    ``thr=None`` 時閾值取 ``np.quantile(pmap, quantile)``(``thr`` 優先,明確契約)。
    面積 < ``H*W*min_area_frac`` 的連通區域視為雜點濾除。``conf`` = 該區域 pmap 最大值,
    輸出依 conf 降冪。全零 / 全同值圖(沒有相對突起 → 沒有異常區)回 ``[]``。
    """
    arr = np.asarray(pmap, dtype=float)
    if arr.ndim != 2:  # 不靜默壓維:呼叫端傳錯形狀是 bug,要炸出來
        raise ValueError(f"pmap must be 2-D (H, W), got shape {arr.shape}")

    level = float(np.quantile(arr, quantile)) if thr is None else float(thr)
    # 嚴格大於:常數圖的 quantile 閾值 == 該常數 → mask 全 False → 回 []
    mask = arr > level
    if not mask.any():
        return []

    lab, n = ndimage.label(mask, structure=_CONNECTIVITY_8)
    if n == 0:
        return []

    H, W = arr.shape
    min_area = H * W * min_area_frac
    boxes: list[tuple[float, float, float, float, float]] = []
    # find_objects 回每個 component 的 (row_slice, col_slice) 外接框;索引為 [row, col] = [y, x]
    for comp_id, (rs, cs) in enumerate(ndimage.find_objects(lab), start=1):
        region = lab[rs, cs] == comp_id
        if int(region.sum()) < min_area:
            continue
        conf = float(arr[rs, cs][region].max())
        cx = (cs.start + cs.stop) / 2.0 / W
        cy = (rs.start + rs.stop) / 2.0 / H
        w = (cs.stop - cs.start) / W
        h = (rs.stop - rs.start) / H
        boxes.append((cx, cy, w, h, conf))

    boxes.sort(key=lambda b: -b[4])
    return boxes


def box_to_image_space(
    box: tuple[float, float, float, float, float],
    region: tuple[float, float, float, float],
    iw: int,
    ih: int,
) -> tuple[float, float, float, float, float]:
    """裁切座標系的框 → **原圖**正規化座標。``region`` = 該 pmap 對應的裁切矩形(像素 x0,y0,x1,y1)。

    瑕疵偵測的 pmap 是「單一物件裁切圖」的分數圖,框自然落在裁切座標系;但預標 txt 必須是
    原圖座標(下游 ``gt_pred_diff`` 要拿它跟原圖的 GT 比)。整張影像模式下 region == 全圖,
    此映射退化為 identity。
    """
    cx, cy, w, h, conf = box
    x0, y0, x1, y1 = region
    cw, ch = (x1 - x0), (y1 - y0)
    return ((x0 + cx * cw) / iw, (y0 + cy * ch) / ih,
            w * cw / iw, h * ch / ih, conf)


def boxes_by_image(
    objects: list[dict],
    *,
    thr: float | None = None,
    quantile: float = 0.98,
    min_area_frac: float = 0.001,
) -> dict[str, list[tuple[float, float, float, float, float]]]:
    """多個物件的 pmap → 依**原圖**聚合的框(原圖座標系,conf 降冪)。

    ``objects`` 每筆 = ``{"image_path", "pmap", "image_size": (iw, ih), "region": (x0,y0,x1,y1)|None}``。
    同一張圖的多個物件會把各自的框合併到同一份預標。``region=None`` → 整張影像。

    每個物件的閾值**各自**依其 pmap 分佈算(與熱力圖所見一致);不把全圖背景當 0 分混進分位數,
    免得「物件外=沒分數」被誤當「物件外=正常」。
    """
    out: dict[str, list[tuple[float, float, float, float, float]]] = {}
    for o in objects:
        boxes = boxes_from_scoremap(
            o["pmap"], thr=thr, quantile=quantile, min_area_frac=min_area_frac)
        iw, ih = o["image_size"]
        region = o.get("region") or (0, 0, iw, ih)
        key = str(o["image_path"])
        out.setdefault(key, []).extend(
            box_to_image_space(b, region, iw, ih) for b in boxes)
    for boxes in out.values():
        boxes.sort(key=lambda b: -b[4])
    return out


def write_boxes(
    boxes_by_img: dict[str, list[tuple[float, float, float, float, float]]],
    out_root: Path,
    *,
    class_id: int = 0,
) -> dict:
    """已算好的原圖座標框 → ``out_root/labels/<stem>.txt``(YOLO 6 欄:class cx cy w h conf)。

    無框的圖寫**空檔**(語義=「有預測來源、此圖無框」,對齊 YOLO 慣例;缺檔才是「沒跑過」)。
    冪等:每次呼叫先清 ``labels/*.txt``,故改參數重呼不會殘留舊框。

    呼叫端負責保證 ``out_root`` 落在 ``.lv_cache`` 下 —— 絕不寫使用者的資料集。
    """
    out_root = Path(out_root)
    labels_dir = out_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for stale in labels_dir.glob("*.txt"):
        stale.unlink()

    n_boxes = 0
    for img_path, boxes in boxes_by_img.items():
        lines = [f"{class_id} {cx:.6g} {cy:.6g} {w:.6g} {h:.6g} {conf:.6g}"
                 for cx, cy, w, h, conf in boxes]
        n_boxes += len(boxes)
        (labels_dir / f"{Path(img_path).stem}.txt").write_text(
            "\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    (out_root / "classes.txt").write_text(DEFECT_CLASS_NAME + "\n", encoding="utf-8")
    return {"n_images": len(boxes_by_img), "n_boxes": n_boxes, "out_root": str(out_root)}


def write_pred_labels(
    items: list[tuple[Path, np.ndarray]],
    out_root: Path,
    *,
    class_id: int = 0,
    thr: float | None = None,
    quantile: float = 0.98,
    min_area_frac: float = 0.001,
) -> dict:
    """``[(原圖路徑, 該圖 pmap)]`` → 預測資料夾(一圖一 pmap 的直路;整張影像模式即此形)。

    多物件、需要裁切→原圖座標映射時走 :func:`boxes_by_image` + :func:`write_boxes`。
    """
    return write_boxes(
        {str(p): boxes_from_scoremap(pm, thr=thr, quantile=quantile,
                                     min_area_frac=min_area_frac)
         for p, pm in items},
        out_root, class_id=class_id)
