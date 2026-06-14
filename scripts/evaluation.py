"""Detection evaluation gate — per-type recall on a CONSENSUS subset.

The defect-escape problem's central pain is "重訓有時有效有時沒用、無法向自己
交代有沒有變好" (defect_problem_redefinition §5.3). The fix is a stable target:
evaluate model predictions against ground truth — but only on the subset where
annotators agree (組考卷 consensus), because recall measured against an unstable
ruler is "model vs one person's mood, on one day".

This module is the framework-free core: IoU-match predicted boxes to GT boxes
(COCO-style greedy, highest-confidence prediction first), count TP/FP/FN per
class, and surface the false negatives (the escapes) for triage. It takes plain
dicts (no streamlit, no disk) so it is fully unit-testable; the app wires GT
(parse_yolo_boxes), predictions (load_predictions_csv) and the optional
consensus filter around it.

Boxes are normalized YOLO dicts ``{cls, cx, cy, w, h[, score]}`` (cx,cy,w,h in
[0,1]). Recall denominator = consensus GT only (signable); gray-band GT is
reported separately, never as a model failure.
"""
from __future__ import annotations

from collections.abc import Sequence


def iou_xywh(a: dict, b: dict) -> float:
    """IoU of two normalized YOLO boxes (center cx,cy + size w,h)."""
    ax0, ay0, ax1, ay1 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2, \
        a["cx"] + a["w"] / 2, a["cy"] + a["h"] / 2
    bx0, by0, bx1, by1 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2, \
        b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) \
        + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def match_image(
    gt: Sequence[dict],
    preds: Sequence[dict],
    *,
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.0,
    class_aware: bool = True,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy match one image's predictions to its GT (COCO-style: each
    prediction, in descending confidence, claims the best unclaimed GT of the
    same class with IoU ≥ ``iou_thresh``).

    Returns (tp, fn, fp) where tp = [(pred_idx, gt_idx, iou)], fn = [gt_idx of
    unmatched GT], fp = [pred_idx of unmatched predictions above conf].
    """
    order = sorted(
        (j for j, p in enumerate(preds) if p.get("score", 1.0) >= conf_thresh),
        key=lambda j: -preds[j].get("score", 1.0))
    matched_gt: set[int] = set()
    tp: list[tuple[int, int, float]] = []
    fp: list[int] = []
    for j in order:
        p = preds[j]
        best, best_iou = None, iou_thresh
        for gi, g in enumerate(gt):
            if gi in matched_gt:
                continue
            if class_aware and str(p["cls"]) != str(g["cls"]):
                continue
            v = iou_xywh(p, g)
            if v >= best_iou:
                best, best_iou = gi, v
        if best is not None:
            matched_gt.add(best)
            tp.append((j, best, best_iou))
        else:
            fp.append(j)
    fn = [gi for gi in range(len(gt)) if gi not in matched_gt]
    return tp, fn, fp


def _pr(tp: int, fp: int, fn: int) -> tuple[float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return round(prec, 4), round(rec, 4)


def evaluate_detections(
    gt_by_image: dict[str, list[dict]],
    pred_by_image: dict[str, list[dict]],
    *,
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.0,
    class_aware: bool = True,
    consensus_by_image: dict[str, list[bool]] | None = None,
) -> dict:
    """Per-class TP/FP/FN + recall on the CONSENSUS subset, with the escape
    (false-negative) list and a confusion tally.

    ``consensus_by_image`` (optional) marks, per GT box, whether annotators
    agreed on it. When given, only consensus GT counts toward recall; gray-band
    GT (no trustworthy answer) is tallied separately under ``gray`` and never
    counts as a model miss. When ``None`` every GT box is treated as consensus.

    Returns ``{per_class, overall, false_negatives, confusion, gray,
    n_images}``. ``false_negatives`` are the consensus escapes (filename, cls,
    box) for the gallery; ``confusion`` is a list of ``(gt_cls|None,
    pred_cls|None)`` pairs the UI rolls into a matrix.
    """
    per: dict[str, dict] = {}
    fn_list: list[dict] = []
    confusion: list[tuple] = []
    gray = {"tp": 0, "fn": 0}
    images = set(gt_by_image) | set(pred_by_image)

    def cell(cls: str) -> dict:
        return per.setdefault(str(cls), {"tp": 0, "fp": 0, "fn": 0})

    for fname in images:
        gt = gt_by_image.get(fname, [])
        preds = pred_by_image.get(fname, [])
        cons = (consensus_by_image or {}).get(fname)
        tp, fn, fp = match_image(gt, preds, iou_thresh=iou_thresh,
                                 conf_thresh=conf_thresh, class_aware=class_aware)
        matched = {gi: pj for pj, gi, _ in tp}
        for gi, g in enumerate(gt):
            is_cons = cons[gi] if cons is not None and gi < len(cons) else True
            if gi in matched:
                if is_cons:
                    cell(g["cls"])["tp"] += 1
                    confusion.append((str(g["cls"]), str(preds[matched[gi]]["cls"])))
                else:
                    gray["tp"] += 1
            else:
                if is_cons:
                    cell(g["cls"])["fn"] += 1
                    confusion.append((str(g["cls"]), None))
                    fn_list.append({"filename": fname, "cls": str(g["cls"]),
                                    "box": {k: g[k] for k in ("cx", "cy", "w", "h")}})
                else:
                    gray["fn"] += 1
        for pj in fp:
            cell(preds[pj]["cls"])["fp"] += 1
            confusion.append((None, str(preds[pj]["cls"])))

    per_class = {}
    o_tp = o_fp = o_fn = 0
    for cls, c in sorted(per.items()):
        prec, rec = _pr(c["tp"], c["fp"], c["fn"])
        per_class[cls] = {**c, "n_gt": c["tp"] + c["fn"],
                          "precision": prec, "recall": rec}
        o_tp += c["tp"]; o_fp += c["fp"]; o_fn += c["fn"]
    o_prec, o_rec = _pr(o_tp, o_fp, o_fn)
    return {
        "per_class": per_class,
        "overall": {"tp": o_tp, "fp": o_fp, "fn": o_fn, "n_gt": o_tp + o_fn,
                    "precision": o_prec, "recall": o_rec},
        "false_negatives": fn_list,
        "confusion": confusion,
        "gray": {**gray, "total": gray["tp"] + gray["fn"]},
        "n_images": len(images),
    }
