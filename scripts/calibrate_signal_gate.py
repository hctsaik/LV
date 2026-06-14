"""Calibrate the 桶① signal-strength gate from a labelled defect dataset.

The gate's default SNR thresholds (``signal_strength.classify_signal``) are
textbook starting points. This script bootstraps DATA-DRIVEN thresholds from
a YOLO-labelled set: it measures every labelled box's ROI-vs-local-background
SNR, then asks ``calibrate_thresholds`` to place the 確無 line below the real
defect distribution and the 明顯 line at the Rose criterion (capped by the
data). Point it at your real escapes/defects to replace the illustrative
numbers a COCO demo produces.

Multiple image types (optics / material) have very different signal floors, so
``--type-from`` calibrates PER TYPE as well as globally; the app then applies
each type's own threshold and falls back to the global one for unseen types.

Usage:
    python scripts/calibrate_signal_gate.py --dataset demo/coco8/train
    python scripts/calibrate_signal_gate.py --dataset A --type-from class --write
    python scripts/calibrate_signal_gate.py --images A/images --labels A/labels
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from interaction import bbox_to_pixels, parse_yolo_boxes, yolo_label_path_for
from signal_strength import (
    calibrate_thresholds,
    classify_signal,
    config_path,
    roi_background_metrics,
    save_calibration,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GLOBAL_KEY = "_global"


def _iter_images(images_dir: Path, labels_dir: Path | None):
    for p in sorted(images_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        if labels_dir is not None:
            lbl = labels_dir / p.relative_to(images_dir).with_suffix(".txt")
        else:
            lbl = yolo_label_path_for(p)
        yield p, lbl


def _load_class_names(start: Path) -> list[str] | None:
    for d in [start, *start.parents][:6]:
        f = d / "classes.txt"
        if f.exists():
            return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines()
                    if ln.strip()]
    return None


def collect_snr(images_dir: Path, labels_dir: Path | None, type_from: str):
    """Measure SNR + frac_signal for every labelled box, grouped by image type.
    Returns (groups{type: {snr:[], frac:[]}}, n_images, n_boxes)."""
    class_names = _load_class_names(images_dir) if type_from == "class" else None
    groups: dict[str, dict[str, list[float]]] = {}
    n_imgs = n_boxes = 0
    for img_path, lbl_path in _iter_images(images_dir, labels_dir):
        boxes = parse_yolo_boxes(lbl_path)
        if not boxes:
            continue
        n_imgs += 1
        try:
            with Image.open(img_path) as im:
                g = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
        except (OSError, ValueError):
            continue
        h, w = g.shape
        for cid, cx, cy, bw, bh in boxes:
            m = roi_background_metrics(g, bbox_to_pixels(cx, cy, bw, bh, w, h))
            if not m:
                continue
            if type_from == "parent":
                t = img_path.parent.name
            elif type_from == "class":
                t = (class_names[cid] if class_names and 0 <= cid < len(class_names)
                     else f"class_{cid}")
            else:
                t = GLOBAL_KEY
            grp = groups.setdefault(t, {"snr": [], "frac": []})
            grp["snr"].append(m["snr"])
            grp["frac"].append(m["frac_signal"])
            n_boxes += 1
    return groups, n_imgs, n_boxes


def _tally(snrs, fracs, snr_none, snr_obvious) -> dict:
    return dict(Counter(
        classify_signal({"snr": float(s), "frac_signal": float(f)},
                        snr_none=snr_none, snr_obvious=snr_obvious)
        for s, f in zip(snrs, fracs)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, help="folder holding images/ and labels/")
    ap.add_argument("--images", type=Path, help="image dir (if not using --dataset)")
    ap.add_argument("--labels", type=Path, help="YOLO label dir")
    ap.add_argument("--type-from", choices=["parent", "class", "none"],
                    default="parent",
                    help="how to group image types: image's parent dir (default), "
                         "the box's class, or none (global only)")
    ap.add_argument("--write", action="store_true",
                    help="persist the suggested thresholds so the app applies them")
    ap.add_argument("--config", type=Path, default=None,
                    help="config path to write (default: <repo>/signal_gate.json)")
    args = ap.parse_args()

    if args.dataset:
        images_dir, labels_dir = args.dataset / "images", args.dataset / "labels"
    else:
        images_dir, labels_dir = args.images, args.labels
    if not images_dir or not Path(images_dir).exists():
        ap.error(f"images dir not found: {images_dir}")

    groups, n_imgs, n_boxes = collect_snr(Path(images_dir), labels_dir, args.type_from)
    print(f"掃描 {n_imgs} 張有標註影像、{n_boxes} 個可量測框"
          f"（分組依據：{args.type_from}，共 {len(groups)} 組）。")
    if n_boxes == 0:
        print("沒有可量測的框；無法校準。")
        return

    all_snr = np.concatenate([np.asarray(g["snr"]) for g in groups.values()])
    all_frac = np.concatenate([np.asarray(g["frac"]) for g in groups.values()])
    print("\n全域 SNR 分佈（已確認的標註框 = 『可偵測』的經驗參考）：")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<2} = {np.percentile(all_snr, q):7.2f}")
    print(f"\n預設門檻 (snr_none=1.5, snr_obvious=4.0)："
          f"{_tally(all_snr, all_frac, 1.5, 4.0)}")

    g_cal = calibrate_thresholds(all_snr)
    if not g_cal["calibrated"]:
        print(f"\n全域框數 {g_cal['n']} < 10，樣本太少不足以校準，沿用預設。")
        if args.write:
            print("   未寫入（樣本太少）。")
        return
    print(f"\n建議全域門檻 (snr_none={g_cal['snr_none']}, snr_obvious={g_cal['snr_obvious']})："
          f"{_tally(all_snr, all_frac, g_cal['snr_none'], g_cal['snr_obvious'])}")

    # per-type calibration (only types with enough boxes; others fall back)
    per_type: dict[str, dict] = {}
    print("\n逐類型門檻（框數 ≥10 才校準，其餘執行時退回全域）：")
    for t in sorted(groups, key=lambda k: -len(groups[k]["snr"])):
        snr = np.asarray(groups[t]["snr"])
        c = calibrate_thresholds(snr)
        if c["calibrated"]:
            per_type[t] = {"snr_none": c["snr_none"],
                           "snr_obvious": c["snr_obvious"], "n": c["n"]}
            print(f"  {t:<18} n={c['n']:<5} none={c['snr_none']:<5} "
                  f"obvious={c['snr_obvious']:<5} (p50={c['p50']})")
        else:
            print(f"  {t:<18} n={len(snr):<5} (樣本不足 → 退回全域)")

    if args.write:
        ds = str(args.dataset) if args.dataset else str(images_dir)
        cfg = {**g_cal, "dataset": ds, "n_images": n_imgs,
               "type_from": args.type_from, "per_type": per_type}
        written = save_calibration(cfg, args.config)
        print(f"\n✅ 已寫入校準設定：{written}")
        print(f"   全域 + {len(per_type)} 個逐類型門檻；app 啟動後自動套用"
              "（體檢卡顯示『校準』與該類型門檻）。")

    print("\n注意：COCO 等『大物件』資料 SNR 偏低、僅供示範；請指向真實 escape/瑕疵框再跑。")


if __name__ == "__main__":
    main()
