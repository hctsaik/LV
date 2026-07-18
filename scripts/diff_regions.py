"""diff_regions:差異分數圖 → Top-K 連通區域 + 代表樣本索引 + 白話摘要。

設計:3_Architect_Design/21_diff_regions.md(M19)。純陣列(numpy + scipy.ndimage)、
無 I/O。區域 = z_map 過門檻後 8 連通合併(沿 heatmap_to_boxes 慣例),依峰值降冪取 Top-K。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def extract_regions(z_map, stability_map, *, z_thresh: float,
                    top_k: int = 5, min_patches: int = 1) -> list[dict]:
    z = np.asarray(z_map, dtype=float)
    st = np.asarray(stability_map, dtype=float)
    if z.ndim != 2:
        raise ValueError(f"z_map 應為 2 維,實得 {z.shape}")
    if st.shape != z.shape:
        raise ValueError(f"stability_map 形狀 {st.shape} != z_map {z.shape}")
    _gh, gw = z.shape

    lab, n = ndimage.label(z >= z_thresh, structure=np.ones((3, 3)))
    regs: list[dict] = []
    for i in range(1, n + 1):
        rows, cols = np.where(lab == i)
        if len(rows) < min_patches:
            continue
        flat = rows * gw + cols
        zi = z[rows, cols]
        k = int(np.argmax(zi))
        regs.append({
            "patch_idx": [int(x) for x in flat],
            "bbox_grid": (int(rows.min()), int(cols.min()),
                          int(rows.max()) + 1, int(cols.max()) + 1),
            "peak_idx": int(flat[k]),
            "peak_z": float(zi[k]),
            "mean_z": float(zi.mean()),
            "stability": float(st[rows[k], cols[k]]),   # 峰值 patch 最能代表此區
        })
    regs.sort(key=lambda r: -r["peak_z"])
    regs = regs[:max(int(top_k), 0)]
    for rank, r in enumerate(regs, 1):
        r["rank"] = rank
    return regs


def pick_representatives(region: dict, dist_good, dist_bad, *,
                         n_each: int = 3) -> dict:
    idx = list(region.get("patch_idx") or [])
    if not idx:
        raise ValueError("region.patch_idx 不得為空")
    dg = np.asarray(dist_good, dtype=float)[:, idx].mean(axis=1)
    db = np.asarray(dist_bad, dtype=float)[:, idx].mean(axis=1)
    good_order = np.argsort(dg, kind="stable")      # 距離小 → 最典型
    bad_order = np.argsort(-db, kind="stable")      # 距離大 → 最能展示差異
    return {"good_idx": [int(i) for i in good_order[:n_each]],
            "bad_idx": [int(i) for i in bad_order[:n_each]]}


def region_summary(region: dict) -> dict:
    pct = int(round(100.0 * float(region["stability"])))
    return {"pct": pct, "text": f"Bad 群 {pct}% 的圖在此處與 Good 群明顯不同"}
