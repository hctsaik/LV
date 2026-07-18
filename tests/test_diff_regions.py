"""驗收:21 diff_regions(差異分數圖 → Top-K 區域 + 代表樣本 + 白話摘要)。

設計:3_Architect_Design/21_diff_regions.md。全部合成 z_map / dist 矩陣、手算釘死。
"""
from __future__ import annotations

import numpy as np
import pytest


def test_ac1_two_regions_pinned():
    """# AC1:雙熱區釘死——patch_idx / bbox / 峰值 / stability / rank 全手算。"""
    from diff_regions import extract_regions
    z = np.zeros((6, 6))
    z[1:3, 1:3] = 5.0
    z[1, 1] = 6.0
    z[4, 4] = 3.0
    st = np.zeros((6, 6))
    st[1:3, 1:3] = 0.9
    st[4, 4] = 0.5
    regs = extract_regions(z, st, z_thresh=2.0)
    assert len(regs) == 2
    r1, r2 = regs
    assert r1["rank"] == 1 and r2["rank"] == 2
    assert r1["peak_idx"] == 7 and r1["peak_z"] == 6.0          # (1,1) → 1*6+1
    assert sorted(r1["patch_idx"]) == [7, 8, 13, 14]
    assert r1["bbox_grid"] == (1, 1, 3, 3)
    assert r1["stability"] == 0.9
    assert r2["peak_idx"] == 28 and r2["peak_z"] == 3.0          # (4,4) → 4*6+4
    assert r2["bbox_grid"] == (4, 4, 5, 5)
    assert r2["stability"] == 0.5


def test_ac2_all_below_threshold_empty():
    """# AC2:全低於門檻 → [](誠實空結果)。"""
    from diff_regions import extract_regions
    regs = extract_regions(np.zeros((5, 5)), np.zeros((5, 5)), z_thresh=1.0)
    assert regs == []


def test_ac3_diagonal_8_connectivity():
    """# AC3:斜角相鄰 → 8 連通合併成同一區域。"""
    from diff_regions import extract_regions
    z = np.zeros((4, 4))
    z[1, 1] = 3.0
    z[2, 2] = 3.0
    regs = extract_regions(z, np.zeros((4, 4)), z_thresh=2.0)
    assert len(regs) == 1
    assert sorted(regs[0]["patch_idx"]) == [5, 10]


def test_ac4_top_k_and_min_patches():
    """# AC4:top_k 截斷 + min_patches 濾單點。"""
    from diff_regions import extract_regions
    z = np.zeros((7, 7))
    z[0, 0] = 5.0                       # 單 patch
    z[2, 3] = 4.0                       # 單 patch
    z[5, 5] = 3.0
    z[5, 6] = 3.0                       # 2 patch 區域(峰 3.0)
    st = np.zeros((7, 7))
    top2 = extract_regions(z, st, z_thresh=2.0, top_k=2)
    assert [r["rank"] for r in top2] == [1, 2]
    assert top2[0]["peak_z"] == 5.0 and top2[1]["peak_z"] == 4.0
    only_multi = extract_regions(z, st, z_thresh=2.0, min_patches=2)
    assert len(only_multi) == 1
    assert sorted(only_multi[0]["patch_idx"]) == [40, 41]        # (5,5)=40、(5,6)=41
    assert extract_regions(z, st, z_thresh=2.0, top_k=0) == []


def test_ac5_representatives_handpicked():
    """# AC5:代表樣本手算——Bad 取區內平均距離最大、Good 取最小;n_each 超量回全部。"""
    from diff_regions import pick_representatives
    region = {"patch_idx": [0, 1]}
    dist_bad = np.array([[.1, .1], [.9, .8], [.5, .5]])
    dist_good = np.array([[.3, .3], [.05, .05], [.2, .2]])
    reps = pick_representatives(region, dist_good, dist_bad, n_each=2)
    assert reps["bad_idx"] == [1, 2]
    assert reps["good_idx"] == [1, 2]
    all_reps = pick_representatives(region, dist_good, dist_bad, n_each=5)
    assert all_reps["bad_idx"] == [1, 2, 0]
    assert all_reps["good_idx"] == [1, 2, 0]


def test_ac6_summary_text():
    """# AC6:白話句——pct 四捨五入、固定句型。"""
    from diff_regions import region_summary
    s = region_summary({"stability": 0.82})
    assert s["pct"] == 82
    assert s["text"] == "Bad 群 82% 的圖在此處與 Good 群明顯不同"
    assert region_summary({"stability": 0.005})["pct"] == 0


def test_derived_peak_inside_region():
    """# 推導:每區 peak_idx ∈ patch_idx,且 peak 的 (r,c) 落在 bbox_grid 內。

    施壓「row/col 對調」「bbox 半開區間寫錯」類 bug。
    """
    from diff_regions import extract_regions
    z = np.zeros((5, 8))
    z[1:4, 2:5] = 3.0
    z[2, 3] = 7.0
    z[0, 7] = 4.0
    regs = extract_regions(z, np.zeros((5, 8)), z_thresh=2.5)
    assert regs
    for r in regs:
        assert r["peak_idx"] in r["patch_idx"]
        pr, pc = divmod(r["peak_idx"], 8)
        r0, c0, r1, c1 = r["bbox_grid"]
        assert r0 <= pr < r1 and c0 <= pc < c1


def test_ac7_validation_errors():
    """# AC7:3D z_map / 形狀不一致 / 空 patch_idx → ValueError。"""
    from diff_regions import extract_regions, pick_representatives
    with pytest.raises(ValueError):
        extract_regions(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), z_thresh=1.0)
    with pytest.raises(ValueError):
        extract_regions(np.zeros((3, 3)), np.zeros((4, 4)), z_thresh=1.0)
    with pytest.raises(ValueError):
        pick_representatives({"patch_idx": []}, np.zeros((2, 4)), np.zeros((2, 4)))
