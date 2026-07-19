"""驗收:30 spatial_sig(三簽名統計 + within-wafer 受限重排 null + 四值誠實結論)。

設計:3_Architect_Design/30_spatial_sig.md。全合成 planted、無 I/O、免模型。
統計鐵則:null 只在**這片實際存在的 die 位置**間重排(die-mask 受限、保 k);
AC5 是「全格點假設會誤判、受限重排不會」的機器證明(09/10 裁決核心)。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

GRID = [(x, y) for x in range(10) for y in range(10)]     # n=100
_CX = sum(p[0] for p in GRID) / len(GRID)
_CY = sum(p[1] for p in GRID) / len(GRID)
_R = max(math.dist(p, (_CX, _CY)) for p in GRID)


def _dies(positions, defects):
    dset = set(defects)
    return [{"x": float(x), "y": float(y), "flag": int((x, y) in dset)}
            for x, y in positions]


def test_ac1_clustered_center_block():
    """# AC1(G1):中心緊鄰 6 格 → clustered 顯著(角落塊會與 edge 打平,故選中心)。"""
    from spatial_sig import spatial_signature
    defects = [(4, 4), (4, 5), (5, 4), (5, 5), (4, 6), (6, 4)]
    r = spatial_signature(_dies(GRID, defects))
    assert r["verdict"] == "clustered"
    assert r["p"]["clustered"] <= 0.05 / 3


def test_ac2_edge_ring():
    """# AC2(G1):最外圈 8 格 → edge 顯著。"""
    from spatial_sig import spatial_signature
    defects = [(0, 0), (0, 9), (9, 0), (9, 9), (0, 5), (9, 5), (5, 0), (5, 9)]
    r = spatial_signature(_dies(GRID, defects))
    assert r["verdict"] == "edge"
    assert r["p"]["edge"] <= 0.05 / 3


def test_ac3_linear_row_and_tie_priority():
    """# AC3(G1):一列 8 格共線 → linear(平手序 linear>edge>clustered 保證),
    stats.linear==1.0。"""
    from spatial_sig import spatial_signature
    defects = [(x, 5) for x in range(8)]
    r = spatial_signature(_dies(GRID, defects))
    assert r["verdict"] == "linear"
    assert r["stats"]["linear"] == pytest.approx(1.0)


def test_ac4_random_honest_single_and_fp_rate():
    """# AC4(G2):隨機缺陷單例=none;200 seeds 假陽性率 ≤ 0.075。"""
    from spatial_sig import spatial_signature
    rng = np.random.default_rng(123)
    idx = rng.choice(len(GRID), size=8, replace=False)
    r = spatial_signature(_dies(GRID, [GRID[i] for i in idx]))
    assert r["verdict"] == "none"
    hits = 0
    for s in range(200):
        rr = np.random.default_rng(s)
        ii = rr.choice(len(GRID), size=8, replace=False)
        out = spatial_signature(_dies(GRID, [GRID[i] for i in ii]))
        hits += int(out["verdict"] != "none")
    assert hits / 200 <= 0.075, f"隨機資料誤報率過高:{hits}/200"


def test_ac5_die_mask_contrast():
    """# AC5(G2②=裁決核心):環狀抽檢 mask 內的隨機缺陷——
    within-mask null 誠實 none;錯誤的全格點 mask 會誤判 edge。"""
    from spatial_sig import spatial_signature
    ring = [p for p in GRID if math.dist(p, (_CX, _CY)) >= 0.7 * _R]
    assert len(ring) >= 20, "前置:環上 die 數要過 n_min"
    rng = np.random.default_rng(7)
    idx = rng.choice(len(ring), size=6, replace=False)
    defects = [ring[i] for i in idx]
    honest = spatial_signature(_dies(ring, defects))
    assert honest["verdict"] == "none"
    naive = spatial_signature(_dies(GRID, defects))   # 錯誤假設完整格點
    assert naive["p"]["edge"] <= 0.05 / 3


def test_ac6_conservative_gates():
    """# AC6:die 太少/缺陷太少/無對比 → none+理由,p is None。"""
    from spatial_sig import spatial_signature
    small = [(x, y) for x in range(5) for y in range(2)]   # n=10
    r1 = spatial_signature(_dies(small, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]))
    assert r1["verdict"] == "none" and any("die 太少" in x for x in r1["reasons"])
    assert r1["p"] is None
    r2 = spatial_signature(_dies(GRID, [(0, 0), (1, 1)]))
    assert any("缺陷太少" in x for x in r2["reasons"])
    many = [p for p in GRID if p not in {(0, 0), (0, 1)}]  # k=98, n-k=2
    r3 = spatial_signature(_dies(GRID, many))
    assert "無對比" in "".join(r3["reasons"])


def test_ac7_deterministic():
    """# AC7:AC1 輸入呼叫兩次 → 完全相等。"""
    from spatial_sig import spatial_signature
    defects = [(4, 4), (4, 5), (5, 4), (5, 5), (4, 6), (6, 4)]
    a = spatial_signature(_dies(GRID, defects))
    b = spatial_signature(_dies(GRID, defects))
    assert a == b


def test_ac8_method_fragments():
    """# AC8:method line 固定片段。"""
    from spatial_sig import spatial_signature
    defects = [(4, 4), (4, 5), (5, 4), (5, 5), (4, 6), (6, 4)]
    m = spatial_signature(_dies(GRID, defects))["method"]
    for frag in ("受限重排", "N=", "k=", "α", "Bonferroni", "exploratory"):
        assert frag in m, f"method line 缺 {frag}:{m}"


def test_ac9_error_paths():
    """# AC9:空/壞 flag/NaN 座標/alpha 出界 → ValueError。"""
    from spatial_sig import spatial_signature
    with pytest.raises(ValueError):
        spatial_signature([])
    with pytest.raises(ValueError):
        spatial_signature([{"x": 0.0, "y": 0.0, "flag": 2}] * 25)
    with pytest.raises(ValueError):
        spatial_signature([{"x": float("nan"), "y": 0.0, "flag": 0}] * 25)
    with pytest.raises(ValueError):
        spatial_signature(_dies(GRID, [(0, 0)] ), alpha=1.5)


# ── PM 推導測試(設計未明列;metamorphic)─────────────────────────────


def test_derived_rigid_motion_invariance():
    """推導:全體座標平移+等比縮放 → 三統計量與 verdict 不變
    (clustered/edge 經 R 正規化、linear 是比例;剛體+尺度不變)。"""
    from spatial_sig import spatial_signature
    defects = [(4, 4), (4, 5), (5, 4), (5, 5), (4, 6), (6, 4)]
    base = spatial_signature(_dies(GRID, defects))
    moved_grid = [(3 * x + 100, 3 * y - 50) for x, y in GRID]
    moved_defects = [(3 * x + 100, 3 * y - 50) for x, y in defects]
    moved = spatial_signature(_dies(moved_grid, moved_defects))
    assert moved["verdict"] == base["verdict"]
    assert moved["stats"]["edge"] == pytest.approx(base["stats"]["edge"])
    assert moved["stats"]["linear"] == pytest.approx(base["stats"]["linear"])
    assert moved["p"] == base["p"]   # 同 seed、同幾何 → 同 p


def test_derived_flag_complement_not_significant_alias():
    """推導:k 與 n−k 對稱邊界——k 剛好等於 k_min(=5)可進統計,
    k_min−1 被擋(門檻邊界的 off-by-one 稽查)。"""
    from spatial_sig import spatial_signature
    r_ok = spatial_signature(_dies(GRID, [(4, 4), (4, 5), (5, 4), (5, 5), (4, 6)]))
    assert r_ok["p"] is not None            # k=5 → 有算
    r_no = spatial_signature(_dies(GRID, [(4, 4), (4, 5), (5, 4), (5, 5)]))
    assert r_no["p"] is None                # k=4 → 擋
    assert any("缺陷太少" in x for x in r_no["reasons"])
