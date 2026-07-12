"""驗收:heatmap_to_boxes(設計 3_Architect_Design/13_heatmap_to_boxes.md 之單元 AC)。

anomaly 熱力圖(patch 分數圖)→ 連通區域 → YOLO 6 欄預標框。純函式、決定論。
重點函式:boxes_from_scoremap / write_pred_labels。

先紅策略(同 test_pool_registry 慣例):目標模組 `heatmap_to_boxes` 尚未實作 →
只在 fixture / 函式內 import(collection 乾淨;未實作時為『執行期紅』test error,
非 collection error)。top-level 僅 import stdlib / numpy / pytest。

界線:本檔只放**單元** AC(AC1–AC7),合成 pmap、不需模型。GUI 的 AC-G1..G3 屬
真實 E2E(tests/e2e/test_heatmap_to_boxes_e2e.py),不寫進本檔。

數值皆照設計手算釘死(勿從實作反推)。numpy 索引為 [row, col] = [y, x]。
"""
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def htb():
    """import 隔離後的 heatmap_to_boxes(未實作 → 此處 ModuleNotFoundError → test error)。"""
    import importlib
    import heatmap_to_boxes as _htb
    importlib.reload(_htb)
    return _htb


# ─────────────────────────── 單元 AC(照設計釘死) ───────────────────────────

def test_ac1_two_hotspots_pinned(htb):  # AC1 雙熱區釘死值 + conf 降冪 + [row,col]=[y,x] 軸向
    pmap = np.zeros((20, 20), dtype=float)
    pmap[2:6, 2:6] = 1.0       # rows/cols 2..5 → cx=cy=(2+6)/2/20=0.2, w=h=4/20=0.2
    pmap[12:18, 10:16] = 0.8   # rows 12..17→cy=0.75,h=6/20=0.3;cols 10..15→cx=0.65,w=0.3
    boxes = htb.boxes_from_scoremap(pmap, thr=0.5)
    assert len(boxes) == 2                                  # 恰 2 框
    assert boxes[0] == pytest.approx((0.2, 0.2, 0.2, 0.2, 1.0))   # conf 較高在前
    assert boxes[1] == pytest.approx((0.65, 0.75, 0.3, 0.3, 0.8))  # 第二框(cx≠cy 釘死 y=row,x=col)


def test_ac2_quantile_default(htb):  # AC2 thr=None → 閾值 == np.quantile(pmap, quantile)
    pmap = np.zeros((20, 20), dtype=float)
    pmap[9:11, 9:11] = 1.0     # 4 像素 = 1% < 2% → 落在 0.98 分位「之上」
    # 契約:thr=None(預設 quantile=0.98)等價於顯式帶入 np.quantile(pmap, 0.98)
    thr = float(np.quantile(pmap, 0.98))
    boxes_default = htb.boxes_from_scoremap(pmap)
    boxes_explicit = htb.boxes_from_scoremap(pmap, thr=thr)
    assert len(boxes_default) == len(boxes_explicit) == 1   # 低分位 → 熱區存在
    assert boxes_default[0] == pytest.approx(boxes_explicit[0])
    # 臨界:高分位把閾值頂到熱點值(1.0)→ 嚴格 > 判定使熱區消失
    assert htb.boxes_from_scoremap(pmap, quantile=0.999) == []


def test_ac3_min_area_filters_speckle(htb):  # AC3 min_area_frac 濾單像素雜點
    pmap = np.zeros((100, 100), dtype=float)
    pmap[5, 5] = 1.0            # 單像素雜點,面積 1 < 0.001*10000=10 → 濾除
    pmap[50:60, 50:60] = 1.0    # 10×10 熱區,面積 100 → 保留
    boxes = htb.boxes_from_scoremap(pmap, thr=0.5, min_area_frac=0.001)
    assert len(boxes) == 1
    # rows/cols 50..59 → 外接框 50..60 → cx=cy=(50+60)/2/100=0.55, w=h=10/100=0.1
    assert boxes[0] == pytest.approx((0.55, 0.55, 0.1, 0.1, 1.0))


def test_ac4_empty_map(htb):  # AC4 全零 pmap → []
    assert htb.boxes_from_scoremap(np.zeros((20, 20), dtype=float)) == []


def test_ac5_non_2d_raises(htb):  # AC5 非 2D 輸入拋 ValueError(不靜默壓維)
    with pytest.raises(ValueError):
        htb.boxes_from_scoremap(np.zeros((4, 4, 3), dtype=float))
    with pytest.raises(ValueError):
        htb.boxes_from_scoremap(np.zeros((16,), dtype=float))


def test_ac6_write_round_trip(htb, tmp_path):  # AC6 write_pred_labels 寫檔往返
    import interaction  # 跨模組 parser(讀回驗證用);產品碼不循環,測試層 import 即可
    pmap_a = np.zeros((20, 20), dtype=float)
    pmap_a[2:6, 2:6] = 1.0
    pmap_a[12:18, 10:16] = 0.8      # 有框圖(2 框,conf 1.0 / 0.8)
    pmap_b = np.zeros((20, 20), dtype=float)   # 無框圖(全零)
    img_a = tmp_path / "imgs" / "a.png"
    img_b = tmp_path / "imgs" / "b.png"
    out_root = tmp_path / "pred_out"

    stats = htb.write_pred_labels([(img_a, pmap_a), (img_b, pmap_b)], out_root, thr=0.5)

    labels_dir = out_root / "labels"
    assert (labels_dir / "a.txt").exists()          # 依 stem 命名
    assert (labels_dir / "b.txt").exists()
    # 有框者:parse_yolo_boxes_conf 讀回框數 / conf 一致
    boxes_a = interaction.parse_yolo_boxes_conf(labels_dir / "a.txt")
    assert len(boxes_a) == 2
    assert sorted((b[5] for b in boxes_a), reverse=True) == pytest.approx([1.0, 0.8])
    # 無框者:空檔(語義=「有預測來源、此圖無框」)
    assert (labels_dir / "b.txt").read_text(encoding="utf-8").strip() == ""
    # classes.txt 單行 "defect"
    assert (out_root / "classes.txt").read_text(encoding="utf-8").strip() == "defect"
    # 回傳統計正確
    assert stats["n_images"] == 2
    assert stats["n_boxes"] == 2
    assert Path(stats["out_root"]) == out_root


def test_ac7_idempotent_overwrite(htb, tmp_path):  # AC7 改參數重呼 → 舊框不殘留
    import interaction
    pmap = np.zeros((20, 20), dtype=float)
    pmap[2:6, 2:6] = 1.0        # conf 1.0 熱區
    pmap[12:18, 10:16] = 0.8    # conf 0.8 熱區
    img = tmp_path / "x.png"
    out_root = tmp_path / "pred_out"

    htb.write_pred_labels([(img, pmap)], out_root, thr=0.5)   # 低閾值 → 2 框
    b1 = interaction.parse_yolo_boxes_conf(out_root / "labels" / "x.txt")
    assert len(b1) == 2

    htb.write_pred_labels([(img, pmap)], out_root, thr=0.9)   # 高閾值 → 只剩 1.0 熱區
    b2 = interaction.parse_yolo_boxes_conf(out_root / "labels" / "x.txt")
    assert len(b2) == 1                                       # 舊 2 框未殘留,只反映新參數
    assert b2[0][5] == pytest.approx(1.0)


def test_ac8_crop_to_image_space(htb):  # AC8 裁切座標 → 原圖座標(設計 §2.1 反向閘門)
    """物件模式的 pmap 是**單一物件裁切圖**的分數圖,框天生落在裁切座標系;寫成預標前
    必須映射回原圖,否則下游拿它跟原圖 GT 比 IoU 必錯。整張影像模式(region==全圖)
    則必須退化成 identity —— 這條同時鎖住兩邊。"""
    # ① region == 全圖 → identity(整張影像模式不得被這層改動)
    assert htb.box_to_image_space((0.5, 0.5, 1.0, 1.0, 0.9), (0, 0, 400, 200), 400, 200) \
        == pytest.approx((0.5, 0.5, 1.0, 1.0, 0.9))
    # ② 400×200 原圖、region=右半 (200,0,400,200)、裁切內置中框 (0.5,0.5,0.5,0.5)
    #    手算:cx=(200+0.5*200)/400=0.75、cy=(0+0.5*200)/200=0.5、w=0.5*200/400=0.25、h=0.5*200/200=0.5
    assert htb.box_to_image_space((0.5, 0.5, 0.5, 0.5, 0.7), (200, 0, 400, 200), 400, 200) \
        == pytest.approx((0.75, 0.5, 0.25, 0.5, 0.7))

    # ③ 同一張圖的兩個物件 → 該圖回兩個框(合併到同一份預標,不互相覆蓋),conf 降冪
    left = np.zeros((20, 20), dtype=float)
    left[8:12, 8:12] = 0.6
    right = np.zeros((20, 20), dtype=float)
    right[8:12, 8:12] = 1.0
    by_img = htb.boxes_by_image([
        {"image_path": "img.png", "pmap": left,
         "image_size": (400, 200), "region": (0, 0, 200, 200)},
        {"image_path": "img.png", "pmap": right,
         "image_size": (400, 200), "region": (200, 0, 400, 200)},
    ], thr=0.5)
    boxes = by_img["img.png"]
    assert len(boxes) == 2                                  # 兩個物件的框都在
    assert [b[4] for b in boxes] == pytest.approx([1.0, 0.6])   # conf 降冪
    assert boxes[0][0] > 0.5 > boxes[1][0]                  # 強的那個在右半、弱的在左半


# ───────────── 推導 metamorphic 測試(設計未明列,PM 自加不變量) ─────────────

def test_derived_translation_equivariance(htb):  # 推導:平移熱區 → 框中心等量平移,尺寸/信心/框數不變
    """boxes_from_scoremap 應對整塊熱區的純平移具等變性:框只跟著移,w/h/conf/框數
    都不變。此不變量能逼出『把 row/col 混用』或『用絕對像素而非正規化中心』的實作 bug。"""
    base = np.zeros((20, 20), dtype=float)
    base[4:8, 4:8] = 1.0
    shifted = np.zeros((20, 20), dtype=float)
    shifted[4:8, 10:14] = 1.0          # 沿 x(col)右移 6 格,y 不動
    b0 = htb.boxes_from_scoremap(base, thr=0.5)
    b1 = htb.boxes_from_scoremap(shifted, thr=0.5)
    assert len(b0) == len(b1) == 1
    cx0, cy0, w0, h0, c0 = b0[0]
    cx1, cy1, w1, h1, c1 = b1[0]
    assert (w1, h1, c1) == pytest.approx((w0, h0, c0))   # 尺寸/信心不變
    assert cy1 == pytest.approx(cy0)                     # y 中心不變
    assert cx1 - cx0 == pytest.approx(6 / 20)            # x 中心等量平移 6/20


def test_derived_conf_sorted_descending(htb):  # 推導:多熱區輸出恆按 conf 降冪(排序不變量)
    """『依 conf 降冪』對任意數量/擺放的熱區都成立,而不只是 AC1 的兩框。"""
    pmap = np.zeros((30, 30), dtype=float)
    pmap[2:6, 2:6] = 0.6
    pmap[2:6, 20:24] = 1.0
    pmap[20:24, 2:6] = 0.8
    boxes = htb.boxes_from_scoremap(pmap, thr=0.5)
    assert len(boxes) == 3
    confs = [b[4] for b in boxes]
    assert confs == sorted(confs, reverse=True)          # 單調非遞增
    assert confs[0] == pytest.approx(1.0)                # 最強熱區排第一


def test_derived_constant_map_empty(htb):  # 推導/邊界:全同值圖 → [](quantile 閾值=該值,嚴格 > → 空)
    """任一常數圖(非零)都應回 [] —— 沒有相對突起就沒有異常區。這條把『全零』一般化為
    『全同值』,對齊設計邊界表且是強不變量。"""
    assert htb.boxes_from_scoremap(np.full((20, 20), 0.7, dtype=float)) == []
