"""驗收:gt_pred_diff(設計 3_Architect_Design/10_gt_pred_diff.md §5「單元」AC)。

GT×預測比對挖錯的純函式:iou_cxcywh / match_boxes / image_error_report /
pred_label_path_for / scan_dataset。純合成 box,不需影像檔。

先紅策略:目標模組 `gt_pred_diff` 尚未實作 → 於 fixture/函式內 import
(collection 乾淨;未實作時為『執行期紅』ModuleNotFoundError,非 collection error)。
top-level 只 import stdlib(pathlib)+ pytest。每測註解標對應 # ACn。

────────────────────────────────────────────────────────────────────────────
⚠ REVERSE-GATE → /architect(設計自相矛盾,PM 不代為裁決)
  §5 AC1 第三式與 §6 提示要求:
      iou_cxcywh((.5,.5,.5,.5),(.625,.5,.5,.5)) == 3/7
  但依 §2.1 的 IoU 定義手算(兩框各 .5×.5、水平位移 .125):
      交集 = .375 × .5                     = .1875
      聯集 = .25 + .25 − .1875             = .3125   ← §6 誤寫成 .4375
      IoU  = .1875 / .3125                 = 0.6 = 3/5   (不是 3/7)
  要得 3/7 需位移 .2(即座標 .7),與給定的 .625 不符 →「座標 .625」與
  「pin 值 3/7」二者必有一誤。依鐵則3(雙向閘門),修法(改 pin→3/5 或改座標→.7)
  是 architect 的決定,PM 不逕自選定,故此第三式暫不落測(見下方註解),
  待 /architect 修正後補。partial-overlap 區間改由 test_metamorphic_iou_properties
  以自導、無爭議的位移 .25(IoU=1/3)覆蓋,避免留白。
────────────────────────────────────────────────────────────────────────────
"""
from pathlib import Path

import pytest


@pytest.fixture
def gpd():
    """import 隔離後的 gt_pred_diff(純函式、無 env 依賴;未實作 → ModuleNotFoundError)。"""
    import importlib
    import gt_pred_diff as _m
    importlib.reload(_m)
    return _m


# ── AC1:IoU 釘死(手算)─────────────────────────────────────────────────────
def test_ac1_iou_identical_and_disjoint(gpd):  # AC1
    # 完全重合 → 1.0
    assert gpd.iou_cxcywh((.5, .5, .4, .4), (.5, .5, .4, .4)) == 1.0
    # 水平相鄰、僅邊界相接、面積無交集 → 0.0
    assert gpd.iou_cxcywh((.25, .5, .5, 1.), (.75, .5, .5, 1.)) == 0.0
    # 第三式(部分重疊):架構師已修 pin 3/7→0.6(交 .1875 / 聯 .3125 = 0.6),見檔頭。
    assert abs(gpd.iou_cxcywh((.5, .5, .5, .5), (.625, .5, .5, .5)) - 0.6) < 1e-9


# ── AC2:tp / fn / fp ────────────────────────────────────────────────────────
def test_ac2_tp_fn_fp(gpd):  # AC2
    gt = [(0, .25, .25, .2, .2),        # gi=0:被 pred 命中 → tp
          (0, .75, .75, .2, .2)]        # gi=1:無 pred 命中 → fn
    pred = [(0, .25, .25, .2, .2, .9),      # pi=0:與 gt0 完全重合、同類 → tp(iou=1.0)
            (0, .05, .95, .06, .06, .8)]    # pi=1:遠框、與任何 gt 無交集 → fp
    r = gpd.match_boxes(gt, pred, iou_thr=.5)
    assert r["tp"] == [(0, 0, 1.0)]
    assert r["fn"] == [1]
    assert r["fp"] == [1]
    assert r["confused"] == []


# ── AC3:類別混淆 ────────────────────────────────────────────────────────────
def test_ac3_class_confusion(gpd):  # AC3
    gt = [(0, .5, .5, .4, .4)]
    pred = [(1, .5, .5, .4, .4, .9)]   # 同位置(IoU=1.0)但類別 id 不同 → confused
    r = gpd.match_boxes(gt, pred, iou_thr=.5)
    assert r["confused"] == [(0, 0, 1.0)]
    assert r["tp"] == []               # 已從 tp 移到 confused
    assert r["fn"] == []               # gt 已被占用(不算漏抓)
    assert r["fp"] == []               # pred 已被占用(不算誤抓)


# ── AC4:貪婪決定論 ──────────────────────────────────────────────────────────
def test_ac4_greedy_determinism(gpd):  # AC4
    gt = [(0, .5, .5, .4, .4)]
    # 兩個 pred 都完全蓋住同一 gt、同類;把低 conf 放前面,驗證『依 conf 而非索引』貪婪
    pred = [(0, .5, .5, .4, .4, .6),       # pi=0,conf .6
            (0, .5, .5, .4, .4, .9)]       # pi=1,conf .9 → 應搶到 tp
    first = gpd.match_boxes(gt, pred, iou_thr=.5)
    assert first["tp"] == [(0, 1, 1.0)]    # 高 conf(pi=1)得 tp
    assert first["fp"] == [0]              # 低 conf(pi=0)落 fp
    assert first["fn"] == []
    assert first["confused"] == []
    # 決定論:重跑 100 次結果完全相同(不依賴 set / dict 迭代順序)
    for _ in range(100):
        assert gpd.match_boxes(gt, pred, iou_thr=.5) == first


# ── AC5:low_conf 與 score ───────────────────────────────────────────────────
def test_ac5_low_conf_and_score(gpd):  # AC5
    gt = [(0, .2, .2, .2, .2),          # gi=0:被 pred0 命中 → tp
          (0, .8, .8, .2, .2)]          # gi=1:無命中 → fn(1)
    pred = [(0, .2, .2, .2, .2, .4),        # pi=0:命中 gt0(tp);conf .4<.5 → 也入 low_conf(含已匹配者)
            (0, .5, .05, .08, .08, .3)]     # pi=1:遠框 → fp(1);conf .3<.5 → low_conf
    r = gpd.image_error_report(gt, pred, iou_thr=.5, low_conf=.5)
    assert len(r["fn"]) == 1
    assert len(r["fp"]) == 1
    assert len(r["confused"]) == 0
    assert sorted(r["low_conf"]) == [0, 1]     # 兩個低信心 pred(含已匹配的 pi=0)
    assert r["score"] == 4.0                    # 2*1 + 1*1 + 2*0 + 0.5*2 = 4.0


# ── AC6:pred 佈局雙形(labels/ 子目錄 vs 平鋪)───────────────────────────────
def test_ac6_pred_layout_dual(gpd, tmp_path):  # AC6
    img = tmp_path / "images" / "frame_007.jpg"    # 只用其 stem;影像檔不必存在
    # (A)有 labels/ 子目錄 → 用 YOLO 慣例路徑
    root_a = tmp_path / "pred_a"
    (root_a / "labels").mkdir(parents=True)
    assert gpd.pred_label_path_for(img, root_a) == root_a / "labels" / "frame_007.txt"
    # (B)無 labels/ 子目錄 → 平鋪
    root_b = tmp_path / "pred_b"
    root_b.mkdir()
    assert gpd.pred_label_path_for(img, root_b) == root_b / "frame_007.txt"


# ── AC7:scan 依 score 降冪 + 兩邊皆空剔除 ────────────────────────────────────
def test_ac7_scan_orders_by_score_and_drops_empty(gpd, tmp_path):  # AC7
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    pred_root = tmp_path / "pred"          # 平鋪佈局(無 labels/ 子目錄)
    pred_root.mkdir()

    def img(name):
        return root / "images" / f"{name}.jpg"

    def gt_txt(name, text):
        (root / "labels" / f"{name}.txt").write_text(text, encoding="utf-8")

    def pred_txt(name, text):
        (pred_root / f"{name}.txt").write_text(text, encoding="utf-8")

    # imgA:score=4(fn=1, fp=1, low_conf=2)—— 同 AC5 佈局
    gt_txt("imgA", "0 0.2 0.2 0.2 0.2\n0 0.8 0.8 0.2 0.2\n")
    pred_txt("imgA", "0 0.2 0.2 0.2 0.2 0.4\n0 0.5 0.05 0.08 0.08 0.3\n")
    # imgB:兩邊皆空(空 gt 檔存在 → 不觸發標註 fallback;無 pred 檔)→ 不進列表
    gt_txt("imgB", "")
    # imgC:score=7(fn=3, fp=1)—— 3 個未命中 gt + 1 個遠 pred(conf .9≥.5,不入 low_conf)
    gt_txt("imgC", "0 0.1 0.1 0.1 0.1\n0 0.5 0.5 0.1 0.1\n0 0.9 0.9 0.1 0.1\n")
    pred_txt("imgC", "0 0.3 0.7 0.06 0.06 0.9\n")

    records = [{"path": str(img("imgA"))},
               {"path": str(img("imgB"))},
               {"path": str(img("imgC"))}]
    out = gpd.scan_dataset(records, pred_root, iou_thr=.5, low_conf=.5)
    assert len(out) == 2                                    # imgB 兩邊皆空被剔除
    assert [Path(o["path"]).stem for o in out] == ["imgC", "imgA"]  # score 7 → 4 降冪
    assert out[0]["report"]["score"] == 7.0
    assert out[1]["report"]["score"] == 4.0
    assert (out[0]["n_gt"], out[0]["n_pred"]) == (3, 1)     # imgC 框數
    assert (out[1]["n_gt"], out[1]["n_pred"]) == (2, 2)     # imgA 框數


# ── AC8:預標 seed 來源契約(pred 但無 pred_root → ValueError)───────────────
def test_ac8_seed_source_pred_requires_pred_root():  # AC8
    # labeling_handoff.send_to_labeling 擴充 seed_source/pred_root(延後 import;
    # 擴充未落地時 seed_source/pred_root 為未知 kwarg → TypeError,亦為紅)。
    # 註:AC8 的「pred seed 框數==pred txt 框數且含 conf」與「預設 byte-level 回歸」
    #     需真實影像 fixture 且與 AC-G4(E2E)重疊,屬 tests/e2e/,本單元檔僅釘契約邊界。
    import labeling_handoff  # noqa: PLC0415
    with pytest.raises(ValueError):
        labeling_handoff.send_to_labeling(
            [{"path": "x.jpg"}], [0],
            source="viz", task="detect", class_options=["a"],
            seed_source="pred", pred_root=None,
        )


# ── AC8(續)/ AC9:預標 seed 落盤 + 「未動過的預標不是標註」──────────────────
def _seed_fixture(tmp_path):
    """一張真圖 + 一份 2 框的 pred txt(labels/ 佈局)→ (records, pred_root, log_dir)。"""
    from PIL import Image
    img_dir = tmp_path / "ds" / "images"
    img_dir.mkdir(parents=True)
    img = img_dir / "a.png"
    Image.new("RGB", (100, 80), "white").save(img)

    pred_root = tmp_path / "pred"
    (pred_root / "labels").mkdir(parents=True)
    (pred_root / "labels" / "a.txt").write_text(
        "0 0.25 0.5 0.2 0.4 0.9\n0 0.75 0.5 0.2 0.4 0.6\n", encoding="utf-8")

    return [{"path": str(img), "label": "orig"}], pred_root, tmp_path / "log"


def test_ac8_pred_seed_lands_with_conf(tmp_path):  # AC8 seed=pred:框數==pred 框數且帶 conf
    import json as _json

    import labeling_handoff as LH
    records, pred_root, log_dir = _seed_fixture(tmp_path)
    out = LH.send_to_labeling(records, [0], source="viz", task="detect",
                              class_options=["defect"], seed_source="pred",
                              pred_root=pred_root, log_dir=log_dir)
    assert out is not None
    spec = LH.load_spec(out)
    sha = spec["items"][0]["sha256"]
    sidecar = _json.loads((out / "images" / f"{sha}.json").read_text(encoding="utf-8"))
    assert len(sidecar["shapes"]) == 2                      # 框數 == pred txt 框數
    assert all(s["shape_type"] == "rectangle" for s in sidecar["shapes"])
    assert sorted(s["score"] for s in sidecar["shapes"]) == pytest.approx([0.6, 0.9])  # 帶 conf
    assert spec["items"][0].get("seed_sha")                 # seed 內容 hash 記進 spec


def test_ac9_untouched_seed_is_not_a_label(tmp_path):  # AC9 未經人手的預標不得被當人工標註
    """設計 10 §2.2 反向閘門:模型自己的預測若被當成人工答案讀回,模型就會把自己的錯誤
    當真值餵回自己 —— 主動學習最該避免的失敗模式。這條測就是那道防線。"""
    import json as _json

    import labeling_handoff as LH
    records, pred_root, log_dir = _seed_fixture(tmp_path)
    out = LH.send_to_labeling(records, [0], source="viz", task="detect",
                              class_options=["defect"], seed_source="pred",
                              pred_root=pred_root, log_dir=log_dir)
    sha = LH.load_spec(out)["items"][0]["sha256"]
    ann = out / "images" / f"{sha}.json"

    # ① 送標當下不得顯示「已標註」
    assert LH.handoff_status(out)["n_annotated"] == 0
    # ② 讀回:沒有 label、狀態 pending
    res = LH.read_labeling_results(out)
    assert res[sha]["label"] is None
    assert res[sha]["status"] == "pending"
    # ③ apply_readback 不得改動任何 record 的 label
    changes = LH.apply_readback(out, records)
    assert changes == []
    assert records[0]["label"] == "orig"                    # 模型的預測沒有變成真值

    # ④ 人真的動過(sidecar 內容改變)→ 就算數了
    data = _json.loads(ann.read_text(encoding="utf-8"))
    data["shapes"][0]["label"] = "scratch"                  # 標註者改了類別
    ann.write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert LH.handoff_status(out)["n_annotated"] == 1
    assert LH.read_labeling_results(out)[sha]["label"] == "scratch"


# ── 推導(metamorphic / property;設計未明列)────────────────────────────────
def test_metamorphic_iou_properties(gpd):  # 推導:IoU 不變量(自反 / 範圍 / 對稱 / 退化 / partial)
    """IoU 的結構性不變量,逼出『非對稱』『>1 或 <0』『退化框未防呆』等實作 bug。"""
    boxes = [(.5, .5, .4, .4), (.55, .5, .4, .4), (.2, .8, .3, .3), (.5, .5, .8, .2)]
    for a in boxes:
        assert gpd.iou_cxcywh(a, a) == 1.0                          # 自反 = 1
        for b in boxes:
            v = gpd.iou_cxcywh(a, b)
            assert 0.0 <= v <= 1.0                                  # 範圍 [0,1]
            assert abs(v - gpd.iou_cxcywh(b, a)) < 1e-12            # 對稱
    # 退化框(w<=0 或 h<=0)→ 0.0(設計 §2.1)
    assert gpd.iou_cxcywh((.5, .5, 0., .4), (.5, .5, .4, .4)) == 0.0
    assert gpd.iou_cxcywh((.5, .5, .4, -.1), (.5, .5, .4, .4)) == 0.0
    # 自導 partial-overlap 手算(位移 .25:交集 .25×.5=.125 / 聯集 .25+.25−.125=.375 = 1/3)。
    # 覆蓋被 reverse-gate 暫緩的 AC1 第三式所屬之「部分重疊」區間。
    assert abs(gpd.iou_cxcywh((.5, .5, .5, .5), (.75, .5, .5, .5)) - 1 / 3) < 1e-9


def test_metamorphic_match_conservation(gpd):  # 推導:match_boxes 分割守恆(不重不漏)
    """每個 gt / pred 索引恰好落入一個桶 —— 逼出『雙重計數』或『漏配』的折疊 bug。
    此不變量不釘死各框歸屬,只斷言分割完整性,對合理的實作差異穩健。"""
    gt = [(0, .1, .1, .1, .1),          # → tp(與 p0)
          (0, .3, .3, .1, .1),          # → confused(與 p1,異類)
          (0, .5, .5, .1, .1),          # → fn
          (0, .7, .7, .1, .1)]          # → tp(與 p2)
    pred = [(0, .1, .1, .1, .1, .9),
            (1, .3, .3, .1, .1, .8),        # 異類 → confused
            (0, .7, .7, .1, .1, .7),
            (0, .92, .04, .05, .05, .6),    # 遠 → fp
            (0, .04, .92, .05, .05, .5)]    # 遠 → fp
    r = gpd.match_boxes(gt, pred, iou_thr=.5)
    G, P = len(gt), len(pred)
    # 每個 gt 恰出現一次:tp.gi ∪ confused.gi ∪ fn == range(G)
    gt_seen = [t[0] for t in r["tp"]] + [c[0] for c in r["confused"]] + list(r["fn"])
    assert sorted(gt_seen) == list(range(G))                # 不重不漏
    # 每個 pred 恰出現一次:tp.pi ∪ confused.pi ∪ fp == range(P)
    pred_seen = [t[1] for t in r["tp"]] + [c[1] for c in r["confused"]] + list(r["fp"])
    assert sorted(pred_seen) == list(range(P))
    # 計數守恆
    assert len(r["tp"]) + len(r["confused"]) + len(r["fn"]) == G
    assert len(r["tp"]) + len(r["confused"]) + len(r["fp"]) == P
