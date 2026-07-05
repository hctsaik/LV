"""驗收:11 prelabel(看過類別分類預標 → 安全 YOLO 標註檔)。

設計:3_Architect_Design/11_prelabel.md;對應表:4_PM_Feedback/prelabel.md。
每測註對應的 # ACn。需 head 的 AC 用 dino_head.train_head 在線性可分 2D embedding 上訓真頭
(判定確定),閘控走向用「分數 vs 門檻」+ 極端 min_conf 強制(不依賴機率精確值)。

延遲匯入 prelabel(scripts/prelabel.py 由 /pg 建立):先紅時為 test failure 而非 collection error。
"""
import numpy as np
import pytest


def _pl():
    import prelabel
    return prelabel


@pytest.fixture
def head():
    """真頭:scratch(near [1,0])×4 + stain(near [0,1])×4,線性可分。classes=['scratch','stain']。"""
    from dino_head import train_head
    emb = np.array([[1.0, 0.02], [0.98, 0.0], [0.99, 0.03], [1.0, 0.01],
                    [0.0, 1.0], [0.02, 0.98], [0.01, 0.99], [0.03, 1.0]], dtype=np.float32)
    labels = ["scratch"] * 4 + ["stain"] * 4
    return train_head(emb, labels)


# ── prelabel_records ────────────────────────────────────────────────────────
def test_ac1_normal_gated_out(head):  # AC1:異常低於門檻 → 正常分流、不 accept
    pl = _pl()
    recs = [{"image_path": "/d/a.jpg", "bbox": (0.5, 0.5, 0.4, 0.6)},
            {"image_path": "/d/a.jpg", "bbox": (0.25, 0.75, 0.2, 0.2)}]
    emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    out = pl.prelabel_records(recs, emb, [0.1, 0.9], head, anomaly_threshold=0.5, min_conf=0.0)
    assert out[0]["pred"] == "正常" and out[0]["accept"] is False
    assert out[1]["pred"] in ("scratch", "stain") and out[1]["accept"] is True


def test_ac2_unknown_left_blank(head):  # AC2:異常高但沒把握(min_conf 不可達) → Unknown、留白
    pl = _pl()
    recs = [{"image_path": "/d/a.jpg", "bbox": (0.5, 0.5, 0.4, 0.6)}]
    emb = np.array([[1.0, 0.0]], dtype=np.float32)
    out = pl.prelabel_records(recs, emb, [0.9], head, anomaly_threshold=0.5, min_conf=1.1)
    assert out[0]["pred"] == "Unknown" and out[0]["accept"] is False


def test_ac3_target_class_filter(head):  # AC3:只接受選定類別的預標
    pl = _pl()
    recs = [{"image_path": "/d/a.jpg", "bbox": (0.5, 0.5, 0.4, 0.6)}]
    emb = np.array([[1.0, 0.0]], dtype=np.float32)   # near scratch
    r_scr = pl.prelabel_records(recs, emb, [0.9], head, anomaly_threshold=0.5,
                                min_conf=0.0, target_classes=["scratch"])
    assert r_scr[0]["pred"] == "scratch" and r_scr[0]["accept"] is True
    r_st = pl.prelabel_records(recs, emb, [0.9], head, anomaly_threshold=0.5,
                               min_conf=0.0, target_classes=["stain"])
    assert r_st[0]["pred"] == "scratch" and r_st[0]["accept"] is False


def test_ac4_head_none_raises():  # AC4:無分類頭 → raise,不靜默亂猜
    pl = _pl()
    with pytest.raises(ValueError, match="分類頭"):
        pl.prelabel_records([{"image_path": "/a.jpg", "bbox": (0.5, 0.5, 0.4, 0.6)}],
                            np.array([[1.0, 0.0]], dtype=np.float32), [0.5], None,
                            anomaly_threshold=0.5)


def test_ac5_length_mismatch_raises(head):  # AC5:長度不符 → raise
    pl = _pl()
    recs = [{"image_path": "/a", "bbox": (0.5, 0.5, 0.4, 0.6)},
            {"image_path": "/b", "bbox": (0.5, 0.5, 0.4, 0.6)}]
    with pytest.raises(ValueError):
        pl.prelabel_records(recs, np.zeros((3, 2), dtype=np.float32), [0.1, 0.2, 0.3],
                            head, anomaly_threshold=0.5)


def test_ac6_empty(head):  # AC6:空輸入 → []
    pl = _pl()
    assert pl.prelabel_records([], np.zeros((0, 2), dtype=np.float32), [], head,
                               anomaly_threshold=0.5) == []


# ── to_yolo_lines ───────────────────────────────────────────────────────────
def test_ac7_to_yolo_only_accept_format():  # AC7:只含 accept + 6 位小數格式 + cls_id
    pl = _pl()
    recs = [{"image_path": "/d/img_a.jpg", "bbox": (0.5, 0.5, 0.4, 0.6)},
            {"image_path": "/d/img_a.jpg", "bbox": (0.25, 0.75, 0.2, 0.2)}]
    dec = [{"item": 0, "pred": "scratch", "conf": 0.9, "accept": True},
           {"item": 1, "pred": "stain", "conf": 0.8, "accept": False}]
    out = pl.to_yolo_lines(recs, dec, class_names=["scratch", "stain"])
    assert out == {"/d/img_a.jpg": ["0 0.500000 0.500000 0.400000 0.600000"]}


def test_ac8_roundtrip():  # AC8:座標往返誤差 < 1e-6
    pl = _pl()
    recs = [{"image_path": "/a.jpg", "bbox": (0.5, 0.5, 0.4, 0.6)}]
    dec = [{"item": 0, "pred": "scratch", "accept": True}]
    line = pl.to_yolo_lines(recs, dec, class_names=["scratch", "stain"])["/a.jpg"][0]
    cx, cy, w, h = map(float, line.split()[1:])
    for got, exp in zip((cx, cy, w, h), (0.5, 0.5, 0.4, 0.6)):
        assert abs(got - exp) < 1e-6


def test_ac9_unknown_class_raises():  # AC9:pred 不在 class_names → raise(含該名)
    pl = _pl()
    with pytest.raises(ValueError, match="ghost"):
        pl.to_yolo_lines([{"image_path": "/a.jpg", "bbox": (0.5, 0.5, 0.4, 0.6)}],
                         [{"item": 0, "pred": "ghost", "accept": True}],
                         class_names=["scratch", "stain"])


def test_ac10_normal_unknown_not_output():  # AC10:正常/Unknown 不輸出(雙保險)
    pl = _pl()
    out = pl.to_yolo_lines([{"image_path": "/a.jpg", "bbox": (0.5, 0.5, 0.4, 0.6)}],
                           [{"item": 0, "pred": "正常", "accept": True}],
                           class_names=["scratch", "stain"])
    assert out == {}


def test_ac11_multi_image_group():  # AC11:多圖分組
    pl = _pl()
    recs = [{"image_path": "/a.jpg", "bbox": (0.5, 0.5, 0.4, 0.6)},
            {"image_path": "/b.jpg", "bbox": (0.3, 0.3, 0.2, 0.2)}]
    dec = [{"item": 0, "pred": "scratch", "accept": True},
           {"item": 1, "pred": "stain", "accept": True}]
    out = pl.to_yolo_lines(recs, dec, class_names=["scratch", "stain"])
    assert set(out) == {"/a.jpg", "/b.jpg"}
    assert len(out["/a.jpg"]) == 1 and len(out["/b.jpg"]) == 1
    assert out["/b.jpg"][0].split()[0] == "1"   # stain=index1


# ── export_prelabels ────────────────────────────────────────────────────────
def test_ac12_export_and_readback(tmp_path):  # AC12:寫檔 + 讀回 + 回傳計數
    pl = _pl()
    od = tmp_path / "out"
    res = pl.export_prelabels({"/x/img_a.jpg": ["0 0.5 0.5 0.4 0.6"]}, od,
                              class_names=["scratch", "stain"], source_dirs=[])
    assert (od / "labels" / "img_a.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.4 0.6\n"
    assert res["written"] == 1 and res["objects"] == 1


def test_ac13_c6_refuse_inside_source(tmp_path):  # AC13:輸出位於來源內 → raise
    pl = _pl()
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(ValueError):
        pl.export_prelabels({"/x/a.jpg": ["0 0.5 0.5 0.4 0.6"]}, src / "nested_out",
                            source_dirs=[src])


def test_ac14_c6_refuse_images_dir(tmp_path):  # AC14:輸出目錄含 images/ → raise
    pl = _pl()
    od = tmp_path / "od2"
    (od / "images").mkdir(parents=True)
    with pytest.raises(ValueError):
        pl.export_prelabels({"/x/a.jpg": ["0 0.5 0.5 0.4 0.6"]}, od, source_dirs=[])


def test_ac15_reexport_allowed(tmp_path):  # AC15:重複匯出(已含 labels/)不 raise、覆寫
    pl = _pl()
    od = tmp_path / "od3"
    pl.export_prelabels({"/x/a.jpg": ["0 0.5 0.5 0.4 0.6"]}, od, source_dirs=[])
    pl.export_prelabels({"/x/a.jpg": ["1 0.5 0.5 0.4 0.6"]}, od, source_dirs=[])
    assert (od / "labels" / "a.txt").read_text(encoding="utf-8") == "1 0.5 0.5 0.4 0.6\n"


def test_ac16_classes_txt_and_empty(tmp_path):  # AC16:classes.txt + 空輸入不 crash
    pl = _pl()
    od = tmp_path / "od4"
    res = pl.export_prelabels({}, od, class_names=["scratch", "stain"], source_dirs=[])
    assert (od / "classes.txt").read_text(encoding="utf-8") == "scratch\nstain\n"
    assert res["written"] == 0


# ── 衍生 / 不變量(設計未明列,PM 自加)────────────────────────────────────────
def test_ac17_source_untouched(tmp_path, head):  # AC17:C6 真實不變量(來源零寫入)
    pl = _pl()
    src = tmp_path / "srcds"
    (src / "labels").mkdir(parents=True)
    (src / "labels" / "keep.txt").write_text("orig\n", encoding="utf-8")
    before = {p.name: p.stat().st_size for p in (src / "labels").iterdir()}
    recs = [{"image_path": "/z/a.jpg", "bbox": (0.5, 0.5, 0.4, 0.6)}]
    emb = np.array([[1.0, 0.0]], dtype=np.float32)
    dec = pl.prelabel_records(recs, emb, [0.9], head, anomaly_threshold=0.5, min_conf=0.0)
    lines = pl.to_yolo_lines(recs, dec, class_names=list(head["classes"]))
    pl.export_prelabels(lines, tmp_path / "out17", class_names=list(head["classes"]),
                        source_dirs=[src])
    after = {p.name: p.stat().st_size for p in (src / "labels").iterdir()}
    assert before == after
    assert (src / "labels" / "keep.txt").read_text(encoding="utf-8") == "orig\n"


def test_ac18_prelabel_idempotent(head):  # AC18:純函式冪等(同輸入兩次逐欄相等)
    pl = _pl()
    recs = [{"image_path": "/a.jpg", "bbox": (0.5, 0.5, 0.4, 0.6)}]
    emb = np.array([[1.0, 0.0]], dtype=np.float32)
    a = pl.prelabel_records(recs, emb, [0.9], head, anomaly_threshold=0.5, min_conf=0.0)
    b = pl.prelabel_records(recs, emb, [0.9], head, anomaly_threshold=0.5, min_conf=0.0)
    assert a == b
