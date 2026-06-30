"""驗收:whole_image_source(設計 3_Architect_Design/07_whole_image_source.md)。
先紅:`discover_whole_images` 未實作、`run_pipeline` 未加 `object_source`。
- 新函式 import 放在「測試函式內」→ 未實作時是『執行期紅』(test fail),不是 collection error。
- run_pipeline 走 score_mode="object"(不需 extractor),用注入式 stub embed_fn → 不需真模型。
約定:記錄欄位與 discover_yolo_objects 同形;run_pipeline records 依 meta 順序。"""
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from anomaly_tool import run_pipeline
from interaction import crop_bbox, discover_yolo_objects

# discover_yolo_objects 的 meta 記錄欄位(whole_image 必須完全同形,防下游漂移)
YOLO_KEYS = {"image_path", "label", "class_id", "bbox", "obj_index", "score"}


def _make_images(folder: Path, n: int):
    """造 n 張可讀影像(**無 labels/**),回 Path 串。"""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(n):
        p = folder / f"img_{k}.png"
        Image.new("RGB", (32, 32), (10 + 7 * k, 180, 60)).save(p)
        paths.append(p)
    return paths


def _seq_embed():
    """注入式 embed_fn(crop)->(384,):每物件回相異單位向量(確定性、彼此可分),
    讓 run_pipeline 物件級分群/評分非退化。同一序呼叫 → 同一序輸出(供回歸逐筆比對)。"""
    c = {"i": 0}

    def f(crop):
        v = np.zeros(384, dtype=np.float32)
        v[c["i"] % 384] = 1.0
        v[(c["i"] + 7) % 384] = 0.3
        c["i"] += 1
        return v
    return f


# ── 單元 AC ───────────────────────────────────────────────────────────────
def test_record_content_exact():  # AC1
    from interaction import discover_whole_images
    out = discover_whole_images([Path("a.jpg"), Path("b.png"), Path("c.jpg")])
    assert out[1] == {"image_path": Path("b.png"), "label": "image", "class_id": 0,
                      "bbox": (0.5, 0.5, 1.0, 1.0), "obj_index": 0, "score": 1.0}


def test_keys_are_yolo_shape():  # AC2
    from interaction import discover_whole_images
    for rec in discover_whole_images([Path("a.jpg"), Path("b.jpg")]):
        assert set(rec.keys()) == YOLO_KEYS


def test_count_and_order_preserved():  # AC3
    from interaction import discover_whole_images
    ins = [Path(f"x{i}.jpg") for i in range(5)]
    out = discover_whole_images(ins)
    assert len(out) == 5
    assert [r["image_path"] for r in out] == ins


def test_label_param():  # AC4
    from interaction import discover_whole_images
    assert discover_whole_images([Path("p.jpg")], label="frame")[0]["label"] == "frame"
    assert discover_whole_images([Path("p.jpg")])[0]["label"] == "image"


def test_empty_input():  # AC5
    from interaction import discover_whole_images
    assert discover_whole_images([]) == []


def test_run_pipeline_whole_image_no_labels(tmp_path):  # AC6
    paths = _make_images(tmp_path / "imgs", 4)
    r_whole = run_pipeline(paths, None, mode="one_stage", score_mode="object",
                           object_source="whole_image", embed_fn=_seq_embed())
    assert len(r_whole["records"]) == 4              # 無 labels 仍每張一筆
    r_yolo = run_pipeline(paths, None, mode="one_stage", score_mode="object",
                          object_source="yolo", embed_fn=_seq_embed())
    assert r_yolo["records"] == []                   # 對照:yolo 無 labels → 空(差異來自路由)


def test_run_pipeline_default_is_yolo_regression(synthetic_yolo_dataset):  # AC7
    from object_eval import list_images, classes_for
    root = synthetic_yolo_dataset["root"]
    ip, cn = list_images(root), classes_for(root)
    K = len(discover_yolo_objects(ip, cn))
    r_default = run_pipeline(ip, cn, mode="one_stage", score_mode="object",
                             embed_fn=_seq_embed())
    r_yolo = run_pipeline(ip, cn, mode="one_stage", score_mode="object",
                          object_source="yolo", embed_fn=_seq_embed())
    assert len(r_default["records"]) == K            # 預設未傳 object_source = yolo
    proj = lambda rs: [(x["image_path"], x["obj_index"], x["label"]) for x in rs]
    assert proj(r_default["records"]) == proj(r_yolo["records"])


def test_invalid_object_source_raises(tmp_path):  # AC8
    paths = _make_images(tmp_path / "imgs", 2)
    with pytest.raises(ValueError):
        run_pipeline(paths, None, mode="one_stage", score_mode="object",
                     object_source="bogus", embed_fn=_seq_embed())


def test_full_frame_bbox_is_whole_image():  # AC10
    img = Image.new("RGB", (64, 48))
    img.putdata([(i % 256, (2 * i) % 256, (3 * i) % 256) for i in range(64 * 48)])
    for pad in (0.0, 0.1):
        crop = crop_bbox(img, 0.5, 0.5, 1.0, 1.0, pad=pad)
        assert crop.size == (64, 48)                 # 全幅 bbox 經 clamp = 整張影像
        assert list(crop.getdata()) == list(img.getdata())


# ── 推導測試(設計未明列,PM 自加施壓:防下游漂移的不變量)──────────────────
def test_derived_keys_match_real_yolo_record(synthetic_yolo_dataset):  # 推導:與真實 yolo 記錄同形
    """metamorphic:whole_image 記錄欄位 == discover_yolo_objects 真實記錄欄位。
    這是下游(embedding/分群/bank/評分/散點/匯出)不需改動的契約;一旦漂移即破。"""
    from interaction import discover_whole_images
    from object_eval import list_images, classes_for
    root = synthetic_yolo_dataset["root"]
    ip, cn = list_images(root), classes_for(root)
    yolo_keys = set(discover_yolo_objects(ip, cn)[0].keys())
    whole_keys = set(discover_whole_images(ip[:1])[0].keys())
    assert whole_keys == yolo_keys


def test_derived_all_records_full_frame_invariant():  # 推導:每筆皆全幅、obj_index=0 等不變量
    from interaction import discover_whole_images
    out = discover_whole_images([Path(f"z{i}.jpg") for i in range(7)])
    assert len(out) == 7
    assert all(r["bbox"] == (0.5, 0.5, 1.0, 1.0) and r["obj_index"] == 0
               and r["class_id"] == 0 and r["score"] == 1.0 for r in out)
