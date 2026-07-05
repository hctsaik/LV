"""驗收:13c retrieval_export(以樣搜樣確認 → YOLO 預標沿用粗框幾何 + retrieval_report.csv)。
設計:3_Architect_Design/13c_retrieval_export.md。復用 prelabel 匯出。延遲匯入。"""
import csv
import io
import pytest
from pathlib import Path
from PIL import Image


def _re():
    import retrieval_export
    return retrieval_export


def _real_records(tmp_path):
    """造真實影像檔 + 指向它們的 records(copy_images 需要能真的複製)。"""
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (16, 16), (200, 0, 0)).save(src / "img_a.jpg")
    Image.new("RGB", (16, 16), (0, 0, 200)).save(src / "img_b.jpg")
    recs = [
        {"image_path": str(src / "img_a.jpg"), "obj_index": 0, "bbox": [0.5, 0.5, 0.4, 0.6],
         "score": 0.9, "suggested_class": "cat", "similarity": 0.8},
        {"image_path": str(src / "img_b.jpg"), "obj_index": 0, "bbox": [0.25, 0.75, 0.2, 0.2],
         "score": None, "suggested_class": "dog", "similarity": 0.3},
    ]
    return src, recs


def _records():
    return [
        {"image_path": "/d/img_a.jpg", "obj_index": 0, "bbox": [0.5, 0.5, 0.4, 0.6],
         "score": 0.9, "suggested_class": "cat", "similarity": 0.8},
        {"image_path": "/d/img_b.jpg", "obj_index": 0, "bbox": [0.25, 0.75, 0.2, 0.2],
         "score": None, "suggested_class": "dog", "similarity": 0.3},
    ]


def test_ac_ce1_csv_header_values():  # AC-CE1
    re = _re()
    decs = [{"item": 0, "decision": "accepted", "final_class": "cat"},
            {"item": 1, "decision": "skipped", "final_class": None}]
    rows = list(csv.reader(io.StringIO(re.retrieval_report_csv(_records(), decs))))
    assert rows[0] == ["image_path", "obj_index", "cx", "cy", "w", "h", "proposal_conf",
                       "suggested_class", "similarity", "decision", "final_class"]
    assert rows[1][9] == "accepted" and rows[1][10] == "cat" and rows[1][6] == "0.9"
    assert rows[2][9] == "skipped" and rows[2][10] == "" and rows[2][6] == ""


def test_ac_ce2_yolo_accept_only_proposal_geom(tmp_path):  # AC-CE2
    re = _re()
    decs = [{"item": 0, "decision": "accepted", "final_class": "cat"},
            {"item": 1, "decision": "skipped", "final_class": None}]
    re.export_retrieval(_records(), decs, tmp_path / "out", class_names=["cat", "dog"], source_dirs=[])
    assert (tmp_path / "out" / "labels" / "img_a.txt").read_text(encoding="utf-8").strip() \
        == "0 0.500000 0.500000 0.400000 0.600000"
    assert not (tmp_path / "out" / "labels" / "img_b.txt").exists()


def test_ac_ce3_relabel(tmp_path):  # AC-CE3
    re = _re()
    decs = [{"item": 0, "decision": "relabeled", "final_class": "dog"}]
    re.export_retrieval([_records()[0]], decs, tmp_path / "out",
                        class_names=["cat", "dog"], source_dirs=[])
    assert (tmp_path / "out" / "labels" / "img_a.txt").read_text(encoding="utf-8").split()[0] == "1"


def test_ac_ce4_dual_export_readback(tmp_path):  # AC-CE4
    re = _re()
    decs = [{"item": 0, "decision": "accepted", "final_class": "cat"},
            {"item": 1, "decision": "skipped", "final_class": None}]
    res = re.export_retrieval(_records(), decs, tmp_path / "out",
                              class_names=["cat", "dog"], source_dirs=[])
    assert (tmp_path / "out" / "labels" / "img_a.txt").exists()
    assert (tmp_path / "out" / "retrieval_report.csv").exists()
    assert (tmp_path / "out" / "classes.txt").exists()
    assert res["written"] == 1 and res["objects"] == 1 and res["csv_rows"] == 2
    rows = list(csv.reader((tmp_path / "out" / "retrieval_report.csv").read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 3


def test_ac_ce5_c6(tmp_path):  # AC-CE5:輸出在來源內 → raise
    re = _re()
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(ValueError):
        re.export_retrieval([_records()[0]], [{"item": 0, "decision": "accepted", "final_class": "cat"}],
                            src / "out", class_names=["cat", "dog"], source_dirs=[src])


def test_ac_ce6_final_class_unknown(tmp_path):  # AC-CE6
    re = _re()
    with pytest.raises(ValueError, match="bird"):
        re.export_retrieval([_records()[0]], [{"item": 0, "decision": "relabeled", "final_class": "bird"}],
                            tmp_path / "out", class_names=["cat", "dog"], source_dirs=[])


# ── M14c:copy_images(匯出成 standalone YOLO 資料集 images/+labels/)──────────
def test_ac_rimg1_copy_images(tmp_path):  # AC-RIMG-1:copy_images=True → images/ 含有標影像 + images_copied
    re = _re()
    src, recs = _real_records(tmp_path)
    decs = [{"item": 0, "decision": "accepted", "final_class": "cat"},
            {"item": 1, "decision": "accepted", "final_class": "dog"}]
    res = re.export_retrieval(recs, decs, tmp_path / "out", class_names=["cat", "dog"],
                              source_dirs=[src], copy_images=True)
    assert (tmp_path / "out" / "images" / "img_a.jpg").exists()
    assert (tmp_path / "out" / "images" / "img_b.jpg").exists()
    assert res["images_copied"] == 2
    assert (tmp_path / "out" / "labels" / "img_a.txt").exists()   # 與 labels 對應


def test_ac_rimg2_c6_and_reexport(tmp_path):  # AC-RIMG-2:來源零寫入 + 同 out_dir 再匯出不 raise
    re = _re()
    src, recs = _real_records(tmp_path)
    before = {p.name: p.stat().st_size for p in src.iterdir()}
    decs = [{"item": 0, "decision": "accepted", "final_class": "cat"}]
    re.export_retrieval([recs[0]], decs, tmp_path / "out", class_names=["cat", "dog"],
                        source_dirs=[src], copy_images=True)
    assert {p.name: p.stat().st_size for p in src.iterdir()} == before, "來源零寫入(C6)"
    re.export_retrieval([recs[0]], decs, tmp_path / "out", class_names=["cat", "dog"],
                        source_dirs=[src], copy_images=True)   # images/ 已存在 → 不 raise(allow_images)


def test_ac_rimg3_default_no_images(tmp_path):  # AC-RIMG-3 不變量:預設無 images/、images_copied==0
    re = _re()
    src, recs = _real_records(tmp_path)
    res = re.export_retrieval([recs[0]], [{"item": 0, "decision": "accepted", "final_class": "cat"}],
                              tmp_path / "out", class_names=["cat", "dog"], source_dirs=[src])
    assert not (tmp_path / "out" / "images").exists()
    assert res.get("images_copied", 0) == 0


def test_ac_rimg4_only_accepted_images(tmp_path):  # AC-RIMG-4:只複製 accepted/relabeled 的影像
    re = _re()
    src, recs = _real_records(tmp_path)
    decs = [{"item": 0, "decision": "accepted", "final_class": "cat"},
            {"item": 1, "decision": "skipped", "final_class": None}]
    re.export_retrieval(recs, decs, tmp_path / "out", class_names=["cat", "dog"],
                        source_dirs=[src], copy_images=True)
    assert (tmp_path / "out" / "images" / "img_a.jpg").exists()
    assert not (tmp_path / "out" / "images" / "img_b.jpg").exists()   # img_b 全 skip → 不複製
