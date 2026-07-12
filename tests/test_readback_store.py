"""單元驗收:14 readback_store(讀回落地層——覆蓋層 + append-only 變更日誌)。

契約見 3_Architect_Design/14_readback_store.md。純函式,不需影像檔(sha 用假值)。
每個測試註解標對應 AC。tmp_path + monkeypatch LV_CACHE_DIR,不碰真 .lv_cache。
"""
from __future__ import annotations

import subprocess
import sys

import pytest

import readback_store as RS

SHA_A = "a" * 64
SHA_B = "b" * 64


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_ac1_record_then_overlay():
    # AC1: 記入一筆 → 覆蓋層反映 new_label,回傳筆數 1
    n = RS.record_corrections([{"sha256": SHA_A, "old_label": "x",
                                "new_label": "y", "filename": "f.jpg"}])
    assert n == 1
    assert RS.load_overlay()[SHA_A] == "y"


def test_ac2_latest_wins_history_kept():
    # AC2: 同 sha 兩次修正 → 覆蓋取最新;變更日誌保留兩筆(append-only)
    RS.record_corrections([{"sha256": SHA_A, "new_label": "y"}])
    RS.record_corrections([{"sha256": SHA_A, "new_label": "z"}])
    assert RS.load_overlay()[SHA_A] == "z"
    assert len(RS.load_changelog()) == 2


def test_ac3_apply_overlay():
    # AC3: apply_overlay 覆蓋命中的 record.label,未命中不動
    RS.record_corrections([{"sha256": SHA_A, "new_label": "z"}])
    records = [{"path": "/p/f.jpg", "label": "x"}, {"path": "/p/g.jpg", "label": "w"}]
    n = RS.apply_overlay(records, sha_of=lambda r: {"/p/f.jpg": SHA_A}.get(r["path"]))
    assert n == 1
    assert records[0]["label"] == "z"
    assert records[1]["label"] == "w"


def test_ac4_append_only_audit_fields():
    # AC4: append-only,行數 == Σ筆數,每行含稽核欄位
    RS.record_corrections([{"sha256": SHA_A, "old_label": "x", "new_label": "y"}],
                          annotator="alice", batch_id="b1", round_id="r1")
    RS.record_corrections([{"sha256": SHA_B, "old_label": "p", "new_label": "q"}],
                          annotator="bob", batch_id="b2", round_id="r2")
    log = RS.load_changelog()
    assert len(log) == 2
    for e in log:
        for k in ("ts", "annotator", "batch_id", "round_id", "old_label", "new_label"):
            assert k in e
    assert log[0]["annotator"] == "alice" and log[0]["batch_id"] == "b1"


def test_ac5_corrupt_lines_tolerated():
    # AC5: 壞行(非 JSON / 缺 sha256)不崩潰,其餘照回
    RS.record_corrections([{"sha256": SHA_A, "new_label": "y"}])
    p = RS.changelog_path()
    with p.open("a", encoding="utf-8") as f:
        f.write("not json\n")
        f.write('{"new_label": "z"}\n')       # 缺 sha256
    assert RS.load_overlay() == {SHA_A: "y"}
    assert len(RS.load_changelog()) == 1


def test_ac6_missing_file():
    # AC6: 全新目錄 → 空覆蓋、apply 0、summary n_events 0
    assert RS.load_overlay() == {}
    assert RS.apply_overlay([{"path": "x", "label": "l"}], sha_of=lambda r: SHA_A) == 0
    assert RS.corrections_summary()["n_events"] == 0


def test_ac7_incomplete_skipped():
    # AC7: 缺 new_label 或缺 sha256 → 不寫
    assert RS.record_corrections([{"sha256": SHA_A}]) == 0
    assert RS.record_corrections([{"new_label": "y"}]) == 0
    assert RS.load_overlay() == {}


def test_ac8_noop_not_counted():
    # AC8: 現 label 已等於覆蓋值 → apply_overlay 不計入(不假覆蓋)
    RS.record_corrections([{"sha256": SHA_A, "new_label": "z"}])
    assert RS.apply_overlay([{"path": "p", "label": "z"}], sha_of=lambda r: SHA_A) == 0


def test_ac9_export_version(tmp_path):
    # AC9: export_version 產出 CSV,含表頭 + 資料列
    RS.record_corrections([{"sha256": SHA_A, "old_label": "x", "new_label": "y",
                            "filename": "f.jpg", "annotator": "alice"}])
    out = RS.export_version(tmp_path, "v1")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines[0].startswith("sha256")          # header
    assert len(lines) == 2
    assert SHA_A in lines[1] and "f.jpg" in lines[1] and "y" in lines[1]


def test_ac10_no_heavy_dependency():
    # AC10: 乾淨子行程 import 不拉 numpy
    code = "import readback_store, sys; assert 'numpy' not in sys.modules; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(__import__("pathlib").Path(RS.__file__).parent))
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_ac11_filter_shas():
    # AC11: load_overlay(shas) 只回命中的
    RS.record_corrections([{"sha256": SHA_A, "new_label": "y"},
                           {"sha256": SHA_B, "new_label": "q"}])
    assert RS.load_overlay([SHA_A]) == {SHA_A: "y"}
