"""驗收:al_workspace(設計 3_Architect_Design/09_al_workspace.md)。

先紅:`scripts/al_workspace.py` 尚未實作(import 放函式內 → 執行期紅,非 collection error)。
全用 tmp 目錄造 workspace + 假來源資料夾;不觸真模型/真掃描。

契約鎖點:增量分類正確+冪等(AC3)、單寫者不互污(AC-D1)、id=al_batch item_id 原地對應、原子性(AC10)。
"""
import json
import os
import time
import numpy as np  # noqa: F401  (與其他 anomaly 測試一致的 sys.path 觸發)
import pytest
from pathlib import Path


def _wsmod():
    import al_workspace
    return al_workspace


def _touch(p: Path, content: str = "x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _src(tmp_path):
    """造一個假來源資料夾,3 張『圖』(內容檔即可,scan 只看 mtime/size)。"""
    root = tmp_path / "src"
    for i in range(3):
        _touch(root / f"img_{i}.jpg", f"content-{i}")
    return root


# ══════════════════════════════════════════════════════════════════════
# 設定(profile)
# ══════════════════════════════════════════════════════════════════════

def test_ac1_profile_roundtrip_and_default_k(tmp_path):
    # AC1:profile round-trip;default_profile k==100
    ws = _wsmod()
    prof = ws.default_profile(name="line3", watch_folders=[str(tmp_path / "src")],
                              model_dir=str(tmp_path / "mdl"))
    assert prof["k"] == 100
    ws.save_profile(tmp_path / "wsdir", prof)
    got = ws.load_profile(tmp_path / "wsdir")
    assert got == prof, "profile 存讀必須逐鍵相等(含補齊的 schema_version/預設)"


def test_ac2_missing_or_bad_profile_raises(tmp_path):
    # AC2:缺 profile / 缺 model_dir / schema_version 未知 → raise
    ws = _wsmod()
    with pytest.raises((FileNotFoundError, ValueError)):
        ws.load_profile(tmp_path / "empty_ws")     # 無 profile.yaml
    import yaml
    d = tmp_path / "ws2"; d.mkdir()
    (d / "profile.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "name": "x", "watch_folders": ["a"]}),
        encoding="utf-8")   # 缺 model_dir
    with pytest.raises(ValueError):
        ws.load_profile(d)
    (d / "profile.yaml").write_text(
        yaml.safe_dump({"schema_version": 999, "name": "x",
                        "watch_folders": ["a"], "model_dir": "m"}), encoding="utf-8")
    with pytest.raises(ValueError) as e:
        ws.load_profile(d)
    assert "schema" in str(e.value).lower()


def test_ac_ws_dec_retrieve_model_dir_optional(tmp_path):
    # AC-WS-DEC(M14):objective="retrieve" 時 model_dir 可空(特徵器身分改由 sample_bank 提供),不 raise
    ws = _wsmod()
    import yaml
    d = tmp_path / "wsr"; d.mkdir()
    (d / "profile.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "name": "x", "watch_folders": ["a"],
                        "model_dir": "", "objective": "retrieve",
                        "sample_bank_dir": str(tmp_path / "sbd")}),
        encoding="utf-8")
    prof = ws.load_profile(d)   # 不因 model_dir 空而 raise
    assert prof["objective"] == "retrieve" and prof.get("sample_bank_dir")
    # 對照:非 retrieve(novelty)+ 空 model_dir 仍必填 → raise
    (d / "profile.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "name": "x", "watch_folders": ["a"],
                        "model_dir": "", "objective": "novelty"}),
        encoding="utf-8")
    with pytest.raises(ValueError):
        ws.load_profile(d)


# ══════════════════════════════════════════════════════════════════════
# 增量掃描
# ══════════════════════════════════════════════════════════════════════

def test_ac3_scan_incremental_classes_and_idempotent(tmp_path):
    # AC3:new/changed/deleted 正確且僅含該檔;第二次掃(更新後 state)全空(冪等)
    ws = _wsmod()
    src = _src(tmp_path)
    r1 = ws.scan_incremental([src], {})              # 首掃:3 個都是 new
    assert sorted(Path(p).name for p in r1["new"]) == ["img_0.jpg", "img_1.jpg", "img_2.jpg"]
    assert r1["changed"] == [] and r1["deleted"] == []
    state = r1["state"]
    # 加 1 檔(new)、改 1 檔 mtime(changed)、刪 1 檔(deleted)
    _touch(src / "img_new.jpg", "new")
    _p1 = src / "img_1.jpg"
    os.utime(_p1, (time.time() + 100, time.time() + 100))  # 改 mtime
    (src / "img_2.jpg").unlink()
    r2 = ws.scan_incremental([src], state)
    assert [Path(p).name for p in r2["new"]] == ["img_new.jpg"]
    assert [Path(p).name for p in r2["changed"]] == ["img_1.jpg"]
    assert [Path(p).name for p in r2["deleted"]] == ["img_2.jpg"] or \
           any("img_2.jpg" in str(d) for d in r2["deleted"])
    # 冪等:對更新後 state 再掃 → 全空
    r3 = ws.scan_incremental([src], r2["state"])
    assert r3["new"] == [] and r3["changed"] == [] and r3["deleted"] == []


def test_ac4_size_change_is_changed(tmp_path):
    # AC4:mtime 不變、size 變 → changed(雙鍵)
    ws = _wsmod()
    src = tmp_path / "src4"
    p = _touch(src / "a.jpg", "short")
    state = ws.scan_incremental([src], {})["state"]
    mt = p.stat().st_mtime
    p.write_text("a much longer content changing size", encoding="utf-8")
    os.utime(p, (mt, mt))   # 還原 mtime,只有 size 變
    r = ws.scan_incremental([src], state)
    assert [Path(x).name for x in r["changed"]] == ["a.jpg"], "size 變(mtime 同)須歸 changed"


def test_ac5_scan_never_writes_source(tmp_path):
    # AC5:scan 只讀來源,來源檔案清單+mtime 快照零變化(C6)
    ws = _wsmod()
    src = _src(tmp_path)
    snap = {p: p.stat().st_mtime_ns for p in src.rglob("*")}
    ws.scan_incremental([src], {})
    after = {p: p.stat().st_mtime_ns for p in src.rglob("*")}
    assert after == snap, "scan 不可寫/動來源資料夾"


# ══════════════════════════════════════════════════════════════════════
# 佇列 / 標註 / 合併
# ══════════════════════════════════════════════════════════════════════

def test_ac6_queue_roundtrip(tmp_path):
    # AC6:queue 原子 round-trip,逐筆相等順序保留
    ws = _wsmod()
    items = [{"id": f"h{i}", "image_path": f"/x/{i}.jpg", "score": i / 10, "status": "pending"}
             for i in range(4)]
    ws.write_queue(tmp_path / "q", items)
    assert ws.read_queue(tmp_path / "q") == items


def test_ac7_queue_bad_line_tolerated(tmp_path):
    # AC7:queue.jsonl 插一行非法 json → read_queue 回其餘完好,不 raise
    ws = _wsmod()
    d = tmp_path / "q7"
    ws.write_queue(d, [{"id": "a"}, {"id": "b"}])
    qf = d / "queue.jsonl"
    qf.write_text(qf.read_text(encoding="utf-8") + "{壞行不是json\n", encoding="utf-8")
    got = ws.read_queue(d)
    assert [r["id"] for r in got] == ["a", "b"], "壞行應跳過、不整份失敗"


def test_ac8_labels_append_and_last_wins(tmp_path):
    # AC8:append_label ×3(兩筆同 id 不同 decision)→ 3 行;read_labels 對該 id 取最後一筆
    ws = _wsmod()
    d = tmp_path / "l8"
    ws.append_label(d, {"id": "x", "decision": "good"})
    ws.append_label(d, {"id": "y", "decision": "defect"})
    ws.append_label(d, {"id": "x", "decision": "defect"})   # 覆蓋 x
    assert (d / "labels.jsonl").read_text(encoding="utf-8").count("\n") == 3
    labs = ws.read_labels(d)
    byid = {l["id"]: l["decision"] for l in labs}
    assert byid == {"x": "defect", "y": "defect"}, "同 id 取最後一筆(後覆前)"


def test_ac9_merge_labels_into_pending(tmp_path):
    # AC9:queue 5 筆,標其中 2(good/defect)+ deleted 1 → 剩 2 pending;skip 留 pending
    ws = _wsmod()
    queue = [{"id": f"q{i}", "status": "pending"} for i in range(5)]
    labels = [{"id": "q0", "decision": "good"}, {"id": "q1", "decision": "defect"},
              {"id": "q2", "decision": "skip"}]
    deleted = {"q3"}
    pending = ws.merge_labels_into_pending(queue, labels, deleted_ids=deleted)
    ids = sorted(r["id"] for r in pending)
    assert ids == ["q2", "q4"], f"移出已標(q0,q1)+deleted(q3);skip(q2)留下:{ids}"


# ══════════════════════════════════════════════════════════════════════
# 原子性 / 鎖 / 安全
# ══════════════════════════════════════════════════════════════════════

def test_ac10_write_queue_atomic_on_failure(tmp_path, monkeypatch):
    # AC10:write_queue 在 rename 前失敗 → queue.jsonl 維持舊內容(未半寫)
    ws = _wsmod()
    d = tmp_path / "q10"
    ws.write_queue(d, [{"id": "old1"}, {"id": "old2"}])
    orig = (d / "queue.jsonl").read_text(encoding="utf-8")

    def _boom(*a, **k):
        raise OSError("simulated rename failure")
    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        ws.write_queue(d, [{"id": "new1"}])
    monkeypatch.undo()
    assert (d / "queue.jsonl").read_text(encoding="utf-8") == orig, "原子寫失敗必須保留舊檔"


def test_ac11_assert_safe_workspace(tmp_path, monkeypatch):
    # AC11:workspace 落在 watch_folder 內 → raise;.lv_cache 下通過
    ws = _wsmod()
    watch = tmp_path / "watch"; watch.mkdir()
    with pytest.raises(ValueError):
        ws.assert_safe_workspace(watch / "wsdir", watch_folders=[watch])
    # .lv_cache 下:設 LV_CACHE_DIR 到 tmp,workspace 放其下 → 通過
    cache = tmp_path / "lvcache"; cache.mkdir()
    monkeypatch.setenv("LV_CACHE_DIR", str(cache))
    ws.assert_safe_workspace(cache / "al_workspace" / "p1", watch_folders=[watch])  # 不 raise


def test_ac12_lock_mutex_and_stale_steal(tmp_path):
    # AC12:held+fresh → False;time 過期 → 可搶佔
    ws = _wsmod()
    d = tmp_path / "lk"
    assert ws.acquire_lock(d, pid=111, started=time.time(), stale_after=3600.0) is True
    assert ws.acquire_lock(d, pid=222, started=time.time(), stale_after=3600.0) is False, \
        "已被持有且未過期 → 第二實例不得取得"
    # 把 lock 的 started 設成很久以前(過期)→ 可搶
    import json as _j
    lk = d / "lock"
    info = _j.loads(lk.read_text(encoding="utf-8"))
    info["started"] = time.time() - 7200.0
    lk.write_text(_j.dumps(info), encoding="utf-8")
    assert ws.acquire_lock(d, pid=222, started=time.time(), stale_after=3600.0) is True, \
        "過期鎖應可搶佔"


def test_ac13_release_only_own_pid(tmp_path):
    # AC13:release 只移本 PID 的鎖(非本 PID 不動)
    ws = _wsmod()
    d = tmp_path / "lk13"
    ws.acquire_lock(d, pid=111, started=time.time())
    ws.release_lock(d, pid=222)     # 別人的 → 不移
    assert (d / "lock").exists(), "非本 PID 不得移除他人鎖"
    ws.release_lock(d, pid=111)     # 自己的 → 移
    assert not (d / "lock").exists()


# ══════════════════════════════════════════════════════════════════════
# 推導(PM 自加):單寫者不變式
# ══════════════════════════════════════════════════════════════════════

def test_derived_single_writer_invariant(tmp_path):
    # AC-D1:交錯「服務 write_queue/save_state」與「GUI append_label」後,
    #        labels 只被 append(行數單調增、既有行不變);queue 只被服務重寫;互不污染。
    ws = _wsmod()
    d = tmp_path / "sw"
    ws.write_queue(d, [{"id": "a", "status": "pending"}, {"id": "b", "status": "pending"}])
    ws.append_label(d, {"id": "a", "decision": "good"})
    labels_after_1 = (d / "labels.jsonl").read_text(encoding="utf-8")
    ws.save_state(d, {"schema_version": 1, "files": {}})
    ws.write_queue(d, [{"id": "b", "status": "pending"}])      # 服務重寫 queue
    ws.append_label(d, {"id": "b", "decision": "defect"})       # GUI append
    labels_after_2 = (d / "labels.jsonl").read_text(encoding="utf-8")
    # labels 單調增 + 既有行不變(前綴保留)
    assert labels_after_2.startswith(labels_after_1)
    assert labels_after_2.count("\n") == 2
    # queue 只含服務最後寫的內容(GUI 沒碰 queue)
    assert [r["id"] for r in ws.read_queue(d)] == ["b"]
