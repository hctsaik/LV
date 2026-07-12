"""驗收:round_ledger(設計 3_Architect_Design/09_round_ledger.md)。

回合帳本:append-only JSONL,round 狀態由事件折疊而成。
先紅策略:目標模組 `round_ledger` 尚未實作 → 在 fixture 內 import(collection 乾淨)。
隔離:monkeypatch LV_OUTPUT_DIR 到 tmp_path;reload 確保讀 env 拿到本測目錄。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture
def rl(tmp_path, monkeypatch):
    """import 隔離後的 round_ledger;rounds.jsonl 落在本測 tmp_path 下。"""
    monkeypatch.setenv("LV_OUTPUT_DIR", str(tmp_path))
    import importlib
    import round_ledger as _rl
    importlib.reload(_rl)
    return _rl


def test_ac1_lifecycle_pinned(rl):  # AC1 生命週期:四類欄位值全對(釘死)
    rid = rl.start_round("hybrid:disagreement", {"k": 10}, "viz")
    rl.attach_batch(rid, "b1", 10)
    rl.record_readback(rid, {"batch_id": "b1", "n_labeled": 8, "n_changed": 3,
                             "changed_by_class": {"door": 3}})
    rl.attach_metric(rid, "probe_acc", 0.9, "probe")
    rounds = rl.load_rounds()
    assert len(rounds) == 1
    r = rounds[0]
    assert r["strategy"] == "hybrid:disagreement"
    assert r["n_sent"] == 10
    assert r["n_labeled"] == 8
    assert r["n_changed"] == 3
    assert r["metrics"][-1]["name"] == "probe_acc"
    assert r["metrics"][-1]["value"] == 0.9
    assert r["metrics"][-1]["source"] == "probe"


def test_ac2_accumulate(rl):  # AC2 多批多讀回累加
    rid = rl.start_round("s", {}, "viz")
    rl.attach_batch(rid, "b1", 3)
    rl.attach_batch(rid, "b2", 4)
    rl.record_readback(rid, {"batch_id": "b1", "n_labeled": 3, "n_changed": 0,
                             "changed_by_class": {}})
    rl.record_readback(rid, {"batch_id": "b2", "n_labeled": 4, "n_changed": 1,
                             "changed_by_class": {"x": 1}})
    r = rl.load_rounds()[0]
    assert r["n_sent"] == 7
    assert r["n_labeled"] == 7


def test_ac3_round_id_unique(rl):  # AC3 round_id 格式與唯一:連呼 100 次全異、皆以 r 開頭
    ids = [rl.start_round("s", {}, "viz") for _ in range(100)]
    assert len(set(ids)) == 100
    assert all(i.startswith("r") for i in ids)


def test_ac4_orphan(rl):  # AC4 孤兒:掛不存在 round 的事件進 orphan_events,load_rounds 乾淨
    rid = rl.start_round("s", {}, "viz")
    rl.attach_batch(rid, "b1", 5)
    rl.record_readback("r_does_not_exist", {"batch_id": "bx", "n_labeled": 2,
                                            "n_changed": 0, "changed_by_class": {}})
    assert len(rl.orphan_events()) == 1
    rounds = rl.load_rounds()
    assert len(rounds) == 1                # 真回合不受影響
    assert rounds[0]["n_labeled"] == 0     # 孤兒沒被算進任何回合


def test_ac5_learning_curve_pinned(rl):  # AC5 learning_curve:累積 x + 缺 metric 回合跳過(釘死)
    for n_lab, acc in ((10, 0.5), (20, None), (30, 0.8)):
        rid = rl.start_round("s", {}, "viz")
        rl.attach_batch(rid, f"b{n_lab}", n_lab)
        rl.record_readback(rid, {"batch_id": f"b{n_lab}", "n_labeled": n_lab,
                                 "n_changed": 0, "changed_by_class": {}})
        if acc is not None:
            rl.attach_metric(rid, "acc", acc, "manual")
    xs, ys = rl.learning_curve(rl.load_rounds(), "acc")
    assert xs == [10, 60]        # 第 3 回合累積含被跳過的第 2 回合(10+20+30)
    assert ys == [0.5, 0.8]


def test_ac6_corrupt_tolerated(rl):  # AC6 壞行容忍:load_rounds 照常
    rid = rl.start_round("s", {}, "viz")
    rl.attach_batch(rid, "b1", 5)
    p = Path(rl.ledger_path())
    with p.open("a", encoding="utf-8") as f:
        f.write("not json at all\n")
        f.write(json.dumps({"kind": "batch"}) + "\n")   # 缺 round_id 的壞事件
    rounds = rl.load_rounds()
    assert len(rounds) == 1
    assert rounds[0]["n_sent"] == 5


def test_ac7_no_heavy_dependency():  # AC7 無重依賴:乾淨子行程 import 不拉 numpy
    code = (
        "import sys, round_ledger\n"
        "bad = [m for m in sys.modules if m == 'numpy' or m.startswith('numpy.')]\n"
        "assert not bad, bad\n"
    )
    env = dict(os.environ, PYTHONPATH=str(_SCRIPTS))
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr


# ── 推導測試(PM 自加)──
def test_acD1_metric_last_wins_and_x_monotonic(rl):  # AC-D1 同名 metric 取最後一筆 + xs 單調不減
    """(1) 同回合對同名 metric 掛兩次 → 曲線取最後一筆(不是第一筆、不是平均);
    (2) 曲線 x(累積標註量)必單調不減。這條逼出『metric 取 first / 未累積 x』的 bug。"""
    r1 = rl.start_round("s", {}, "viz")
    rl.record_readback(r1, {"batch_id": "b1", "n_labeled": 10, "n_changed": 0,
                            "changed_by_class": {}})
    rl.attach_metric(r1, "acc", 0.5, "manual")
    rl.attach_metric(r1, "acc", 0.7, "manual")   # 較晚 → 應勝出
    r2 = rl.start_round("s", {}, "viz")
    rl.record_readback(r2, {"batch_id": "b2", "n_labeled": 5, "n_changed": 0,
                            "changed_by_class": {}})
    rl.attach_metric(r2, "acc", 0.9, "manual")
    xs, ys = rl.learning_curve(rl.load_rounds(), "acc")
    assert ys == [0.7, 0.9]                       # last-wins,非 0.5
    assert all(b >= a for a, b in zip(xs, xs[1:]))  # xs 單調不減
    assert xs == [10, 15]
