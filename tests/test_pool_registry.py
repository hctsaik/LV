"""驗收:pool_registry(設計 3_Architect_Design/08_pool_registry.md)。

全域樣本池記帳:以 sha256 為身分的 append-only JSONL 狀態機。
先紅策略:目標模組 `pool_registry` 尚未實作 → 在 fixture 內 import(collection 乾淨,
未實作時為『執行期紅』test error,非 collection error)。
隔離:每測 monkeypatch LV_CACHE_DIR 到 tmp_path,不碰真 registry;reload 確保
call-time / import-time 讀 env 都拿到本測目錄(防跨測污染)。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture
def pr(tmp_path, monkeypatch):
    """import 隔離後的 pool_registry;registry 落在本測 tmp_path/.lv_cache 下。"""
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path))
    import importlib
    import pool_registry as _pr
    importlib.reload(_pr)
    return _pr


def _lines(pr):
    """registry 檔的非空行(供 append-only / 手動塞壞行用)。"""
    p = Path(pr.registry_path())
    if not p.exists():
        return []
    return [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_ac1_round_trip(pr):  # AC1 往返:mark_sent 後 load_states 反映 state/batch_id
    pr.mark_sent([SHA_A], "b1", "viz")
    st = pr.load_states()
    assert st[SHA_A]["state"] == "sent"
    assert st[SHA_A]["batch_id"] == "b1"


def test_ac2_priority(pr):  # AC2 優先序:labeled > excluded > sent
    pr.mark_sent([SHA_A], "b1", "viz")
    pr.mark_labeled([SHA_A], "b1")
    assert pr.load_states()[SHA_A]["state"] == "labeled"
    pr.mark_excluded([SHA_A], "gray")
    assert pr.load_states()[SHA_A]["state"] == "labeled"  # excluded 不得降級 labeled


def test_ac3_reset(pr):  # AC3 reset:清回 unseen,之後事件重新累積
    pr.mark_sent([SHA_A], "b1", "viz")
    pr.record_events([{"sha256": SHA_A, "event": "reset"}])
    assert SHA_A not in pr.load_states()  # unseen
    pr.mark_sent([SHA_A], "b2", "viz")
    assert pr.load_states()[SHA_A]["state"] == "sent"


def test_ac4_partition_new_order_preserved(pr):  # AC4 分流:新樣本順序保留 + 已有狀態回報
    pr.mark_sent([SHA_A], "b1", "viz")
    new, have = pr.partition_new([SHA_B, SHA_A, SHA_C])
    assert new == [SHA_B, SHA_C]                 # 順序保留、A 被濾掉
    assert set(have) == {SHA_A}
    assert have[SHA_A]["state"] == "sent"


def test_ac5_corrupt_lines_tolerated(pr):  # AC5 壞行容忍 + corrupt_line_count
    pr.mark_sent([SHA_A], "b1", "viz")
    p = Path(pr.registry_path())
    with p.open("a", encoding="utf-8") as f:
        f.write("not json\n")                                  # 壞行 1:非 JSON
        f.write(json.dumps({"sha256": SHA_B}) + "\n")          # 壞行 2:缺 event 鍵
    st = pr.load_states()
    assert st[SHA_A]["state"] == "sent"                        # 好行照常
    assert all(len(k) == 64 for k in st)                       # 乾淨:鍵皆 sha、無診斷欄位
    assert SHA_B not in st
    assert pr.corrupt_line_count() == 2


def test_ac6_missing_file(pr):  # AC6 檔不存在:load_states=={};partition_new 短路
    assert pr.load_states() == {}
    assert pr.partition_new([SHA_A]) == ([SHA_A], {})


def test_ac7_append_only(pr):  # AC7 append-only:兩次 mark_sent → 檔案兩行
    pr.mark_sent([SHA_A], "b1", "viz")
    pr.mark_sent([SHA_B], "b2", "viz")
    assert len(_lines(pr)) == 2


def test_ac8_no_heavy_dependency():  # AC8 無重依賴:乾淨子行程 import 不拉 numpy
    code = (
        "import sys, pool_registry\n"
        "bad = [m for m in sys.modules if m == 'numpy' or m.startswith('numpy.')]\n"
        "assert not bad, bad\n"
    )
    env = dict(os.environ, PYTHONPATH=str(_SCRIPTS))
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr


# ── 推導測試(設計未明列,PM 自加:優先序單調性不變量)──
def test_acD1_priority_monotonic_no_downgrade(pr):  # AC-D1 低優先事件不得降級高優先狀態(reset 例外)
    """labeled 之後再來 sent/excluded 都不得把狀態降回去 —— 折疊必須取『最高優先』
    而非『最後寫入』(reset 是唯一合法的清除)。這條 metamorphic 不變量能逼出
    『用 last-write-wins 折疊』的實作 bug(該 bug 會讓 AC2 的第二段偶爾漏抓)。"""
    pr.mark_labeled([SHA_A], "b1")
    pr.mark_sent([SHA_A], "b2", "viz")       # 較低優先,晚到
    pr.mark_excluded([SHA_A], "gray")        # 中優先,更晚到
    assert pr.load_states()[SHA_A]["state"] == "labeled"
    # 但 reset 可清除,之後最高優先重新生效
    pr.record_events([{"sha256": SHA_A, "event": "reset"}])
    pr.mark_sent([SHA_A], "b3", "viz")
    assert pr.load_states()[SHA_A]["state"] == "sent"
