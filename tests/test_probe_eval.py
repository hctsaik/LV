"""驗收:probe_eval(設計 3_Architect_Design/12_probe_eval.md · § 5 單元 AC)。

linear probe 回合代理評估(train_probe)+ 報酬遞減判讀(plateau_advice)。
本檔只寫「單元」AC(AC1–AC6);GUI 的 AC-G1..G3 屬 tests/e2e/test_probe_eval_e2e.py,不在此檔。

先紅策略:目標模組 `probe_eval` 尚未實作 → 一律在 `pe` fixture / 函式內 import
(collection 乾淨;未實作時為『執行期紅』test error,而非 collection error)。
top-level 只 import stdlib / numpy / pytest。數值一律照設計釘死,不從實作反推。
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture
def pe():
    """import 隔離後的 probe_eval(未實作 → 此處 ModuleNotFoundError,測試為 error 紅)。"""
    import probe_eval as _pe
    return _pe


# ── 合成資料 helper(僅用 numpy;決定論)────────────────────────────────
def _two_class(n_a, n_b, *, loc=10.0, scale=1.0, d=8, seed=0):
    """兩類高斯:A 在 +loc、B 在 -loc(loc 大 → 線性可分)。回 (emb, labels)。"""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=+loc, scale=scale, size=(n_a, d))
    b = rng.normal(loc=-loc, scale=scale, size=(n_b, d))
    emb = np.vstack([a, b]).astype(np.float64)
    labels = ["A"] * n_a + ["B"] * n_b
    return emb, labels


def test_ac1_separable_acc_perfect(pe):  # AC1 可分資料 → acc==1.0 / macro_f1==1.0 / n 切分
    emb, labels = _two_class(20, 20, loc=10.0, seed=1)   # 兩類各 20 點、x 軸 ±10 高斯
    r = pe.train_probe(emb, labels)
    assert r["acc"] == 1.0
    assert r["macro_f1"] == 1.0
    assert r["n_train"] + r["n_test"] == 40
    assert r["n_test"] == 10                              # test_size=0.25 → 40*0.25


def test_ac2_stratified_test_ratio(pe):  # AC2 分層:3:1 的 80 點 → test 集內比例仍 3:1(±1 張)
    emb, labels = _two_class(60, 20, loc=10.0, seed=2)   # A:B = 60:20 = 3:1
    r = pe.train_probe(emb, labels)
    n_a = r["per_class"]["A"]["n"]                        # per_class 在 test 上算 → n = test 支持數
    n_b = r["per_class"]["B"]["n"]
    assert r["n_train"] + r["n_test"] == 80
    assert r["n_test"] == 20                              # 80*0.25
    assert n_a + n_b == 20
    assert 14 <= n_a <= 16                                # 理想 15,±1 張
    assert 4 <= n_b <= 6                                  # 理想 5,±1 張


def test_ac3_deterministic(pe):  # AC3 決定論:同輸入重呼 → dict 逐鍵相等
    emb, labels = _two_class(20, 20, loc=3.0, seed=3)     # 稍重疊也無妨,只驗決定論
    r1 = pe.train_probe(emb, labels)
    r2 = pe.train_probe(emb, labels)
    assert r1 == r2


def test_ac4_insufficient_class_raises(pe):  # AC4 某類 3 張(min_per_class=4)→ ValueError 含類名
    emb, labels = _two_class(20, 3, loc=10.0, seed=4)     # B 僅 3 張 < 4
    with pytest.raises(ValueError) as ei:
        pe.train_probe(emb, labels)
    msg = str(ei.value)
    assert "insufficient_class" in msg                    # 設計釘死格式 insufficient_class:<類名>
    assert "B" in msg                                     # 訊息須含不足的類名


def test_ac5_plateau_pinned(pe):  # AC5 報酬遞減:plateau,slope=0.002(<0.005),reason 含 "0.002"
    r = pe.plateau_advice([100, 200, 300], [.70, .80, .802])
    # slope = (0.802-0.80)/max(100,1)*100 = 0.002 < eps 0.005
    assert r["status"] == "plateau"
    assert abs(r["slope_per_100"] - 0.002) < 1e-9
    assert "0.002" in r["reason"]                         # %.3f(0.002) == "0.002"


def test_ac5_insufficient_points(pe):  # AC5 點數 < min_points(3)→ insufficient
    r = pe.plateau_advice([100, 200], [.70, .80])         # 只有 2 點
    assert r["status"] == "insufficient"


def test_ac5_improving(pe):  # AC5 slope=0.1(>0.005)→ improving
    r = pe.plateau_advice([100, 200, 300], [.70, .80, .90])
    # slope = (0.90-0.80)/max(100,1)*100 = 0.1
    assert r["status"] == "improving"
    assert abs(r["slope_per_100"] - 0.1) < 1e-9


def test_ac6_no_streamlit_import():  # AC6 乾淨子行程 import probe_eval → 不得拉入 streamlit
    code = (
        "import sys, probe_eval\n"
        "st = [m for m in sys.modules if m == 'streamlit' or m.startswith('streamlit.')]\n"
        "assert not st, st\n"
    )
    env = dict(os.environ, PYTHONPATH=str(_SCRIPTS))
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr                    # 未實作時 import 失敗 → 非零 → 紅


# ── 推導 metamorphic(設計未明列,PM 自加)───────────────────────────────
def test_metamorphic_relabel_invariance(pe):  # 保序一致改名 → acc/macro_f1/n 切分不變
    """metrics 對「保序、一致的類別改名」不變:A→cat、B→dog(cat<dog 保持與 A<B 同序,
    故分層分割落在相同的列 → 相同數值問題)。用稍重疊資料使 acc<1,讓不變性非平凡。
    這條逼出『把類名混進度量計算』或『分層依名字順序而非群組』的實作 bug。"""
    emb, labels = _two_class(30, 30, loc=0.8, seed=5)     # 重疊 → acc 介於 (0.5, 1.0)
    base = pe.train_probe(emb, labels)
    mapping = {"A": "cat", "B": "dog"}                    # 保序改名(cat<dog、A<B)
    renamed = pe.train_probe(emb, [mapping[l] for l in labels])
    assert renamed["acc"] == base["acc"]
    assert renamed["macro_f1"] == base["macro_f1"]
    assert renamed["n_train"] == base["n_train"]
    assert renamed["n_test"] == base["n_test"]
    assert set(renamed["classes"]) == {"cat", "dog"}


def test_metamorphic_plateau_vertical_shift(pe):  # ys 整體平移常數 → slope/status 不變
    """slope 只依 y[-1]-y[-2] 的差,對所有 y 加同一常數應完全不變(status 與 slope 皆同)。
    這條逼出『用絕對值而非增量判 plateau』的實作 bug。"""
    xs = [100, 200, 300]
    ys = [.70, .80, .802]
    base = pe.plateau_advice(xs, ys)
    shifted = pe.plateau_advice(xs, [y + 0.05 for y in ys])
    assert shifted["status"] == base["status"]
    assert abs(shifted["slope_per_100"] - base["slope_per_100"]) < 1e-9
