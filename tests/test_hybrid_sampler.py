"""驗收:hybrid_sampler(設計 3_Architect_Design/11_hybrid_sampler.md · 第 5 節單元 AC)。

不確定 × 多樣性 × 類配額 混合選樣:先取 top-(oversample*k) 不確定,再在其 embedding 上
跑 FPS 取 k,可選類配額(弱類優先)。核心函式 hybrid_select / near_duplicate_pairs。

先紅策略(對齊 test_pool_registry 慣例):目標模組 `hybrid_sampler` 尚未實作 →
只在 fixture `hs` 內 import(collection 乾淨;未實作時為『執行期紅』test error,
非 collection error)。top-level 僅 import stdlib / numpy / pytest。

本檔只放**單元** AC(合成 embedding、決定論 seed)。GUI 的 AC-G(策略可用 / 接 08 / 接 09)
屬 tests/e2e/test_hybrid_sampler_e2e.py,不在本輪、不寫進本檔。
"""
import numpy as np
import pytest


@pytest.fixture
def hs():
    """import 隔離後的 hybrid_sampler(scripts/ 已在 conftest sys.path)。
    未實作時此處 ModuleNotFoundError → 該測 error(先紅),collection 不受影響。"""
    import importlib
    import hybrid_sampler
    return importlib.reload(hybrid_sampler)


# ── 共用合成 fixture(設計 §6 建議:AC3/AC7 共用)────────────────────────────
def _cluster_fixture(seed=0):
    """3 個緊密叢(各 20 點,叢內 cosine 距 ≈0)+ 40 散點,共 N=100、D=16。
    不確定分數:叢0 最高(→ top-10/top-20 全落叢0)、散點居中、叢1/叢2 最低。
    → 候選池 top-(3*10)=top-30 = 叢0(20)+散點(10),FPS 能把多樣性攤開;
      純 top-10 全在叢0(45 對近重複),混合選樣近重複遠少於其半數。"""
    rng = np.random.default_rng(seed)
    D, N = 16, 100
    emb = np.zeros((N, D))
    unc = np.zeros(N)
    centers = np.zeros((3, D))
    centers[0, 0] = centers[1, 1] = centers[2, 2] = 1.0
    sigma = 0.003  # 叢內擾動極小 → 叢內 cosine 距 << 0.05
    for c in range(3):                     # 0-19=叢0, 20-39=叢1, 40-59=叢2
        for j in range(20):
            emb[c * 20 + j] = centers[c] + sigma * rng.standard_normal(D)
    for i in range(60, 100):               # 60-99 = 40 個散點(單位球面隨機)
        v = rng.standard_normal(D)
        emb[i] = v / np.linalg.norm(v)
    for i in range(0, 20):                  # 叢0 最不確定
        unc[i] = 100.0 + rng.random()
    for i in range(60, 100):                # 散點居中
        unc[i] = 50.0 + rng.random()
    for i in range(20, 60):                 # 叢1 + 叢2 最不確定度最低
        unc[i] = 10.0 + rng.random()
    return emb, unc


def _stable_order(unc):
    """設計指定的候選排序:np.argsort(-uncertainty, kind='stable')。"""
    return [int(i) for i in np.argsort(-np.asarray(unc), kind="stable")]


def _quota_fixture(seed=1, n_rare=3):
    """類配額用:N=30、D=8,uncertainty=30-i(index0 最不確定)。
    labels 三類 common/medium/rare。n_rare=3 → rare 落 {0,1,17}(皆入 top-18 候選池,
    池內 rare 最不確定 2 張 = {0,1});n_rare=2 → rare 僅 {0,1}(全體只有 2 張 rare)。"""
    rng = np.random.default_rng(seed)
    N, D = 30, 8
    emb = rng.standard_normal((N, D))
    unc = np.array([30.0 - i for i in range(N)])
    rare_idx = {0, 1, 17} if n_rare == 3 else {0, 1}
    labels = []
    for i in range(N):
        if i in rare_idx:
            labels.append("rare")
        elif i % 2 == 0:
            labels.append("common")
        else:
            labels.append("medium")
    return emb, unc, labels, rare_idx


# ── 單元 AC ────────────────────────────────────────────────────────────────
def test_ac1_length_and_range(hs):  # AC1 長度與範圍:N=100,k=10 → 10 個相異索引 ∈ [0,100)
    emb, unc = _cluster_fixture()
    sel = hs.hybrid_select(emb, unc, 10)
    assert len(sel) == 10
    assert len(set(sel)) == 10                      # 相異
    assert all(0 <= i < 100 for i in sel)           # 皆在 [0,100)


def test_ac2_uncertainty_pool(hs):  # AC2 不確定門檻:選中皆 ∈ top-(oversample*k),池外永不入選
    emb, unc = _cluster_fixture()
    sel = hs.hybrid_select(emb, unc, 10)             # 預設 oversample=3
    cand = set(_stable_order(unc)[:3 * 10])          # 候選池 = top-30
    assert set(sel) <= cand


def test_ac3_diversity_beats_topk(hs):  # AC3(核心)多樣性勝 top-K:近重複對 < 純 top-K 的一半
    emb, unc = _cluster_fixture()
    order = _stable_order(unc)
    top10 = order[:10]
    assert all(i < 20 for i in top10)                # 前提:最不確定前 10 全落叢0
    topk_dups = hs.near_duplicate_pairs(emb, top10, 0.05)
    assert topk_dups == 45                            # 叢0 內 10 點兩兩皆近重複 = C(10,2)
    sel = hs.hybrid_select(emb, unc, 10)              # 混合(oversample=3)
    hyb_dups = hs.near_duplicate_pairs(emb, sel, 0.05)
    assert hyb_dups < topk_dups / 2                   # 釘死:混合近重複 < 純 top-K 半數


def test_ac4_exclude(hs):  # AC4 exclude:top-3 不確定放入 exclude → 與選中集合交集為空
    emb, unc = _cluster_fixture()
    top3 = set(_stable_order(unc)[:3])
    sel = hs.hybrid_select(emb, unc, 10, exclude=set(top3))
    assert set(sel).isdisjoint(top3)
    assert len(sel) == 10                             # 剔 3 後候選仍充足,補滿到 k


def test_ac5_quota(hs):  # AC5 配額:quota={rare:2}、池內 rare 3 張 → 選中 rare≥2 且為池內最不確定 2 張
    emb, unc, labels, _ = _quota_fixture(n_rare=3)
    pool = _stable_order(unc)[:3 * 6]                 # 候選池 top-18
    rare_in_pool = [i for i in pool if labels[i] == "rare"]
    assert len(rare_in_pool) == 3                     # 前提:池內恰 3 張 rare
    top2_rare = sorted(rare_in_pool, key=lambda i: -unc[i])[:2]
    sel = hs.hybrid_select(emb, unc, 6, class_quota={"rare": 2}, labels=labels)
    sel_rare = [i for i in sel if labels[i] == "rare"]
    assert len(sel_rare) >= 2                         # 至少配額張數
    assert set(top2_rare) <= set(sel)                 # 且正是池內最不確定的 2 張


def test_ac6_quota_underfilled_honest(hs):  # AC6 配額不足誠實:quota 要 5 但只 2 張 → rare==2、總數仍==k
    emb, unc, labels, _ = _quota_fixture(seed=2, n_rare=2)
    sel = hs.hybrid_select(emb, unc, 6, class_quota={"rare": 5}, labels=labels)
    sel_rare = [i for i in sel if labels[i] == "rare"]
    assert len(sel_rare) == 2                         # 有多少拿多少,不硬湊
    assert len(sel) == 6                              # 缺口由 FPS 補滿到 k


def test_ac7_deterministic(hs):  # AC7 決定論:同輸入重呼 20 次逐位相等
    emb, unc = _cluster_fixture()
    runs = [hs.hybrid_select(emb, unc, 10) for _ in range(20)]
    assert all(r == runs[0] for r in runs)


def test_ac8_guards(hs):  # AC8 防呆:len(emb)!=len(unc) 拋 ValueError;k=0 回 []
    emb, unc = _cluster_fixture()
    with pytest.raises(ValueError):
        hs.hybrid_select(emb[:5], unc[:6], 3)         # 長度不符 → 明確契約錯
    assert hs.hybrid_select(emb, unc, 0) == []        # k<=0 → []


# ── 推導 metamorphic 測試(設計未明列,PM 自加)─────────────────────────────
def test_md1_monotone_uncertainty_invariance(hs):  # AC-D1 不確定分數正仿射變換 → 選樣結果不變
    """選樣只依 uncertainty 的『排序』(np.argsort);對分數做保序的正仿射變換
    (a*u+b, a>0)不改變任何名次 → 候選池與 FPS 全程決定論一致 → 輸出應逐位相等。
    這條不變量能逼出『拿 uncertainty 絕對值進 FPS / 門檻』之類的實作漏洞。"""
    emb, unc = _cluster_fixture()
    base = hs.hybrid_select(emb, unc, 10)
    scaled = hs.hybrid_select(emb, 3.0 * np.asarray(unc) + 7.0, 10)
    assert scaled == base


def test_md2_near_dup_monotonic_in_threshold(hs):  # AC-D2 near_duplicate_pairs 對 thr 單調非遞減
    """近重複對數 = 集合內 cosine 距 < thr 的無序對數;thr 放大只會納入更多對,
    絕不減少 → 對 thr 單調非遞減。取「叢0 5 點 + 叢1 5 點」跨兩個距離尺度
    (叢內距 ≈7e-5、跨叢距 ≈1.0):小 thr 只計叢內 20 對、放到跨叢尺度才補齊 45 對,
    是個能區分門檻的不變量(設計未明列),防『把 < 寫成 > / 距離取負』等實作反向錯。"""
    emb, _ = _cluster_fixture()
    idxs = [0, 1, 2, 3, 4, 20, 21, 22, 23, 24]        # 叢0 前 5 + 叢1 前 5
    c_tight = hs.near_duplicate_pairs(emb, idxs, 0.001)
    c_mid = hs.near_duplicate_pairs(emb, idxs, 0.5)
    c_loose = hs.near_duplicate_pairs(emb, idxs, 1.05)
    assert c_tight <= c_mid <= c_loose                # 單調非遞減
    assert c_tight == 20                              # 小 thr:僅兩叢各自內部 C(5,2)*2
    assert c_loose == 45                              # thr 跨過叢間距 → C(10,2) 全計入
