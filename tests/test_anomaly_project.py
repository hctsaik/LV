"""驗收:anomaly_project —— 跨資料夾投影器(fit 舊資料基底、transform 新資料進同一座標)。
先紅:scripts/anomaly_project.py 尚未實作。

核心(對齊綜合設計裁決):transform-into-fixed-basis(非 re-fit-together);
l2norm=False 對齊 app.py:1301 現況散點;密集/離群判定走 D 維 cosine,非 2D 座標。
"""
import numpy as np
import pytest

from anomaly_project import (classify_dense, compute_tau, fit_projector,
                             load_projector, nearest_ref, save_projector,
                             transform_new)

D = 8


def _good(n=200, seed=0):
    """方向都偏 +e0 的一群(cosine 互相近)。"""
    rng = np.random.default_rng(seed)
    v = np.zeros((n, D), dtype=np.float32)
    v[:, 0] = 5.0
    v += rng.normal(0, 0.3, (n, D)).astype(np.float32)
    return v


def _outliers(m=10, axis=1):
    """方向偏 +e{axis} 的離群(cosine 對 good 遠)。"""
    v = np.zeros((m, D), dtype=np.float32)
    v[:, axis] = 5.0
    return v


def test_transform_same_batch_equals_fit_coords():  # AC1:transform 同批 == fit 的 ref_coords
    X = _good()
    b = fit_projector(X)
    coords = transform_new(b, X)
    assert coords.shape == (len(X), 2)
    # 同一基底 fit+transform 同批,數學上精確等於 ref_coords(同 vt,符號一致)
    assert np.allclose(coords, b["ref_coords"], atol=1e-4)


def test_default_l2norm_false():  # AC2:預設 l2norm=False(對齊 app.py raw SVD)
    b = fit_projector(_good())
    assert b["l2norm"] is False


def test_transform_dim_mismatch_raises():  # AC3:維度不符 → 明確 raise(非靜默 broadcasting)
    b = fit_projector(_good())
    with pytest.raises(ValueError):
        transform_new(b, np.zeros((5, D + 3), dtype=np.float32))


def test_classify_dense_uses_Ddim_cosine():  # AC4:D 維 cosine 密度判定 good→dense、離群→否
    good = _good()
    b = fit_projector(good, good_mask=np.ones(len(good), bool))
    in_good = classify_dense(good, b["good_obj_emb"], b["tau"])
    assert in_good.mean() >= 0.9, "多數舊 good 應落在自己的密集區"
    out = classify_dense(_outliers(), b["good_obj_emb"], b["tau"])
    assert not out.any(), "方向不同的離群點不應被判為落在 good 密集區"


def test_tau_excludes_self():  # AC5:tau 必須去自身,否則 self-dist=0 拉低分位
    good = _good()
    tau_self = compute_tau(good, include_self=True)    # 含自身(錯誤做法)
    tau_ok = compute_tau(good)                          # 去自身(預設)
    assert tau_ok > tau_self, "去自身的 tau 應明顯大於含自身(self-dist=0 會拉低)"


def test_nearest_ref_points_to_closest():  # AC6:最近鄰連線端點正確
    ref = np.eye(D, dtype=np.float32) * 5.0             # D 個正交方向參照點
    query = np.zeros((1, D), dtype=np.float32)
    query[0, 3] = 9.0                                   # 最像 ref[3]
    idx, dist = nearest_ref(query, ref)
    assert int(idx[0]) == 3
    assert dist[0] == pytest.approx(0.0, abs=1e-5)      # 同方向 cosine 距離≈0


def test_save_load_roundtrip(tmp_path):  # AC7:存讀後 basis 與 transform 結果一致
    X = _good()
    b = fit_projector(X, good_mask=np.ones(len(X), bool))
    p = tmp_path / "projection.npz"
    save_projector(p, b)
    b2 = load_projector(p)
    assert np.allclose(transform_new(b, X), transform_new(b2, X), atol=1e-5)
    assert np.allclose(b["mean"], b2["mean"])
    assert np.allclose(b["components"], b2["components"])
    assert b2["l2norm"] == b["l2norm"]
    assert float(b2["tau"]) == pytest.approx(float(b["tau"]), rel=1e-5)


def test_outlier_projects_without_crash():  # AC8:新離群 transform 不炸、形狀對
    b = fit_projector(_good())
    c = transform_new(b, _outliers(7))
    assert c.shape == (7, 2)
    assert np.isfinite(c).all()


def test_single_object_ref_coords_2col():  # AC9(對抗 review):N==1 的 ref_coords 仍 (N, dim_out)
    b = fit_projector(np.random.default_rng(0).normal(size=(1, D)).astype(np.float32))
    assert b["ref_coords"].shape == (1, 2)      # 不是 (1,1) → 掛載畫灰底 ref_coords[:,1] 不會 IndexError
    assert b["ref_coords"][:, 1].shape == (1,)
