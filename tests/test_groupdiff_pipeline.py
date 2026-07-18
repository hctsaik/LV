"""驗收:22 groupdiff_pipeline(雙資料夾 → 群差異結果包 + 匯出)。

設計:3_Architect_Design/22_groupdiff_pipeline.md。tmp 檔案樹 + 注入 extractor(免真模型):
Good=噪點圖;Bad=左上 8px 純黑簽名 → extractor 在 patch 5 加位移;左上純白 → 回 (2,2) grid
(製造 grid 不一致)。特徵決定論(以影像像素和當 seed)。
"""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from PIL import Image

D = 16
SHIFT_PATCH = 5


def _mk_imgs(d, n, *, marker=None, seed0=0):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        arr = rng.integers(90, 160, (56, 56, 3)).astype("uint8")
        if marker == "bad":
            arr[:8, :8] = 0
        elif marker == "weird":
            arr[:8, :8] = 255
        Image.fromarray(arr).save(d / f"img_{seed0 + i:03d}.png")


def _mk_extractor(calls):
    """注入 extractor(契約同 patch_features:proc PIL → (feats (P,D), (gh,gw)))。"""
    rng = np.random.default_rng(999)
    base = rng.normal(size=(16, D))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    shift = rng.normal(size=(D,))
    shift /= np.linalg.norm(shift)

    def ext(proc):
        calls.append(1)
        arr = np.asarray(proc, dtype=float)
        corner = arr[:16, :16].mean()
        if corner > 225:                      # weird → 不相容 grid
            f = base[:4].copy()
            f /= np.linalg.norm(f, axis=1, keepdims=True)
            return f.astype(np.float32), (2, 2)
        r2 = np.random.default_rng(int(arr.sum()) % (2 ** 32))
        f = base + r2.normal(scale=0.15, size=(16, D))
        if corner < 30:                       # bad 簽名 → 植入位移
            f[SHIFT_PATCH] += 1.2 * shift
        f /= np.clip(np.linalg.norm(f, axis=1, keepdims=True), 1e-12, None)
        return f.astype(np.float32), (4, 4)

    return ext


def _std(tmp, *, n_good=8, n_bad=8):
    good, bad = tmp / "good", tmp / "bad"
    _mk_imgs(good, n_good, seed0=0)
    _mk_imgs(bad, n_bad, marker="bad", seed0=100)
    return good, bad


def test_ac1_end_to_end_synthetic(tmp_path):
    """# AC1:8 good + 8 bad → verdict True、rank1 峰=植入 patch、形狀/欄位契約。"""
    from groupdiff_pipeline import run_groupdiff
    good, bad = _std(tmp_path)
    res = run_groupdiff(good, bad, extractor=_mk_extractor([]),
                        cache_root=tmp_path / "cache", n_perm=100, seed=0)
    assert res["verdict"] is True
    assert res["regions"], "植入差異應至少產生一個區域"
    assert res["regions"][0]["peak_idx"] == SHIFT_PATCH
    assert res["z_map"].shape == (4, 4)
    assert res["grid"] == (4, 4)
    assert "%" in res["regions"][0]["text"]
    assert len(res["good_paths"]) == 8 and len(res["bad_paths"]) == 8
    for r in res["regions"]:
        x0, y0, x1, y1 = r["bbox_px"]
        assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
    assert res["n_good"] == 8 and res["n_bad"] == 8


def test_ac2_null_honest(tmp_path):
    """# AC2:同分布兩夾(3 組)→ verdict True 至多 1 組。"""
    from groupdiff_pipeline import run_groupdiff
    hits = 0
    for k in range(3):
        a, b = tmp_path / f"a{k}", tmp_path / f"b{k}"
        _mk_imgs(a, 8, seed0=1000 + 50 * k)
        _mk_imgs(b, 8, seed0=2000 + 50 * k)
        res = run_groupdiff(a, b, extractor=_mk_extractor([]),
                            cache_root=tmp_path / f"cache{k}", n_perm=100, seed=0)
        hits += int(res["verdict"])
    assert hits <= 1, f"null 3 組中 {hits} 組誤判有穩定差異"


def test_ac3_cache_hit_skips_extractor(tmp_path):
    """# AC3:同參數重跑 → extractor 零呼叫(.npz 快取命中)、verdict 不變。"""
    from groupdiff_pipeline import run_groupdiff
    good, bad = _std(tmp_path)
    calls = []
    ext = _mk_extractor(calls)
    r1 = run_groupdiff(good, bad, extractor=ext, cache_root=tmp_path / "cache",
                       n_perm=100, seed=0)
    n_first = len(calls)
    assert n_first == 16
    calls.clear()
    r2 = run_groupdiff(good, bad, extractor=ext, cache_root=tmp_path / "cache",
                       n_perm=100, seed=0)
    assert len(calls) == 0, "第二輪應快取命中,不重抽特徵"
    assert r2["verdict"] == r1["verdict"]


def test_ac4_grid_mismatch_skipped_with_warning(tmp_path):
    """# AC4:摻 1 張不相容 grid → 剔除 + skipped 計數 + 對齊警示,其餘照常。"""
    from groupdiff_pipeline import run_groupdiff
    good, bad = _std(tmp_path)
    _mk_imgs(good, 1, marker="weird", seed0=900)
    res = run_groupdiff(good, bad, extractor=_mk_extractor([]),
                        cache_root=tmp_path / "cache", n_perm=100, seed=0)
    assert res["skipped"]["grid"] == 1
    assert res["alignment_warning"] is True
    assert res["warn_msgs"]
    assert res["n_good"] == 8 and res["n_bad"] == 8
    assert res["verdict"] is True


def test_ac5_validation_errors(tmp_path):
    """# AC5:資料夾不存在 / 可讀 <2 / 匯出進來源 → ValueError(訊息含路徑)。"""
    from groupdiff_pipeline import export_report, run_groupdiff
    good, bad = _std(tmp_path)
    missing = tmp_path / "no_such_dir"
    with pytest.raises(ValueError) as ei:
        run_groupdiff(missing, bad, extractor=_mk_extractor([]),
                      cache_root=tmp_path / "cache")
    assert str(missing) in str(ei.value)
    single = tmp_path / "single"
    _mk_imgs(single, 1, seed0=500)
    with pytest.raises(ValueError):
        run_groupdiff(single, bad, extractor=_mk_extractor([]),
                      cache_root=tmp_path / "cache")
    res = run_groupdiff(good, bad, extractor=_mk_extractor([]),
                        cache_root=tmp_path / "cache", n_perm=100, seed=0)
    with pytest.raises(ValueError):
        export_report(res, good / "out")
    with pytest.raises(ValueError):
        export_report(res, good)


def test_ac6_export_roundtrip_and_zero_source_writes(tmp_path):
    """# AC6:匯出檔案齊全非空、CSV/JSON 往返一致、來源資料夾零寫入。"""
    from groupdiff_pipeline import export_report, run_groupdiff
    good, bad = _std(tmp_path)
    before = {p: p.stat().st_size for d in (good, bad) for p in sorted(d.rglob("*"))}
    res = run_groupdiff(good, bad, extractor=_mk_extractor([]),
                        cache_root=tmp_path / "cache", n_perm=100, seed=0)
    out = tmp_path / "report"
    ret = export_report(res, out)
    for name in ("heatmap_good.png", "heatmap_bad.png", "regions.csv",
                 "region_01.png", "summary.json"):
        f = out / name
        assert f.exists() and f.stat().st_size > 0, f"缺匯出檔或空檔:{name}"
    with open(out / "regions.csv", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == len(res["regions"]) + 1
    ranks = [int(r[0]) for r in rows[1:]]
    assert ranks == sorted(ranks)
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summ["verdict"] == res["verdict"]
    assert summ["p_global"] == pytest.approx(res["p_global"])
    after = {p: p.stat().st_size for d in (good, bad) for p in sorted(d.rglob("*"))}
    assert before == after, "匯出不得寫入來源資料夾"
    assert set(ret["files"]) >= {str(out / "regions.csv"), str(out / "summary.json")}


def test_ac7_progress_monotone(tmp_path):
    """# AC7:progress frac 單調不減、終值 ≥0.99、text 非空。"""
    from groupdiff_pipeline import run_groupdiff
    good, bad = _std(tmp_path)
    seen = []
    run_groupdiff(good, bad, extractor=_mk_extractor([]),
                  cache_root=tmp_path / "cache", n_perm=100, seed=0,
                  progress=lambda f, t: seen.append((float(f), str(t))))
    assert seen, "progress 應被呼叫"
    fracs = [f for f, _ in seen]
    assert all(b >= a for a, b in zip(fracs, fracs[1:])), "frac 不得倒退"
    assert fracs[-1] >= 0.99
    assert all(t for _, t in seen)


def test_ac8_pca_side_evidence(tmp_path):
    """# AC8:pca2d 形狀/解釋率遞減;樣本 <3 → ValueError。"""
    from groupdiff_pipeline import pca2d, run_groupdiff
    good, bad = _std(tmp_path)
    res = run_groupdiff(good, bad, extractor=_mk_extractor([]),
                        cache_root=tmp_path / "cache", n_perm=100, seed=0)
    out = pca2d(res["gmean_good"], res["gmean_bad"])
    assert out["xy_good"].shape == (8, 2) and out["xy_bad"].shape == (8, 2)
    assert out["evr"].shape == (2,) and out["evr"][0] >= out["evr"][1]
    with pytest.raises(ValueError):
        pca2d(res["gmean_good"][:1], res["gmean_bad"][:1])


def test_ac9_region_compare_images(tmp_path):
    """# AC9:對照放大圖 good/bad 各 ≤ n_each 張 PIL;代表圖被刪 → 跳過不炸。"""
    from groupdiff_pipeline import region_compare_images, run_groupdiff
    good, bad = _std(tmp_path)
    res = run_groupdiff(good, bad, extractor=_mk_extractor([]),
                        cache_root=tmp_path / "cache", n_perm=100, seed=0)
    region = res["regions"][0]
    imgs = region_compare_images(res, region, n_each=2)
    assert 1 <= len(imgs["good"]) <= 2 and 1 <= len(imgs["bad"]) <= 2
    for im in imgs["good"] + imgs["bad"]:
        assert isinstance(im, Image.Image) and im.size[0] > 0 and im.size[1] > 0
    n_bad_before = len(imgs["bad"])
    from pathlib import Path
    Path(region["bad_paths"][0]).unlink()
    imgs2 = region_compare_images(res, region, n_each=2)
    assert len(imgs2["bad"]) == n_bad_before - 1
