"""驗收:25 audit_pipeline(資料夾+CSV → 體檢報告包 + 匯出)。

設計:3_Architect_Design/25_audit_pipeline.md §4。tmp 檔案樹 + 注入 embed_fn(免真模型)。
植入手算:可讀 25 張(train/classA 12、train/classB 7、train/classC 2、val/classA 4)、
壞檔 1;exact 額外複本 5(dup×3 + leak×1 + conflict×1);leak 對 1;conflict 組 1;
val 缺類 [classB, classC];獨特 sha 20(25−5)。
"""
from __future__ import annotations

import csv
import json
import shutil

import numpy as np
import pytest
from PIL import Image


def _mk_img(path, seed, solid=None):
    if solid is None:
        rng = np.random.default_rng(seed)
        arr = rng.integers(60, 200, (28, 28, 3)).astype("uint8")
    else:
        arr = np.full((28, 28, 3), solid, dtype="uint8")
    Image.fromarray(arr).save(path)


def _mk_tree(root):
    a = root / "train" / "classA"
    b = root / "train" / "classB"
    c = root / "train" / "classC"
    va = root / "val" / "classA"
    for d in (a, b, c, va):
        d.mkdir(parents=True)
    for i in range(8):
        _mk_img(a / f"A_{i:02d}.png", 10 + i)
    for i in range(3):
        shutil.copy(a / "A_00.png", a / f"dup_{i}.png")       # exact ×3
    _mk_img(a / "outlier.png", 0, solid=140)                   # 純色離群
    for i in range(6):
        _mk_img(b / f"B_{i:02d}.png", 30 + i)
    shutil.copy(a / "A_01.png", b / "conflictcopy.png")        # 同內容異標
    for i in range(2):
        _mk_img(c / f"C_{i:02d}.png", 50 + i)
    for i in range(3):
        _mk_img(va / f"V_{i:02d}.png", 70 + i)
    shutil.copy(a / "A_02.png", va / "leakcopy.png")           # 跨 split 洩漏
    (va / "corrupt.png").write_text("not an image")            # 壞檔
    return root


def _mk_embed(calls):
    rng = np.random.default_rng(999)
    base = rng.normal(size=16)
    base /= np.linalg.norm(base)
    far = rng.normal(size=16)
    far /= np.linalg.norm(far)

    def fn(path):
        calls.append(str(path))
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=float)
        if arr.std() < 1.0:                    # 純色 → 遠向量(離群)
            v = far.copy()
        else:
            r = np.random.default_rng(int(arr.sum()) % (2 ** 32))
            v = base + 0.05 * r.normal(size=16)
        v /= np.linalg.norm(v)
        return v.astype(np.float32)

    return fn


def _mk_csv(root, out_csv):
    """train 除 classC 外 → T1(19 列);val 可讀 → T2(4 列);bogus 5 列。"""
    rows = []
    for p in sorted((root / "train").rglob("*.png")):
        if p.parent.name != "classC":
            rows.append((p.name, "T1"))
    for p in sorted((root / "val").rglob("*.png")):
        if p.name != "corrupt.png":
            rows.append((p.name, "T2"))
    rows += [(f"bogus_{i}.png", "T1") for i in range(5)]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "tool"])
        w.writerows(rows)
    return out_csv


def _sec(res, key):
    for s in res["report"]["sections"]:
        if s["key"] == key:
            return s
    raise AssertionError(f"缺節:{key}")


def _run(tmp_path, **kw):
    from audit_pipeline import run_audit
    root = _mk_tree(tmp_path / "ds")
    calls = []
    res = run_audit(root, embed_fn=_mk_embed(calls),
                    cache_root=tmp_path / "cache", **kw)
    return res, root, calls


def test_ac1_planted_numbers_all_exact(tmp_path):
    """# AC1:六訊號植入數字逐項全對。"""
    res, _root, _ = _run(tmp_path)
    assert res["n_images"] == 25
    assert res["report"]["totals"]["n_images"] == 25
    assert _sec(res, "unreadable")["value"] == 1
    assert _sec(res, "exact_dup")["value"] == 5
    assert _sec(res, "leakage")["value"] == 1
    assert _sec(res, "label_conflict")["value"] == 1
    cov = _sec(res, "class_coverage")
    assert cov["value"] == 2
    val_item = next(it for it in cov["items"] if it["split"] == "val")
    assert val_item["missing"] == ["classB", "classC"]


def test_ac2_metadata_join_numbers(tmp_path):
    """# AC2:CSV 分組數字/未匹配雙向手算全對。"""
    from audit_pipeline import run_audit
    root = _mk_tree(tmp_path / "ds")
    csvf = _mk_csv(root, tmp_path / "meta.csv")
    res = run_audit(root, embed_fn=_mk_embed([]), cache_root=tmp_path / "cache",
                    metadata_csv=csvf)
    meta = res["meta_info"]
    assert len(meta["unmatched_row_idx"]) == 5
    assert meta["n_unmatched_images"] == 2          # classC 兩張沒列
    assert meta["n_matched_images"] == 23
    s = _sec(res, "metadata")
    tool_item = next(it for it in s["items"] if it["field"] == "tool")
    assert tool_item["groups"] == {"T1": 19, "T2": 4}


def test_ac3_no_csv_no_metadata_section(tmp_path):
    """# AC3:無 CSV → 無 metadata 節,其餘照常。"""
    res, _root, _ = _run(tmp_path)
    assert all(s["key"] != "metadata" for s in res["report"]["sections"])
    assert _sec(res, "exact_dup")["value"] == 5


def test_ac4_use_embedding_off(tmp_path):
    """# AC4:關 embedding → 零呼叫;emb 兩節「未計算」;其餘不變。"""
    res, _root, calls = _run(tmp_path, use_embedding=False)
    assert calls == []
    assert _sec(res, "outliers")["value"] is None
    assert "未計算" in _sec(res, "outliers")["text"]
    assert _sec(res, "emb_near_dup")["value"] is None
    assert _sec(res, "exact_dup")["value"] == 5


def test_multiple_dataset_roots_are_merged(tmp_path):
    from audit_pipeline import export_audit, run_audit

    root_a = _mk_tree(tmp_path / "ds_a")
    root_b = _mk_tree(tmp_path / "ds_b")
    res = run_audit([root_a, root_b], use_embedding=False,
                    cache_root=tmp_path / "cache")
    assert res["n_images"] == 50
    assert res["roots"] == [str(root_a), str(root_b)]
    with pytest.raises(ValueError):
        export_audit(res, root_b / "out")


def test_ac5_embedding_cache_by_sha(tmp_path):
    """# AC5:首輪呼叫==獨特 sha 數(20);同參數重跑 0 呼叫。"""
    from audit_pipeline import run_audit
    root = _mk_tree(tmp_path / "ds")
    calls = []
    fn = _mk_embed(calls)
    run_audit(root, embed_fn=fn, cache_root=tmp_path / "cache")
    assert len(calls) == 20, f"獨特內容 20 個,實呼叫 {len(calls)}"
    calls.clear()
    run_audit(root, embed_fn=fn, cache_root=tmp_path / "cache")
    assert calls == []


def test_ac6_export_roundtrip_zero_source_writes(tmp_path):
    """# AC6:三檔非空、json 往返、issues.csv 列數、來源零寫入、out 在 root 內拒絕。"""
    from audit_pipeline import export_audit, run_audit
    root = _mk_tree(tmp_path / "ds")
    csvf = _mk_csv(root, tmp_path / "meta.csv")
    before = {p: p.stat().st_size for p in sorted(root.rglob("*"))}
    res = run_audit(root, embed_fn=_mk_embed([]), cache_root=tmp_path / "cache",
                    metadata_csv=csvf)
    out = tmp_path / "report"
    ret = export_audit(res, out)
    for name in ("report.html", "report.json", "issues.csv"):
        f = out / name
        assert f.exists() and f.stat().st_size > 0, f"缺匯出檔:{name}"
    data = json.loads((out / "report.json").read_text(encoding="utf-8"))
    exact = next(s for s in data["sections"] if s["key"] == "exact_dup")
    assert exact["value"] == 5
    with open(out / "issues.csv", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    # 5 複本 + 1 洩漏對 + 2 衝突張 + 10 離群 + 2 meta 未匹配影像 + 表頭
    assert len(rows) == 5 + 1 + 2 + 10 + 2 + 1
    after = {p: p.stat().st_size for p in sorted(root.rglob("*"))}
    assert before == after, "體檢/匯出不得寫來源資料夾"
    with pytest.raises(ValueError):
        export_audit(res, root / "sub")
    with pytest.raises(ValueError):
        export_audit(res, root)


def test_ac7_single_split_and_flat(tmp_path):
    """# AC7:無 train/val 的 class 夾 root → coverage 不適用;平鋪 → conflict 0。"""
    from audit_pipeline import run_audit
    root2 = tmp_path / "single"
    (root2 / "classA").mkdir(parents=True)
    (root2 / "classB").mkdir()
    for i in range(3):
        _mk_img(root2 / "classA" / f"a{i}.png", 100 + i)
    for i in range(2):
        _mk_img(root2 / "classB" / f"b{i}.png", 200 + i)
    res = run_audit(root2, embed_fn=_mk_embed([]), cache_root=tmp_path / "c2")
    cov = _sec(res, "class_coverage")
    assert cov["value"] is None and "不適用" in cov["text"]

    root3 = tmp_path / "flat"
    root3.mkdir()
    for i in range(3):
        _mk_img(root3 / f"f{i}.png", 300 + i)
    res3 = run_audit(root3, embed_fn=_mk_embed([]), cache_root=tmp_path / "c3")
    assert _sec(res3, "label_conflict")["value"] == 0


def test_ac8_validation_and_progress(tmp_path):
    """# AC8:root 不存在(訊息含路徑)/單張 → ValueError;progress 單調終值 ≥0.99。"""
    from audit_pipeline import run_audit
    missing = tmp_path / "nope"
    with pytest.raises(ValueError) as ei:
        run_audit(missing, embed_fn=_mk_embed([]), cache_root=tmp_path / "c")
    assert str(missing) in str(ei.value)
    single = tmp_path / "single1"
    single.mkdir()
    _mk_img(single / "only.png", 1)
    with pytest.raises(ValueError):
        run_audit(single, embed_fn=_mk_embed([]), cache_root=tmp_path / "c")
    seen = []
    root = _mk_tree(tmp_path / "ds")
    run_audit(root, embed_fn=_mk_embed([]), cache_root=tmp_path / "cache",
              progress=lambda f, t: seen.append((float(f), str(t))))
    fracs = [f for f, _ in seen]
    assert seen and all(b >= a for a, b in zip(fracs, fracs[1:]))
    assert fracs[-1] >= 0.99 and all(t for _, t in seen)


def test_ac9_outlier_top1_is_planted(tmp_path):
    """# AC9:注入遠向量的純色圖 → outliers 第一名。"""
    res, _root, _ = _run(tmp_path)
    items = _sec(res, "outliers")["items"]
    assert items and items[0]["path"].endswith("outlier.png")
