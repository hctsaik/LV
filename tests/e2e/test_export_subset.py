"""Unit tests for scripts/export_subset.py — the framework-free copy engine.

Builds tiny YOLO datasets in tmp_path (PIL pngs + 5-col labels + classes.txt)
and exercises the §8 acceptance criteria: yolo layout / dst-containment hard
guard / sha256 dedup / multi-source class_remap / crop-out / on_exists /
manifest-only / full ExportReport ledger / split preservation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from export_subset import (
    ExportItem,
    ExportReport,
    export_subset,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def _img(path: Path, seed: int = 0, size=(48, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (*size[::-1], 3)).astype("uint8")
    Image.fromarray(arr).save(path)
    return path


def _make_dataset(root: Path, *, name: str = "ds", classes=("cat", "dog"),
                  split: str | None = None, seed: int = 0,
                  labels: list[str] | None = None) -> dict:
    """Create a small YOLO dataset under ``root/name`` and return paths.

    images[/split]/img.png, labels[/split]/img.txt, classes.txt.
    """
    base = root / name
    img_dir = base / "images" / split if split else base / "images"
    lbl_dir = base / "labels" / split if split else base / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    img = _img(img_dir / "img.png", seed=seed)
    lbl = lbl_dir / "img.txt"
    lbl.write_text("\n".join(labels or ["0 0.5 0.5 0.4 0.4",
                                        "1 0.25 0.25 0.2 0.2"]) + "\n",
                   encoding="utf-8")
    (base / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    return {"root": base, "image": img, "label": lbl,
            "classes": list(classes), "split": split}


def _read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


# ── 1. yolo layout ───────────────────────────────────────────────────────────
def test_yolo_layout_basic(tmp_path):
    ds = _make_dataset(tmp_path / "src")
    item = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"], source_tool="visualize",
                      source_tag="outlier", reason="3.2σ", dataset_id="ds1")
    dst = tmp_path / "out"
    rep = export_subset([item], dst, mode="copy", layout="yolo")

    assert rep.exported == 1
    assert (dst / "images").is_dir()
    assert (dst / "labels").is_dir()
    assert (dst / "classes.txt").exists()
    assert (dst / "data.yaml").exists()

    imgs = list((dst / "images").glob("*.png"))
    assert len(imgs) == 1
    # 檔名 = 原圖 basename（不再是 sha256）
    assert imgs[0].stem == ds["image"].stem  # "img"
    lbl = (dst / "labels" / f"{imgs[0].stem}.txt")
    assert lbl.exists()
    # subset classes are name-sorted: cat=0, dog=1 → same order here
    classes = (dst / "classes.txt").read_text(encoding="utf-8").split()
    assert classes == ["cat", "dog"]
    rows = lbl.read_text(encoding="utf-8").split("\n")
    assert rows[0].startswith("0 ")   # cat stays 0
    assert rows[1].startswith("1 ")   # dog stays 1


def test_yolo_class_remap_nonidentity(tmp_path):
    # source order [dog, cat] → subset sorted [cat, dog] → ids must flip
    ds = _make_dataset(tmp_path / "src", classes=("dog", "cat"),
                       labels=["0 0.5 0.5 0.4 0.4"])  # 0=dog
    item = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"], dataset_id="ds1")
    dst = tmp_path / "out"
    export_subset([item], dst, layout="yolo")
    classes = (dst / "classes.txt").read_text(encoding="utf-8").split()
    assert classes == ["cat", "dog"]
    lbl = next((dst / "labels").glob("*.txt"))
    # source 0=dog should remap to subset id 1
    assert lbl.read_text(encoding="utf-8").strip().startswith("1 ")


# ── 2. dst-containment hard guard ────────────────────────────────────────────
def test_dst_containment_inside_source_raises_no_write(tmp_path):
    ds = _make_dataset(tmp_path / "src")
    item = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"])
    dst = ds["root"] / "exported_here"  # inside the source dataset root
    with pytest.raises(ValueError, match="污染|containment|overlaps"):
        export_subset([item], dst, layout="yolo")
    assert not dst.exists()  # nothing written


def test_dst_containment_source_inside_dst_raises(tmp_path):
    ds = _make_dataset(tmp_path / "container" / "src")
    item = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"])
    dst = tmp_path / "container"  # the source root sits INSIDE dst
    with pytest.raises(ValueError):
        export_subset([item], dst, layout="yolo")


def test_dst_containment_explicit_source_roots(tmp_path):
    ds = _make_dataset(tmp_path / "src")
    item = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"])
    dst = ds["root"] / "out"  # genuinely inside the supplied source root
    with pytest.raises(ValueError):
        export_subset([item], dst, layout="yolo",
                      source_roots=[ds["root"]])
    assert not dst.exists()


# ── 3. sha256 dedup ──────────────────────────────────────────────────────────
def test_sha256_dedup_merges_source_tags(tmp_path):
    ds = _make_dataset(tmp_path / "src")
    a = ExportItem(image_path=ds["image"], label_path=ds["label"],
                   class_names=ds["classes"], source_tool="visualize",
                   source_tag="outlier", dataset_id="ds1")
    b = ExportItem(image_path=ds["image"], label_path=ds["label"],
                   class_names=ds["classes"], source_tool="objcov",
                   source_tag="rare-class", dataset_id="ds1")
    dst = tmp_path / "out"
    rep = export_subset([a, b], dst, layout="yolo")

    assert rep.exported == 1
    assert rep.deduped == 1
    assert len(list((dst / "images").glob("*.png"))) == 1
    man = _read_csv(dst / "manifest.csv")
    assert len(man) == 1
    tags = man[0]["source_tag"].split("|")
    assert set(tags) == {"outlier", "rare-class"}
    tools = man[0]["source_tool"].split("|")
    assert set(tools) == {"visualize", "objcov"}


# ── 4. multi-source class_remap (union) ──────────────────────────────────────
def test_multi_source_class_remap_union(tmp_path):
    # ds_a classes [cat, dog], ds_b classes [bird, cat]
    a = _make_dataset(tmp_path / "a", name="dsA", classes=("cat", "dog"),
                      seed=1, labels=["1 0.5 0.5 0.3 0.3"])  # 1=dog
    b = _make_dataset(tmp_path / "b", name="dsB", classes=("bird", "cat"),
                      seed=2, labels=["0 0.5 0.5 0.3 0.3"])  # 0=bird
    ia = ExportItem(image_path=a["image"], label_path=a["label"],
                    class_names=a["classes"], dataset_id="A")
    ib = ExportItem(image_path=b["image"], label_path=b["label"],
                    class_names=b["classes"], dataset_id="B")
    dst = tmp_path / "out"
    rep = export_subset([ia, ib], dst, layout="yolo")

    assert rep.exported == 2
    # union, name-sorted: bird=0, cat=1, dog=2
    classes = (dst / "classes.txt").read_text(encoding="utf-8").split()
    assert classes == ["bird", "cat", "dog"]

    # map each exported label back to its source via the manifest "dst" column
    # (檔名不再是 sha；用 dst 寫出的相對路徑回推 label 檔名)
    man = {r["sha256"]: r for r in _read_csv(dst / "manifest.csv")}
    row_a = next(r for r in man.values() if Path(r["source_path"]) == a["image"])
    row_b = next(r for r in man.values() if Path(r["source_path"]) == b["image"])
    lbl_a = (dst / "labels" / f"{Path(row_a['dst']).stem}.txt").read_text(encoding="utf-8")
    lbl_b = (dst / "labels" / f"{Path(row_b['dst']).stem}.txt").read_text(encoding="utf-8")
    assert lbl_a.strip().startswith("2 ")  # dsA dog(1) → subset dog(2)
    assert lbl_b.strip().startswith("0 ")  # dsB bird(0) → subset bird(0)


# ── 5. crop-out ──────────────────────────────────────────────────────────────
def test_crop_out_object_level(tmp_path):
    ds = _make_dataset(tmp_path / "src",
                       labels=["0 0.5 0.5 0.4 0.4", "1 0.25 0.25 0.2 0.2"])
    item = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"], level="object",
                      object_ids=[0, 1], source_tag="rare", dataset_id="ds1")
    dst = tmp_path / "out"
    rep = export_subset([item], dst, layout="crop-out", pad=0.1)

    assert rep.exported == 2
    # ImageFolder layout images/<class>/<sha>_<box>.jpg
    cat_crops = list((dst / "images" / "cat").glob("*_0.jpg"))
    dog_crops = list((dst / "images" / "dog").glob("*_1.jpg"))
    assert len(cat_crops) == 1
    assert len(dog_crops) == 1

    # object-level manifest
    obj = _read_csv(dst / "objects.csv")
    assert len(obj) == 2
    assert {r["class"] for r in obj} == {"cat", "dog"}
    assert all(r["crop_sha256"] and r["source_sha256"] for r in obj)
    assert all(float(r["pad"]) == 0.1 for r in obj)

    # data.yaml → classification
    yaml = (dst / "data.yaml").read_text(encoding="utf-8")
    assert "task: classification" in yaml
    # no detection label files written in crop-out
    assert not (dst / "labels").exists()


# ── 5b. yolo object-level → 原圖 + 整份標記（非裁切圖） ───────────────────────
def test_yolo_object_level_exports_full_image(tmp_path):
    """level=object 的 ExportItem 在 yolo layout 應匯出「整張原圖 + 整份標記」，
    manifest level=object 且記下被挑中的 object_ids（不是裁切小圖）。"""
    ds = _make_dataset(tmp_path / "src",
                       labels=["0 0.5 0.5 0.4 0.4", "1 0.25 0.25 0.2 0.2"])
    item = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"], level="object",
                      object_ids=[1], source_tool="objcov",
                      source_tag="rare", dataset_id="ds1")
    dst = tmp_path / "out"
    rep = export_subset([item], dst, layout="yolo")

    assert rep.exported == 1
    # 整張原圖（檔名 = 原 basename），而非裁切小圖
    imgs = list((dst / "images").glob("*.png"))
    assert len(imgs) == 1
    assert imgs[0].stem == ds["image"].stem
    # 整份標記（兩個框都在，未被裁成單框）
    lbl = (dst / "labels" / f"{imgs[0].stem}.txt")
    assert lbl.exists()
    rows = [r for r in lbl.read_text(encoding="utf-8").split("\n") if r.strip()]
    assert len(rows) == 2
    # manifest level=object，object_ids 記下被挑中的框
    man = _read_csv(dst / "manifest.csv")
    assert len(man) == 1
    assert man[0]["level"] == "object"
    assert man[0]["object_ids"] == "1"
    # crop-out 才有 objects.csv；yolo layout 不應產生
    assert not (dst / "objects.csv").exists()


# ── 5c. 同 basename、不同內容 → 兩張都保留（消歧義，不漏資料） ────────────────
def test_yolo_basename_collision_keeps_both(tmp_path):
    """兩個來源各有同名 img.png 但內容不同（不同 sha）→ 不可互相覆蓋/略過；
    應以 __N 後綴消歧義，兩張都匯出，且與 on_exists 無關。"""
    a = _make_dataset(tmp_path / "a", name="dsA", seed=1)
    b = _make_dataset(tmp_path / "b", name="dsB", seed=2)
    assert a["image"].name == b["image"].name  # 同 basename
    ia = ExportItem(image_path=a["image"], label_path=a["label"],
                    class_names=a["classes"], dataset_id="A")
    ib = ExportItem(image_path=b["image"], label_path=b["label"],
                    class_names=b["classes"], dataset_id="B")
    dst = tmp_path / "out"
    rep = export_subset([ia, ib], dst, layout="yolo", on_exists="skip")

    assert rep.exported == 2
    assert rep.skipped == []  # 撞名消歧義，沒有任何一張被略過
    pngs = sorted(p.name for p in (dst / "images").glob("*.png"))
    assert len(pngs) == 2
    assert "img.png" in pngs
    assert any("__1" in n for n in pngs)
    # 每張都各自有 label
    assert len(list((dst / "labels").glob("*.txt"))) == 2


def test_yolo_same_stem_diff_ext_keeps_both_labels(tmp_path):
    """同 stem、不同副檔名、不同內容（000001.png vs 000001.jpg）→ 影像本就不同名
    不會互蓋，但 label 都跟 stem 命名（000001.txt）會互蓋漏標。消歧義須以 stem 為單位：
    兩張影像各自配對到正確的 label，無一被覆蓋。"""
    a = _make_dataset(tmp_path / "a", name="dsA", seed=1,
                      labels=["0 0.5 0.5 0.4 0.4"])            # 1 box
    # 第二張：同 stem(img) 但 .jpg、不同內容、2 個框
    img_b = _img(tmp_path / "b" / "img.jpg", seed=2)
    lbl_b = tmp_path / "b" / "img.txt"
    lbl_b.write_text("0 0.5 0.5 0.4 0.4\n1 0.25 0.25 0.2 0.2\n", encoding="utf-8")
    ia = ExportItem(image_path=a["image"], label_path=a["label"],
                    class_names=["cat", "dog"], dataset_id="A")
    ib = ExportItem(image_path=img_b, label_path=lbl_b,
                    class_names=["cat", "dog"], dataset_id="B")
    dst = tmp_path / "out"
    rep = export_subset([ia, ib], dst, layout="yolo", on_exists="skip")

    assert rep.exported == 2
    imgs = sorted(p.name for p in (dst / "images").glob("*"))
    assert len(imgs) == 2                                       # 兩張影像都在
    lbls = sorted(p.name for p in (dst / "labels").glob("*.txt"))
    assert len(lbls) == 2, f"label 互蓋漏標：{lbls}"            # 兩個 label 都在、沒互蓋
    # 每張影像都有與它 stem 配對的 label
    for img_name in imgs:
        assert (dst / "labels" / f"{Path(img_name).stem}.txt").exists()
    # 行數對得上：一個 label 1 行、另一個 2 行（內容沒被另一張覆蓋）
    line_counts = sorted(
        len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])
        for p in (dst / "labels").glob("*.txt"))
    assert line_counts == [1, 2], f"label 內容被覆蓋：{line_counts}"


# ── 6. on_exists=skip ────────────────────────────────────────────────────────
def test_on_exists_skip_counts(tmp_path):
    ds = _make_dataset(tmp_path / "src")
    item = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"], dataset_id="ds1")
    dst = tmp_path / "out"
    rep1 = export_subset([item], dst, layout="yolo")
    assert rep1.exported == 1

    rep2 = export_subset([item], dst, layout="yolo", on_exists="skip")
    assert rep2.exported == 0
    assert len(rep2.skipped) == 1
    assert rep2.skipped[0]["reason"].startswith("exists")
    # still exactly one image present
    assert len(list((dst / "images").glob("*.png"))) == 1


def test_on_exists_rename_adds_suffix(tmp_path):
    ds = _make_dataset(tmp_path / "src")
    item = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"], dataset_id="ds1")
    dst = tmp_path / "out"
    export_subset([item], dst, layout="yolo")
    rep2 = export_subset([item], dst, layout="yolo", on_exists="rename")
    assert rep2.exported == 1
    pngs = sorted(p.name for p in (dst / "images").glob("*.png"))
    assert len(pngs) == 2
    assert any("__1" in n for n in pngs)


# ── 7. manifest-only ─────────────────────────────────────────────────────────
def test_manifest_only_no_image_copy(tmp_path):
    ds = _make_dataset(tmp_path / "src")
    item = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"], source_tag="dup",
                      dataset_id="ds1")
    dst = tmp_path / "out"
    rep = export_subset([item], dst, mode="manifest-only", layout="yolo")

    assert (dst / "manifest.csv").exists()
    assert (dst / "lineage.json").exists()
    # no bytes copied
    if (dst / "images").exists():
        assert not list((dst / "images").glob("*"))
    man = _read_csv(dst / "manifest.csv")
    assert len(man) == 1
    assert man[0]["sha256"]


# ── 8. ExportReport full ledger ──────────────────────────────────────────────
def test_report_full_ledger(tmp_path):
    ds = _make_dataset(tmp_path / "src", seed=1)
    ds2 = _make_dataset(tmp_path / "src2", name="ds2", seed=2)
    good = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"], source_tag="a", dataset_id="d1")
    dup = ExportItem(image_path=ds["image"], label_path=ds["label"],
                     class_names=ds["classes"], source_tag="b", dataset_id="d1")
    other = ExportItem(image_path=ds2["image"], label_path=ds2["label"],
                       class_names=ds2["classes"], dataset_id="d2")
    missing = ExportItem(image_path=tmp_path / "src" / "images" / "nope.png",
                         class_names=ds["classes"], dataset_id="d1")
    dst = tmp_path / "out"
    rep = export_subset([good, dup, other, missing], dst, layout="yolo",
                        source_roots=[ds["root"], ds2["root"]])

    assert isinstance(rep, ExportReport)
    assert rep.exported == 2          # ds image + ds2 image
    assert rep.deduped == 1           # dup of ds image
    assert len(rep.unresolved) == 1   # missing image
    assert rep.unresolved[0]["reason"] == "image not found"
    assert rep.dst == str(dst.resolve())


def test_report_label_warnings_not_silent(tmp_path):
    ds = _make_dataset(tmp_path / "src",
                       labels=["0 0.5 0.5 0.4 0.4",
                               "5 0.5 0.5 0.3 0.3",      # unknown class id
                               "0 1.5 0.5 0.3 0.3"])     # coord out of range
    item = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"], dataset_id="ds1")
    dst = tmp_path / "out"
    rep = export_subset([item], dst, layout="yolo")
    kinds = {w["kind"] for w in rep.warnings}
    assert "unknown-class-id" in kinds
    assert "coord-out-of-range" in kinds


# ── 9. preserve split ────────────────────────────────────────────────────────
def test_preserve_split(tmp_path):
    ds = _make_dataset(tmp_path / "src", split="train")
    item = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"], split="train", dataset_id="ds1")
    dst = tmp_path / "out"
    rep = export_subset([item], dst, layout="yolo")
    assert rep.exported == 1
    assert (dst / "images" / "train").is_dir()
    assert (dst / "labels" / "train").is_dir()
    assert len(list((dst / "images" / "train").glob("*.png"))) == 1
    assert len(list((dst / "labels" / "train").glob("*.txt"))) == 1


# ── extras: symlink errors logged, lineage reproducibility ───────────────────
def test_symlink_failure_logged_not_silent_copy(tmp_path, monkeypatch):
    ds = _make_dataset(tmp_path / "src")
    item = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"], dataset_id="ds1")
    dst = tmp_path / "out"

    import export_subset as es

    def _boom(src, dst_):
        raise OSError("no symlink privilege")
    monkeypatch.setattr(es.os, "symlink", _boom)
    rep = export_subset([item], dst, mode="symlink", layout="yolo")
    assert rep.exported == 0
    assert len(rep.errors) == 1
    assert "symlink failed" in rep.errors[0]["reason"]
    # NOT silently copied
    if (dst / "images").exists():
        assert not list((dst / "images").glob("*.png"))


def test_lineage_has_skew_warning_and_self_hash(tmp_path):
    ds = _make_dataset(tmp_path / "src")
    item = ExportItem(image_path=ds["image"], label_path=ds["label"],
                      class_names=ds["classes"], dataset_id="ds1")
    dst = tmp_path / "out"
    export_subset([item], dst, layout="yolo")
    lin = json.loads((dst / "lineage.json").read_text(encoding="utf-8"))
    assert "偏斜" in lin["warning"]
    assert len(lin["sha256_list"]) == 1
    assert lin["export_self_hash"]
    assert lin["class_remap"]
