"""Adversarial E2E scenarios for the 按物件・逐類別補洞 view
(`_render_objcov_view` in scripts/app.py).

Two attack surfaces:
  (A) UI-level via AppTest + _objcov_harness.py (a real subprocess-style render of
      the Streamlit page with a JSON spec). Verifies the renderer survives edge
      datasets and surfaces friendly messages instead of crashing.
  (B) Logic-level: call the real helpers (_viz_kmeans_blocks, _cov_projection,
      _objcov_embed_candidates, mine_candidates, _crop_and_embed_objects) directly,
      because AppTest cannot drive plotly drag / block selection. The brief says
      to verify such logic directly.

Run:
  set PYTHONIOENCODING=utf-8 & set PYTHONUTF8=1
  .venv\\Scripts\\python.exe -m pytest tests/e2e/test_scenarios_objcov.py -v
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from streamlit.testing.v1 import AppTest  # noqa: E402

HARNESS = str(Path(__file__).resolve().parent / "_objcov_harness.py")
COCO8 = REPO / "demo" / "coco8" / "train"
COCO128 = REPO / "demo" / "coco128" / "train"


# ───────────────────────── helpers ─────────────────────────
def run_harness(spec: dict, timeout: int = 180) -> AppTest:
    os.environ["OBJCOV_SPEC"] = json.dumps(spec)
    at = AppTest.from_file(HARNESS, default_timeout=timeout)
    at.run()
    return at


def all_text(at: AppTest) -> str:
    """Concatenate every user-visible text element for substring assertions."""
    bits = []
    for coll in (at.info, at.warning, at.error, at.caption, at.markdown,
                 at.success):
        for el in coll:
            try:
                bits.append(str(el.value))
            except Exception:
                pass
    return "\n".join(bits)


def exc_summary(at: AppTest) -> str:
    return " | ".join(f"{e.type}: {repr(e.value)[:200]}" for e in at.exception)


def make_yolo_dataset(dst: Path, *, images: list[Path], labels: dict[str, str],
                      classes: list[str] | None) -> Path:
    """Build a YOLO dataset matching the demo/coco8 layout:
        <dst>/train/images/<img>      <- cov_records point here
        <dst>/train/labels/<stem>.txt
        <dst>/classes.txt             <- read_classes_txt(root) reads root.parent

    `_cov_object_root` resolves <dst>/train (img.parent.parent), and both
    `_cov_class_names` and `_objcov_embed_candidates` look up classes via
    read_classes_txt(<dst>/train) == <dst>/classes.txt. `labels` maps
    image-stem -> label text. Returns the dataset root (<dst>/train)."""
    root = dst / "train"
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy(img, root / "images" / img.name)
        stem = img.stem
        if stem in labels:
            (root / "labels" / f"{stem}.txt").write_text(labels[stem], encoding="utf-8")
    if classes is not None:
        (dst / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    return root


def _real_helpers():
    import app
    import object_eval as oe
    return app, oe


# ════════════════════════ SCENARIO 1 ════════════════════════
# Frozen blocks: projecting candidates must NOT renumber or move the MAIN
# blocks (high-D centroids the miner queries) for the selected class.
def test_s01_frozen_blocks_after_candidate_projection(tmp_path):
    """Render once (no candidates) → capture block partition + high-D centroids.
    Embed candidates → render again in the SAME session → block IDs, partition,
    and high-D centroids must be byte-for-byte identical (cache keyed on
    main_token+class+K; n(MAIN) never changes)."""
    app, oe = _real_helpers()
    import streamlit as st
    from completeness import mine_candidates  # noqa: F401

    # Build deterministic MAIN: one class "person" with 32 objects in 4 blobs.
    dim, n_per, n_blob = 16, 8, 4
    rng = np.random.default_rng(1)
    centers = [rng.standard_normal(dim) * 3 for _ in range(n_blob)]
    memb = np.vstack([c + rng.standard_normal((n_per, dim)) * 0.25
                      for c in centers]).astype("float32")
    K = 5

    # Stage 1: blocks on MAIN-only projection, cached under a stable key.
    from app import _viz_kmeans_blocks, _cov_projection, _pad_cols
    st.session_state.clear()
    main_token, cls = "tokA_pad12_r224_cls", "person"
    cm0, _ = _cov_projection(memb, None, "pca", 2, f"{main_token}_{cls}", f"_{cls}")
    blocks0, centers0 = _viz_kmeans_blocks(
        _pad_cols(np.asarray(cm0), 3), list(range(len(memb))), K,
        f"objcov_{main_token}_{cls}_{K}")
    # high-D centroid per block, exactly as the renderer computes it
    hi0 = {b: memb[[i for i in range(len(memb)) if blocks0.get(i) == b]].mean(0)
           for b in range(len(centers0))}

    # Stage 2: candidates appear; projection re-fits jointly (coords move) BUT
    # the cached blocks key is unchanged → renderer reuses Stage-1 partition.
    cand = (rng.standard_normal((12, dim)) * 0.05).astype("float32")
    cm1, cc1 = _cov_projection(memb, cand, "pca", 2, f"{main_token}_{cls}",
                               "candtok_" + cls)
    blocks1, centers1 = _viz_kmeans_blocks(
        _pad_cols(np.asarray(cm1), 3), list(range(len(memb))), K,
        f"objcov_{main_token}_{cls}_{K}")  # SAME cache key
    hi1 = {b: memb[[i for i in range(len(memb)) if blocks1.get(i) == b]].mean(0)
           for b in range(len(centers1))}

    coords_moved = not np.allclose(cm0, cm1)
    blocks_frozen = blocks0 == blocks1
    centers_frozen = np.array_equal(centers0, centers1)
    hi_frozen = all(np.array_equal(hi0[b], hi1[b]) for b in hi0)

    print("S01 coords moved (joint refit):", coords_moved,
          "| block IDs frozen:", blocks_frozen,
          "| 2D centers frozen:", centers_frozen,
          "| high-D centroids frozen:", hi_frozen)
    # The whole point: layout shifts, blocks/centroids do NOT.
    assert blocks_frozen, "block IDs changed after candidate projection"
    assert centers_frozen, "block centers changed after candidate projection"
    assert hi_frozen, "high-D centroids (miner queries) changed — NOT frozen"


# ════════════════════════ SCENARIO 2 ════════════════════════
# Same class chosen, but candidates embedded BEFORE the first block render
# (fresh session, ordering attack). Does the partition the user ends up with
# match what they'd get from MAIN-only? (Worst case for "frozen" claim.)
def test_s02_blocks_depend_on_render_order(tmp_path):
    app, oe = _real_helpers()
    import streamlit as st
    from app import _viz_kmeans_blocks, _cov_projection, _pad_cols

    dim, K = 16, 5
    rng = np.random.default_rng(2)
    centers = [rng.standard_normal(dim) * 3 for _ in range(4)]
    memb = np.vstack([c + rng.standard_normal((8, dim)) * 0.25
                      for c in centers]).astype("float32")
    cand = (rng.standard_normal((12, dim)) * 0.05).astype("float32")
    main_token, cls = "tokB_pad12_r224_cls", "person"

    # Path A: MAIN-only first (the intended order).
    st.session_state.clear()
    cmA, _ = _cov_projection(memb, None, "pca", 2, f"{main_token}_{cls}", f"_{cls}")
    bA, _ = _viz_kmeans_blocks(_pad_cols(np.asarray(cmA), 3),
                               list(range(len(memb))), K,
                               f"objcov_{main_token}_{cls}_{K}")
    # Path B: candidates present at the very first render (cache cold).
    st.session_state.clear()
    cmB, _ = _cov_projection(memb, cand, "pca", 2, f"{main_token}_{cls}",
                             "candtok_" + cls)
    bB, _ = _viz_kmeans_blocks(_pad_cols(np.asarray(cmB), 3),
                               list(range(len(memb))), K,
                               f"objcov_{main_token}_{cls}_{K}")

    def comemb(b):
        a = np.array([b[i] for i in range(len(memb))])
        return (a[:, None] == a[None, :])
    partition_same = (comemb(bA) == comemb(bB)).all()
    print("S02 partition identical across render order:", partition_same)
    # Document the finding; the high-D miner query still uses whatever partition
    # is live, so a different-but-valid partition is acceptable as long as it is
    # then frozen (S01). We assert it does not crash and yields K' blocks.
    assert len(set(bA.values())) >= 1 and len(set(bB.values())) >= 1


# ════════════════════════ SCENARIO 3 ════════════════════════
# Tiny class (1–3 objects): per-class table, K>n clamp, sparsity k clamp,
# selectbox default — must not crash, must warn.
def test_s03_tiny_class_guards(tmp_path):
    # coco8 has broccoli=1, giraffe=2 (tiny) → exercise both guards.
    at = run_harness({"main_dir": str(COCO8), "cov_token": "s03"})
    assert not at.exception, f"S03 crashed: {exc_summary(at)}"
    txt = all_text(at)
    print("S03 warnings:", [w.value for w in at.warning])
    # default pick is the thinnest class → tiny-class instability warning
    assert "個物件" in txt or "盲區判斷不穩" in txt or "物件太少" in txt, \
        "no tiny-class friendly guidance surfaced"
    # table + the class/projection/block selectors all rendered
    assert at.dataframe, "per-class sparsity table missing"


# ════════════════════════ SCENARIO 4 ════════════════════════
# Candidate folder has NO objects of the selected class → friendly
# "沒有此類別" message, no crash. (Build a candidate with only 'dog'.)
def test_s04_candidate_missing_selected_class(tmp_path):
    app, oe = _real_helpers()
    import streamlit as st
    # MAIN = coco8 (selected class will be a coco8 class). Candidate folder has a
    # class that does NOT appear in coco8's default pick.
    cand_root = make_yolo_dataset(
        tmp_path / "cand_dog",
        images=[next((COCO8 / "images").glob("*.jpg"))],
        labels={next((COCO8 / "images").glob("*.jpg")).stem: "0 0.5 0.5 0.4 0.4\n"},
        classes=["dog"])  # candidate only knows class "dog"
    at = run_harness({
        "main_dir": str(COCO8), "cov_token": "s04",
        "cand_folders": [str(cand_root)], "preembed_cand": True,
        "cand_offset": 0.0})
    assert not at.exception, f"S04 crashed: {exc_summary(at)}"
    txt = all_text(at)
    print("S04 text tail:", txt[-300:].replace("\n", " | "))
    assert ("沒有「" in txt or "沒有此類別" in txt or "換個資料夾" in txt
            or "提案" in txt or "先在上方" in txt), \
        "no friendly message when candidate lacks the selected class"


# ════════════════════════ SCENARIO 5 ════════════════════════
# Candidate folder is label-less / not a detection dataset → friendly warning
# from _objcov_embed_candidates, no crash.
def test_s05_candidate_not_a_detection_dataset(tmp_path):
    # images present, NO labels/ dir → discover finds images but no bbox.
    cand = tmp_path / "cand_nolabels"
    (cand / "images").mkdir(parents=True)
    for p in list((COCO8 / "images").glob("*.jpg"))[:2]:
        shutil.copy(p, cand / "images" / p.name)
    at = run_harness({
        "main_dir": str(COCO8), "cov_token": "s05",
        "cand_folders": [str(cand)], "preembed_cand": True})
    assert not at.exception, f"S05 crashed: {exc_summary(at)}"
    txt = all_text(at)
    print("S05 warnings:", [w.value for w in at.warning])
    assert ("找不到 bbox" in txt or "YOLO 偵測資料夾" in txt
            or "找不到影像" in txt), "no friendly label-less warning"


# ════════════════════════ SCENARIO 6 ════════════════════════
# All candidates beyond max_distance (radius*1.5) → mine_candidates empty →
# friendly "都太遠" message. Driven at logic level (deterministic distances).
def test_s06_all_candidates_too_far(tmp_path):
    from app import _cov_radius
    from completeness import mine_candidates
    rng = np.random.default_rng(6)
    # tight MAIN block → small radius
    memb = (rng.standard_normal((20, 16)) * 0.1 + 5.0).astype("float32")
    block_hi = memb.mean(0)
    radius = _cov_radius(memb, "s06tok")
    # candidates orthogonal/opposite → cosine distance ~ large (>> radius*1.5)
    cand = (rng.standard_normal((15, 16)) * 0.1 - 5.0).astype("float32")
    idx, dist = mine_candidates(cand, block_hi, k=24,
                                max_distance=radius * 1.5 if radius else None)
    print(f"S06 radius={radius:.4f} max_d={radius*1.5:.4f} "
          f"min_cand_dist={min((1 - (cand/np.linalg.norm(cand,axis=1,keepdims=True)) @ (block_hi/np.linalg.norm(block_hi))).tolist()):.4f} "
          f"-> kept={len(idx)}")
    assert idx == [], "candidates that are all too far were not filtered to empty"


# ════════════════════════ SCENARIO 7 ════════════════════════
# Class-NAME matching across folders with DIFFERENT classes.txt id ordering.
# MAIN id 0='person', candidate id 0='dog', id 1='person'. A bbox of candidate
# class id 1 must be labeled 'person' (match by NAME, not id), so it joins the
# MAIN 'person' analysis.
def test_s07_classname_matching_across_id_orderings(tmp_path):
    app, oe = _real_helpers()
    import streamlit as st
    from interaction import discover_yolo_objects
    from app import read_classes_txt, _cov_object_root

    img = next((COCO8 / "images").glob("*.jpg"))
    # candidate: its own classes.txt with REVERSED ids vs main
    cand_root = make_yolo_dataset(
        tmp_path / "cand_reorder", images=[img],
        labels={img.stem: "1 0.5 0.5 0.4 0.4\n"},  # id 1
        classes=["dog", "person"])  # id1 == 'person'
    recs = [{"path": str(cand_root / "images" / img.name)}]
    cnames = read_classes_txt(_cov_object_root(recs))
    meta = discover_yolo_objects([Path(recs[0]["path"])], cnames)
    print("S07 candidate cnames:", cnames, "-> labels:", [m["label"] for m in meta])
    assert cnames == ["dog", "person"]
    assert meta and meta[0]["label"] == "person", \
        "candidate class matched by id not by NAME (cross-folder id mismatch)"

    # Now feed MAIN whose classes.txt has 'person' at id 0; embedding the
    # candidate should produce a 'person' object that lands in the SAME label
    # bucket the renderer keys on.
    main_root = make_yolo_dataset(
        tmp_path / "main_person", images=[img],
        labels={img.stem: "0 0.5 0.5 0.4 0.4\n0 0.2 0.2 0.2 0.2\n"
                          "0 0.8 0.8 0.2 0.2\n0 0.3 0.7 0.2 0.2\n"},
        classes=["person", "dog"])
    at = run_harness({
        "main_dir": str(main_root), "cov_token": "s07",
        "cand_folders": [str(cand_root)], "preembed_cand": True,
        "cand_offset": 5.0})
    assert not at.exception, f"S07 render crashed: {exc_summary(at)}"
    cand = st_state_get(at, "objcov_cand")
    print("S07 projected candidate labels:", cand and cand.get("labels"))
    assert cand and "person" in cand["labels"], \
        "candidate person object not matched into the 'person' bucket by name"


def st_state_get(at: AppTest, key: str):
    try:
        return at.session_state[key]
    except (KeyError, Exception):
        return None


# ════════════════════════ SCENARIO 8 ════════════════════════
# Object-policy consistency: _objcov_embed_candidates must pass the SAME policy
# (pad/target_res/head) to _crop_and_embed_objects that MAIN used. Verify by
# spying on the call.
def test_s08_policy_consistency_main_vs_candidate(tmp_path, monkeypatch):
    app, oe = _real_helpers()
    import streamlit as st
    st.session_state.clear()

    img = next((COCO8 / "images").glob("*.jpg"))
    cand_root = make_yolo_dataset(
        tmp_path / "cand_pol", images=[img],
        labels={img.stem: "0 0.5 0.5 0.4 0.4\n"}, classes=["person"])

    captured = {}

    def fake_crop(records, model, class_names, pad, *, base_token,
                  session_key="_cov_obj", crops_subdir="object_crops",
                  spinner="裁切物件", policy=None, progress_cb=None):
        captured["pad"] = pad
        captured["policy"] = dict(policy or {})
        recs = [{"path": records[0]["path"], "image_path": Path(records[0]["path"]),
                 "label": "person", "class_id": 0, "bbox": (0.5, 0.5, 0.4, 0.4),
                 "obj_index": 0, "split": ""}]
        return recs, np.ones((1, 16), "float32"), "tok"

    monkeypatch.setattr(app, "_crop_and_embed_objects", fake_crop)
    custom = {"pad": 0.33, "target_res": 196, "head": "meanpool"}
    app._objcov_embed_candidates([cand_root], "dinov2_vits14", custom, ["person"])
    print("S08 captured policy:", captured)
    assert captured.get("policy") == custom, "candidate policy != MAIN policy"
    assert abs(captured.get("pad", -1) - 0.33) < 1e-9, "candidate pad mismatch"


# ════════════════════════ SCENARIO 9 ════════════════════════
# Switching class / projection / dim / K does not crash and re-derives blocks.
# Drive every projection method + 2D/3D + K extremes against coco128 (rich).
@pytest.mark.parametrize("proj,dim,K", [
    ("pca", "2D", 3), ("pca", "3D", 20), ("umap", "2D", 8), ("tsne", "3D", 5),
])
def test_s09_switch_projection_dim_k(proj, dim, K):
    src = COCO128 if COCO128.exists() else COCO8
    at = run_harness({"main_dir": str(src), "cov_token": f"s09_{proj}_{dim}_{K}"},
                     timeout=240)
    if at.exception:
        pytest.fail(f"S09 {proj}/{dim}/K{K} crashed: {exc_summary(at)}")
    # set the controls and rerun
    for sb in at.selectbox:
        if sb.label == "投影方法":
            sb.set_value({"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}[proj])
    for rd in at.radio:
        if rd.label == "維度":
            rd.set_value(dim)
    for sl in at.slider:
        if sl.label == "區塊數 K":
            sl.set_value(K)
    at.run()
    assert not at.exception, f"S09 after switch {proj}/{dim}/K{K}: {exc_summary(at)}"
    print(f"S09 {proj}/{dim}/K{K} OK; selectboxes={[s.label for s in at.selectbox]}")


# ════════════════════════ SCENARIO 10 ════════════════════════
# "加入清單" never writes into the candidate dataset dir; cart + curation log
# updated instead. Drive _batch_add/_append_curation_entry the way the button
# does and assert the dataset folder is byte-identical before/after.
def test_s10_add_to_cart_no_dataset_write(tmp_path, monkeypatch):
    app, oe = _real_helpers()
    import streamlit as st
    st.session_state.clear()
    st.session_state["viz_export_list"] = {}
    st.session_state["viz_manifest"] = {}

    img = next((COCO8 / "images").glob("*.jpg"))
    cand_root = make_yolo_dataset(
        tmp_path / "cand_cart", images=[img],
        labels={img.stem: "0 0.5 0.5 0.4 0.4\n0 0.2 0.2 0.2 0.2\n"},
        classes=["person"])

    # snapshot dataset dir (paths + sizes + mtimes)
    def snapshot(root: Path):
        return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted(root.rglob("*")) if p.is_file()}
    before = snapshot(cand_root)

    # redirect curation log to tmp so we can assert it WAS written
    log = tmp_path / "curation_log.jsonl"
    monkeypatch.setattr(app, "_CURATION_LOG", log)

    cand_recs = [{"path": str(cand_root / "images" / img.name), "label": "person",
                  "split": ""}]
    app._batch_add(cand_recs, [0], source="objcov_backfill", scores={0: 0.1})
    app._append_curation_entry(cand_recs, [0], reason="backfill person #block0")

    after = snapshot(cand_root)
    print("S10 dataset unchanged:", before == after,
          "| cart size:", len(st.session_state["viz_export_list"]),
          "| log exists:", log.exists())
    assert before == after, "WRITE into candidate dataset dir during add-to-cart!"
    assert len(st.session_state["viz_export_list"]) == 1, "cart not updated"
    assert log.exists() and log.read_text(encoding="utf-8").strip(), \
        "curation log not written"


# ════════════════════════ SCENARIO 11 (bonus) ════════════════════════
# Non-detection MAIN dataset → friendly info, returns early (no crash).
def test_s11_non_detection_main(tmp_path):
    # classifier layout: <root>/<class>/<imgs>, no labels/.
    root = tmp_path / "clf"
    for cls in ("cats", "dogs"):
        d = root / cls
        d.mkdir(parents=True)
        for p in list((COCO8 / "images").glob("*.jpg"))[:2]:
            shutil.copy(p, d / f"{cls}_{p.name}")
    at = run_harness({"main_dir": str(root), "cov_token": "s11",
                      "seed_main_cache": False})
    assert not at.exception, f"S11 crashed: {exc_summary(at)}"
    txt = all_text(at)
    print("S11 info:", [i.value for i in at.info])
    assert "YOLO 偵測資料夾" in txt or "偵測資料集" in txt, \
        "non-detection MAIN did not return the friendly info early"


# ════════════════════════ SCENARIO 12 ════════════════════════
# End-to-end POSITIVE path: candidates that genuinely sit on a MAIN block →
# mine_candidates returns hits → the real "🛒 加入清單" button renders → CLICK it
# → cart grows, dataset dir untouched (the rendered-button twin of S10).
def test_s12_e2e_add_button_no_dataset_write(monkeypatch):
    src = COCO128 if COCO128.exists() else COCO8
    near = "person" if COCO128.exists() else "orange"
    at = run_harness({"main_dir": str(src), "cov_token": "s12",
                      "cand_from_main_class": near, "main_seed": 5}, timeout=300)
    assert not at.exception, f"S12 first render: {exc_summary(at)}"
    # select the class the synthetic candidates belong to
    for sb in at.selectbox:
        if sb.label.startswith("選類別"):
            sb.set_value(near)
    at.run()
    assert not at.exception, f"S12 after class pick: {exc_summary(at)}"

    add = [b for b in at.button if "加入清單" in b.label]
    caps = [c.value for c in at.caption]
    mined = [c for c in caps if "接近" in c]
    print("S12 mined caption:", mined[:1], "| add btn:", [b.label for b in add])
    assert add, "no add-to-cart button despite candidates on the block"
    assert mined, "mining did not report any near candidates"

    # snapshot the candidate crop source (here: a MAIN image) — clicking add must
    # not write anywhere near the dataset. We assert the export_list grows and no
    # new files appear under the MAIN dataset root.
    def snap(root: Path):
        return {str(p.relative_to(root)): p.stat().st_size
                for p in sorted(root.rglob("*")) if p.is_file()}
    before = snap(src)
    add[0].click()
    at.run()
    assert not at.exception, f"S12 after add click: {exc_summary(at)}"
    after = snap(src)
    cart = []
    try:
        cart = list(at.session_state["viz_export_list"].keys())
    except Exception:
        pass
    print("S12 dataset unchanged:", before == after, "| cart size:", len(cart))
    assert before == after, "add-to-cart WROTE into the MAIN dataset dir!"
    assert len(cart) >= 1, "cart did not grow after clicking 加入清單"
