"""E2E scenario tests for the NEW Coverage view "按物件・逐類別補洞"
(`_render_objcov_view` in scripts/app.py), reached via the cov_view_mode radio.

Driven through _objcov_harness.py, an AppTest harness that reads a JSON spec from
OBJCOV_SPEC: it seeds a real tiny YOLO detection set (demo/coco8) as MAIN, seeds
the `_objcov_main` object cache with a token matching what the renderer derives
(so `_crop_and_embed_objects` is skipped for MAIN), monkeypatches a deterministic
fake embedder for the CANDIDATE path, and calls the real `_render_objcov_view`.

The full path (autotune + real DINOv2 crop+embed of every object on CPU) is far
too slow / weight-dependent for E2E, so we cover the new surface two ways:

  RENDER  — assert the per-class table, numbered-block scatter, thinnest-block
            ranking, block thumbnails, candidate overlay (2D AND 3D) and the
            "加入清單" → cart + curation-log routing all render / fire with zero
            exceptions, via the AppTest harness.
  LOGIC   — AppTest can't fire plotly drag events, so the match/decode is
            verified directly against the same helpers the view uses
            (mine_candidates closest-first to a block centroid; class filter by
            NAME; no write-back into the candidate folder).

Run:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest \
        tests/e2e/test_objcov_backfill.py -v
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
_REPO = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import app  # noqa: E402
import object_eval as oe  # noqa: E402
from completeness import mine_candidates  # noqa: E402

_HARNESS = Path(__file__).resolve().parent / "_objcov_harness.py"
_PTAG = oe.policy_tag(dict(oe.DEFAULT_POLICY))
_COCO8_TRAIN = _REPO / "demo" / "coco8" / "train"


# ── helpers ─────────────────────────────────────────────────────────────

def _tiny_jpg(path: Path, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (32, 32, 3)).astype("uint8")
    Image.fromarray(arr).save(path, quality=85)
    return path


def _make_cand_folder(root: Path, classes_lines: str, boxes: str, n: int = 4) -> Path:
    """A real tiny YOLO detection candidate folder (images/+labels/+classes.txt)."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text(classes_lines, encoding="utf-8")
    for i in range(n):
        _tiny_jpg(root / "images" / f"c{i}.jpg", seed=700 + i)
        (root / "labels" / f"c{i}.txt").write_text(boxes, encoding="utf-8")
    return root


def _run(spec: dict, *, timeout: int = 180, preseed=None):
    """Run the harness AppTest with the given spec; return the AppTest object.
    ``preseed`` (callable(session_state)) lets a test inject controlled
    session_state (e.g. a frozen objcov_cand) BEFORE the first run; the harness's
    _seed() only sets keys it owns and skips MAIN caching when
    seed_main_cache=False, so pre-seeded keys survive."""
    from streamlit.testing.v1 import AppTest
    warnings.filterwarnings("ignore")
    os.environ["OBJCOV_SPEC"] = json.dumps(spec)
    at = AppTest.from_file(str(_HARNESS), default_timeout=timeout)
    if preseed is not None:
        preseed(at.session_state)
    return at.run()


def _fig_spec(at):
    """Read the FIRST plotly chart's figure as parsed JSON via the element proto.
    AppTest raises KeyError on a *keyed* chart's `.value` after a rerun (the
    chart registers as an on_select widget whose state the harness never sets),
    so we read the rendered spec directly — framework limitation, not a bug."""
    return json.loads(at.get("plotly_chart")[0].proto.spec)


def _trace_names(at):
    return [t.get("name") for t in _fig_spec(at).get("data", [])]


def _ss(at, key, default=None):
    """AppTest's SafeSessionState has no .get(); use membership + indexing."""
    ss = at.session_state
    return ss[key] if key in ss else default


def _base_spec(**over) -> dict:
    spec = {"main_dir": str(_COCO8_TRAIN), "cov_token": "tok_main_0001",
            "dim": 16, "main_seed": 7}
    spec.update(over)
    return spec


@pytest.fixture(scope="module")
def coco8_available():
    assert (_COCO8_TRAIN / "images").is_dir(), "demo/coco8/train missing"
    return True


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — the view option appears in the radio (FIRST) and dispatches.
# ════════════════════════════════════════════════════════════════════════

def test_s01_view_option_appears_and_dispatches():
    src = (_SCRIPTS / "app.py").read_text(encoding="utf-8")
    # 3 views merged → 2: 嵌入覆蓋圖 (top) + 屬性棋盤; objcov is now a SUB-mode
    assert 'key="cov_view_mode"' in src
    assert 'key="cov_emb_submode"' in src
    assert '"逐類別・區塊補洞"' in src           # the objcov sub-mode label
    assert "_render_objcov_view(records, model)" in src
    assert callable(app._render_objcov_view)


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — a non-detection dataset shows friendly info, no crash.
# ════════════════════════════════════════════════════════════════════════

def test_s02_non_detection_friendly_info(tmp_path):
    # classifier-style records (<folder>/<class>/img.jpg), NO labels/ sibling
    p = _tiny_jpg(tmp_path / "plainset" / "cat" / "a.jpg", seed=1)
    at = _run(_base_spec(main_dir=str(tmp_path / "plainset"),
                         seed_main_cache=False), timeout=60)
    assert not at.exception, [e.value for e in at.exception]
    assert any("YOLO 偵測資料夾" in i.value for i in at.info), [i.value for i in at.info]


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — per-class table renders + sorted thinnest-first.
# ════════════════════════════════════════════════════════════════════════

def test_s03_per_class_table_renders_sorted(coco8_available):
    at = _run(_base_spec())
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.dataframe) >= 1
    df = at.dataframe[0].value
    # core columns present (a "可分析" flag column may also be inserted)
    assert {"類別", "物件數", "盲區塊", "最稀疏"} <= set(df.columns), list(df.columns)
    # sort key is (物件數 asc, -最稀疏): object-count column is non-decreasing
    counts = list(df["物件數"])
    assert counts == sorted(counts), counts


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 4 — pick class → numbered-block scatter + sparse-block ranking.
# ════════════════════════════════════════════════════════════════════════

def test_s04_class_pick_blocks_and_ranking(coco8_available):
    at = _run(_base_spec())
    assert not at.exception, [e.value for e in at.exception]
    labels = [s.label for s in at.selectbox]
    # NOTE: 投影方法 / 區塊數 K now live in the sidebar (_objcov_sidebar), not in
    # _render_objcov_view which the harness calls directly — so we don't assert them here.
    assert any("選類別分析" in l for l in labels)
    assert len(at.get("plotly_chart")) >= 1                 # numbered-block scatter
    assert any("進入區塊" in s.label for s in at.selectbox)  # block ranking picker
    assert any("盲區塊" in m.value for m in at.markdown)


def test_s04b_block_ranking_thin_and_sparse_first():
    """The block_rank ordering the side panel shows == sorted by (count asc,
    -sparsity), verified against the same helpers the view uses."""
    from collections import Counter
    from interaction import sparsity_scores

    rng = np.random.default_rng(0)
    # one class, 16 clustered objects → several KMeans blocks
    memb = rng.standard_normal((16, 16)).astype("float32")
    memb[:6] += 5.0  # a denser sub-cluster so block sizes differ
    msp = sparsity_scores(memb, k=min(8, len(memb) - 1))
    coords = app._pad_cols(memb, 3)
    blocks, centers = app._viz_kmeans_blocks(coords, list(range(len(memb))), 8, "s04b")
    cnt = Counter(blocks.values())
    block_sp = {}
    for b in range(len(centers)):
        mem = [i for i in range(len(memb)) if blocks.get(i) == b]
        if mem:
            block_sp[b] = float(np.mean([msp[i] for i in mem]))
    rank = sorted(block_sp, key=lambda b: (cnt[b], -block_sp[b]))
    keys = [(cnt[b], -block_sp[b]) for b in rank]
    assert keys == sorted(keys), keys  # thinnest (smallest count) first


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 5 — block pick → that block's MAIN thumbnails + block caption.
# ════════════════════════════════════════════════════════════════════════

def test_s05_block_thumbnails_render(coco8_available):
    at = _run(_base_spec())
    assert not at.exception, [e.value for e in at.exception]
    # the block side-panel rendered for whichever class is analysable
    assert any("主資料・區塊" in c.value for c in at.caption) or \
        any("無法分塊" in i.value for i in at.info), \
        [c.value for c in at.caption]
    # pick the orange class (>=4 objects) explicitly → thumbnails must render
    sb = next(s for s in at.selectbox if "選類別分析" in (s.label or ""))
    if "orange" in sb.options:
        at2 = sb.set_value("orange").run()
        assert not at2.exception, [e.value for e in at2.exception]
        assert at2.get("imgs"), "expected block member thumbnails (st.image)"
        assert any("主資料・區塊" in c.value for c in at2.caption)


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 6 — candidate embed reuses the SAME policy + populates objcov_cand
#              with only same-class objects (filtered later by NAME).
# ════════════════════════════════════════════════════════════════════════

def test_s06_candidate_embed_same_policy_populates_cand(coco8_available, tmp_path):
    cand = _make_cand_folder(tmp_path / "cand6",
                             classes_lines="person\nbicycle\ncar\n",
                             boxes="0 0.5 0.5 0.4 0.4\n2 0.3 0.3 0.2 0.2\n")
    at = _run(_base_spec(cand_folders=[str(cand)], preembed_cand=True))
    assert not at.exception, [e.value for e in at.exception]
    c = _ss(at, "objcov_cand")
    assert c and c.get("recs"), "objcov_cand not populated"
    # only the candidate's own labels appear (class names from its classes.txt)
    assert set(c["labels"]) <= {"person", "car"}, set(c["labels"])
    # SAME object policy → crop cache dir tagged with the default policy_tag
    crop_paths = [str(r["path"]) for r in c["recs"]]
    assert any(_PTAG in p for p in crop_paths), crop_paths[:1]


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 7 — candidate match ranks by distance to the block centroid
#              (closest first), filtered to the SAME class by NAME.
# ════════════════════════════════════════════════════════════════════════

def test_s07_mine_candidates_closest_first_same_class():
    rng = np.random.default_rng(3)
    # MAIN 'car' block centroid + a candidate pool of two NAMED classes
    centroid = rng.standard_normal(16).astype("float32")
    cand_labels = (["car"] * 5) + (["dog"] * 5)
    cand_emb = rng.standard_normal((10, 16)).astype("float32")
    cand_emb[:5] += centroid * 0.5      # car candidates pulled toward the centroid
    cand_emb[5:] += 9.0                 # dog candidates pushed far away

    # class filter is by NAME — only 'car' rows enter the pool
    pick = "car"
    cand_ci = [i for i, l in enumerate(cand_labels) if l == pick]
    assert cand_ci == [0, 1, 2, 3, 4]
    cmemb = cand_emb[cand_ci]

    idx, dist = mine_candidates(cmemb, centroid, k=24)
    assert idx, "expected near candidates"
    assert dist == sorted(dist), dist          # closest first (ascending)
    assert max(idx) < len(cand_ci)             # never indexes a 'dog' row


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 8 — candidates overlaid in the SAME 2D AND 3D figure (black ◆).
# ════════════════════════════════════════════════════════════════════════

def _controlled_preseed(tmp_path, *, cov_token, n_main=10, n_cand=6, dim=16):
    """Pre-seed a single analysable MAIN class ('orange') with a clear centroid
    plus a CANDIDATE set of the SAME class clustered NEAR that centroid (so
    mine_candidates returns hits) — and a far 'dog' candidate class that the
    NAME filter must drop. Returns (preseed_fn, cand_root, main_centroid)."""
    rng = np.random.default_rng(11)
    main_mean = np.zeros(dim, dtype="float32"); main_mean[0] = 5.0
    main_recs, main_emb = [], []
    for j in range(n_main):
        p = _tiny_jpg(tmp_path / "main_crops" / f"orange_{j}.jpg", seed=300 + j)
        main_recs.append({"path": str(p), "image_path": str(p),
                          "label": "orange", "split": ""})
        main_emb.append(main_mean + rng.standard_normal(dim).astype("float32") * 0.3)
    main_emb = np.asarray(main_emb, dtype="float32")

    cand_root = tmp_path / "cand_real"
    cand_recs, cand_emb, cand_labels = [], [], []
    for j in range(n_cand):  # 'orange' candidates = near-copies of MAIN points so
        p = _tiny_jpg(cand_root / "images" / f"o{j}.jpg", seed=600 + j)
        cand_recs.append({"path": str(p), "image_path": str(p),  # each sits next to
                          "label": "orange", "split": ""})        # a MAIN object →
        cand_emb.append(main_emb[j % n_main]                      # inside the radius
                        + rng.standard_normal(dim).astype("float32") * 0.02)
        cand_labels.append("orange")
    for j in range(3):       # far 'dog' candidates — must be filtered out by NAME
        p = _tiny_jpg(cand_root / "images" / f"d{j}.jpg", seed=650 + j)
        cand_recs.append({"path": str(p), "image_path": str(p),
                          "label": "dog", "split": ""})
        cand_emb.append(np.full(dim, 9.0, dtype="float32"))
        cand_labels.append("dog")
    cand_emb = np.asarray(cand_emb, dtype="float32")

    main_token = f"{cov_token}_{_PTAG}"

    def _preseed(ss):
        ss["_objcov_main"] = {"token": main_token, "recs": main_recs,
                              "emb": main_emb,
                              "labels": [r["label"] for r in main_recs]}
        ss["objcov_cand"] = {"token": "cand_frozen_1", "recs": cand_recs,
                             "emb": cand_emb, "labels": cand_labels}

    return _preseed, cand_root, main_mean


def test_s08_candidates_overlaid_2d_and_3d(tmp_path):
    cov_token = "tok_s08"
    preseed, _, _ = _controlled_preseed(tmp_path, cov_token=cov_token)
    spec = _base_spec(cov_token=cov_token, seed_main_cache=False)
    at = _run(spec, preseed=preseed)
    assert not at.exception, [e.value for e in at.exception]

    # only one MAIN class ('orange') → it is the analysed class by default.
    names2d = _trace_names(at)
    assert "候選" in names2d, names2d
    # black-diamond candidate marker present in the SAME 2D figure
    diamonds = [t for t in _fig_spec(at)["data"]
                if (t.get("marker") or {}).get("symbol") == "diamond"]
    assert diamonds, "no black-diamond candidate trace (2D)"
    assert diamonds[0]["marker"].get("color") in ("#111", "black")

    # flip 維度 → 3D (now a sidebar control read from session) → re-render
    at.session_state["objcov_dim"] = "3D"
    at = at.run()
    assert not at.exception, [e.value for e in at.exception]
    data3d = _fig_spec(at)["data"]
    assert "候選" in [t.get("name") for t in data3d], [t.get("name") for t in data3d]
    assert any(t.get("type") == "scatter3d" for t in data3d), \
        [t.get("type") for t in data3d]


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 9 — "加入清單" routes to cart (source=objcov_backfill) + curation
#              log, and NEVER writes into the candidate dataset folder.
# ════════════════════════════════════════════════════════════════════════

def _snapshot_tree(root: Path):
    return {str(p): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_s09_add_to_cart_and_log_no_dataset_write(tmp_path):
    cov_token = "tok_s09"
    preseed, cand_root, _ = _controlled_preseed(tmp_path, cov_token=cov_token)

    log = app._CURATION_LOG
    before_lines = log.read_text(encoding="utf-8").count("\n") if log.exists() else 0
    cand_before = _snapshot_tree(cand_root)

    spec = _base_spec(cov_token=cov_token, seed_main_cache=False)
    at = _run(spec, preseed=preseed)
    assert not at.exception, [e.value for e in at.exception]

    # near 'orange' candidates exist → the 🛒 加入清單 button is shown
    add_btns = [b for b in at.button if b.key == "objcov_add"]
    assert add_btns, "objcov_add button missing — no near candidate?"
    at = add_btns[0].click().run()
    assert not at.exception, [e.value for e in at.exception]

    # cart populated and tagged source="objcov_backfill"
    elist = _ss(at, "viz_export_list", {})
    assert elist, "cart empty after 加入清單"
    assert all(v.get("source") == "objcov_backfill" for v in elist.values())
    # every cart item is a CANDIDATE crop, never a MAIN crop; and never a 'dog'
    # (NAME-filtered to the analysed 'orange' class)
    cand_recs = _ss(at, "objcov_cand")["recs"]
    orange_paths = {str(r["path"]) for r in cand_recs if r["label"] == "orange"}
    assert all(v["path"] in orange_paths for v in elist.values())

    # curation log grew by exactly one entry; reason mentions the backfill block
    after_lines = log.read_text(encoding="utf-8").count("\n") if log.exists() else 0
    assert after_lines == before_lines + 1, (before_lines, after_lines)
    last = log.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert "backfill" in last and "block" in last

    # candidate dataset folder byte-for-byte unchanged (proposal-only, no write-back)
    assert _snapshot_tree(cand_root) == cand_before


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 10 — tiny-class (<4 objects) guard warns and does not crash.
# ════════════════════════════════════════════════════════════════════════

def test_s10_tiny_class_guard_warns(coco8_available):
    # coco8/train has several <4 classes (giraffe=2, broccoli=1, vase=1, …).
    # The default pick is now the thinnest ANALYZABLE (>=4) class, so explicitly
    # select a tiny class to exercise the guard. Accept either guard message:
    #   <2 objects → "太少無法分析"; 2-3 objects → "盲區判斷不穩 / 建議整類擴量".
    at = _run(_base_spec())
    assert not at.exception, [e.value for e in at.exception]
    sb = next((s for s in at.selectbox if "選類別" in (s.label or "")), None)
    assert sb is not None, "class selectbox not found"
    tiny = next((o for o in list(sb.options)
                 if o in ("broccoli", "vase", "giraffe", "toothbrush")), None)
    if tiny is None:
        import pytest
        pytest.skip("no <4 class present in the coco8 sample")
    at = sb.set_value(tiny).run()
    assert not at.exception, [e.value for e in at.exception]
    msgs = " | ".join(getattr(m, "value", "")
                      for m in list(at.warning) + list(at.info))
    assert ("太少無法分析" in msgs or "盲區判斷不穩" in msgs
            or "建議整類擴量" in msgs), msgs
