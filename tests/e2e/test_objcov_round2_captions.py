"""Round-2 STRICT verification: the three round-1 deductions were about MISSING
user-facing notices. The strict-scenario suite only checks behavior, so this
file asserts the actual captions/columns RENDER in the live AppTest flow.

Run IN ISOLATION:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest \
      tests/e2e/test_objcov_round2_captions.py -v
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
_REPO = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import app  # noqa: E402
import object_eval as oe  # noqa: E402

_HARNESS = Path(__file__).resolve().parent / "_objcov_harness.py"
_PTAG = oe.policy_tag(dict(oe.DEFAULT_POLICY))
_COCO8_TRAIN = _REPO / "demo" / "coco8" / "train"


def _tiny_jpg(path: Path, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (32, 32, 3)).astype("uint8")
    Image.fromarray(arr).save(path, quality=85)
    return path


def _run(spec, *, preseed=None, timeout=180):
    from streamlit.testing.v1 import AppTest
    warnings.filterwarnings("ignore")
    os.environ["OBJCOV_SPEC"] = json.dumps(spec)
    at = AppTest.from_file(str(_HARNESS), default_timeout=timeout)
    if preseed is not None:
        preseed(at.session_state)
    return at.run()


def _base_spec(**over):
    spec = {"main_dir": str(_COCO8_TRAIN), "cov_token": "tok_cap", "dim": 16,
            "main_seed": 7}
    spec.update(over)
    return spec


def _preseed_with_cand(tmp_path, cov_token, *, n_main=12, n_cand=6, dim=16,
                       cand_offset=0.02):
    rng = np.random.default_rng(11)
    main_mean = np.zeros(dim, dtype="float32"); main_mean[0] = 5.0
    main_recs, main_emb = [], []
    for j in range(n_main):
        p = _tiny_jpg(tmp_path / "m" / f"orange_{j}.jpg", seed=300 + j)
        main_recs.append({"path": str(p), "image_path": str(p),
                          "label": "orange", "split": ""})
        main_emb.append(main_mean + rng.standard_normal(dim).astype("float32") * 0.3)
    main_emb = np.asarray(main_emb, dtype="float32")
    cand_recs, cand_emb, cand_labels = [], [], []
    for j in range(n_cand):
        p = _tiny_jpg(tmp_path / "c" / f"o{j}.jpg", seed=600 + j)
        cand_recs.append({"path": str(p), "image_path": str(p),
                          "label": "orange", "split": ""})
        cand_emb.append(main_emb[j % n_main]
                        + rng.standard_normal(dim).astype("float32") * cand_offset)
        cand_labels.append("orange")
    cand_emb = np.asarray(cand_emb, dtype="float32")
    main_token = f"{cov_token}_{_PTAG}"

    def _preseed(ss):
        ss["_objcov_main"] = {"token": main_token, "recs": main_recs,
                              "emb": main_emb,
                              "labels": [r["label"] for r in main_recs]}
        ss["objcov_cand"] = {"token": "cand_frozen", "recs": cand_recs,
                             "emb": cand_emb, "labels": cand_labels}
    return _preseed


def _caps(at):
    return [c.value for c in at.caption]


# Fix 1 (S3): t-SNE + candidates → orange notice RENDERS.
def test_fix1_tsne_candidate_notice_renders(tmp_path):
    cov_token = "tok_cap1"
    preseed = _preseed_with_cand(tmp_path, cov_token)
    spec = _base_spec(cov_token=cov_token, seed_main_cache=False)

    def pre(ss):
        preseed(ss)
        ss["objcov_proj"] = "t-SNE"

    at = _run(spec, preseed=pre)
    assert not at.exception, [e.value for e in at.exception]
    caps = " | ".join(_caps(at))
    assert "t-SNE 無法投影候選" in caps, f"missing t-SNE fallback notice: {caps}"


# Negative control: t-SNE WITHOUT candidates → NO false fallback notice.
def test_fix1_no_notice_without_candidates(tmp_path):
    cov_token = "tok_cap1b"
    preseed = _preseed_with_cand(tmp_path, cov_token)
    spec = _base_spec(cov_token=cov_token, seed_main_cache=False)

    def pre(ss):
        preseed(ss)
        ss["objcov_proj"] = "t-SNE"
        del ss["objcov_cand"]

    at = _run(spec, preseed=pre)
    assert not at.exception, [e.value for e in at.exception]
    assert "t-SNE 無法投影候選" not in " | ".join(_caps(at)), \
        "false fallback notice with no candidates"


# Fix 2 (S8): per-class 可分析 column + tiny-class caption RENDER.
def test_fix2_analysable_column_and_tiny_caption(tmp_path):
    at = _run(_base_spec(cov_token="tok_cap2"))
    assert not at.exception, [e.value for e in at.exception]
    df = at.dataframe[0].value
    assert "可分析" in list(df.columns), f"可分析 column missing: {list(df.columns)}"
    vals = set(df["可分析"].tolist())
    assert vals <= {"✅", "⚠<4"}, vals
    # column reflects the 物件數>=4 rule exactly
    for _, r in df.iterrows():
        assert r["可分析"] == ("✅" if r["物件數"] >= 4 else "⚠<4"), r.to_dict()
    caps = " | ".join(_caps(at))
    assert "無法分塊分析" in caps, f"missing tiny-class caption: {caps}"


# Fix 3 (S5): supervised projection caveat caption RENDERS for both methods.
def test_fix3_supervised_caveat_renders(tmp_path):
    cov_token = "tok_cap3"
    preseed = _preseed_with_cand(tmp_path, cov_token)
    spec = _base_spec(cov_token=cov_token, seed_main_cache=False)
    for method in ("監督UMAP", "LDA(監督)"):
        def pre(ss, m=method):
            preseed(ss)
            ss["objcov_proj"] = m
        at = _run(spec, preseed=pre)
        assert not at.exception, (method, [e.value for e in at.exception])
        caps = " | ".join(_caps(at))
        assert "監督投影" in caps and "不代表兩者真的不同" in caps, \
            f"{method}: missing supervised caveat: {caps}"


# Negative control: unsupervised PCA → NO supervised caveat.
def test_fix3_no_caveat_for_pca(tmp_path):
    cov_token = "tok_cap3b"
    preseed = _preseed_with_cand(tmp_path, cov_token)
    spec = _base_spec(cov_token=cov_token, seed_main_cache=False)

    def pre(ss):
        preseed(ss)
        ss["objcov_proj"] = "PCA"

    at = _run(spec, preseed=pre)
    assert not at.exception, [e.value for e in at.exception]
    assert "監督投影" not in " | ".join(_caps(at)), "false supervised caveat under PCA"
