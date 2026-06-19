"""Auto-tune object-embedding settings, and store/reuse them per dataset fingerprint.

Framework-free (no streamlit) so it runs as a CLI *and* is importable by the app:

    python scripts/object_eval.py --dataset C:/code/dataset/indoor/test
    python scripts/object_eval.py --dataset <dir> --write-profile indoor-furniture
    python scripts/object_eval.py --dataset <dir> --match     # just show the closest profile

What it does
------------
- ``dataset_fingerprint`` summarises a detection dataset by object short-side-px and
  aspect-ratio quantiles + class/object counts — the few factors that actually drive the
  best crop/resize settings, so "similar fingerprint" ≈ "same best settings".
- ``run_autotune`` embeds the objects under a handful of candidate policies
  (pad × target_res × head) and ranks them by macro kNN-purity, broken out per size bucket.
- ``object_policy_profiles.json`` (mirrors the signal_gate.json pattern) stores
  {name, fingerprint, policy, metrics}; ``match_profile`` returns the closest by fingerprint.

Mirrors the calibration pattern in signal_strength.py / calibrate_signal_gate.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from interaction import crop_bbox, discover_yolo_objects, yolo_label_path_for
from _utils import load_model

_REPO = Path(__file__).resolve().parent.parent
PROFILES_PATH = Path(
    os.environ.get("LV_OBJECT_PROFILES_CONFIG") or (_REPO / "object_policy_profiles.json")
)

# short-side source pixels → bucket name (the regime where policy choice matters most)
SIZE_BUCKETS = [(0, 32, "<32"), (32, 96, "32-96"), (96, 224, "96-224"), (224, 1 << 30, ">224")]

# A small staged screen (not a full grid): the two architectural knobs (head) ×
# context (pad) × detail (resolution). All heads stay D=384 (cls / meanpool).
DEFAULT_CANDIDATES = [
    {"pad": 0.12, "target_res": 224, "head": "cls"},       # current default
    {"pad": 0.12, "target_res": 224, "head": "meanpool"},
    {"pad": 0.25, "target_res": 224, "head": "cls"},
    {"pad": 0.25, "target_res": 224, "head": "meanpool"},
    {"pad": 0.12, "target_res": 336, "head": "cls"},
    {"pad": 0.12, "target_res": 336, "head": "meanpool"},
]
DEFAULT_POLICY = {"pad": 0.12, "target_res": 224, "head": "cls"}


# ── geometry / crop ──────────────────────────────────────────────────────────
def _adaptive_pad_px(bbox, iw: int, ih: int, pad: float) -> int:
    long_px = max(bbox[2] * iw, bbox[3] * ih)
    return int(min(64, max(8, round(pad * long_px))))


def policy_tag(policy: dict) -> str:
    return (f"pad{int(round(policy['pad'] * 100))}_r{policy['target_res']}"
            f"_{policy['head']}")


# ── dataset discovery ────────────────────────────────────────────────────────
_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(root: Path) -> list[Path]:
    base = root / "images" if (root / "images").is_dir() else root
    return sorted(p for p in base.rglob("*") if p.suffix.lower() in _IMG_EXT)


def classes_for(root: Path) -> list[str] | None:
    """classes.txt (root or parent) → else data.yaml `names:` (parent)."""
    for folder in (root, root.parent):
        f = folder / "classes.txt"
        if f.exists():
            lines = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                return lines
    for folder in (root, root.parent):
        y = folder / "data.yaml"
        if y.exists():
            names, grab = [], False
            for ln in y.read_text(encoding="utf-8").splitlines():
                s = ln.strip()
                if s.startswith("names:"):
                    grab = True
                    continue
                if grab and s.startswith("- "):
                    names.append(s[2:].strip())
                elif grab and s and not s.startswith("- "):
                    break
            if names:
                return names
    return None


# ── fingerprint + matching ───────────────────────────────────────────────────
def dataset_fingerprint(image_paths, class_names=None) -> dict:
    """Object short-side-px and aspect-ratio quantiles + class/object counts.
    Cheap: reads image headers (no pixel decode), parses YOLO labels."""
    meta = discover_yolo_objects([Path(p) for p in image_paths], class_names)
    sizes: dict = {}
    short, ar = [], []
    for m in meta:
        ip = m["image_path"]
        if ip not in sizes:
            try:
                with Image.open(ip) as im:
                    sizes[ip] = im.size
            except OSError:
                sizes[ip] = None
        if not sizes[ip]:
            continue
        iw, ih = sizes[ip]
        wpx, hpx = m["bbox"][2] * iw, m["bbox"][3] * ih
        short.append(min(wpx, hpx))
        ar.append(wpx / max(hpx, 1e-6))
    short, ar = np.array(short), np.array(ar)
    q = (0.1, 0.5, 0.9)
    return {
        "n_objects": int(len(short)),
        "n_classes": int(len({m["class_id"] for m in meta})),
        "size_q": [float(np.quantile(short, x)) for x in q] if len(short) else [0, 0, 0],
        "ar_q": [float(np.quantile(ar, x)) for x in q] if len(ar) else [1, 1, 1],
    }


def _fp_vec(fp: dict) -> np.ndarray:
    sq = [math.log10(max(s, 1.0)) for s in fp["size_q"]]
    aq = [math.log10(max(a, 1e-3)) for a in fp["ar_q"]]
    return np.array(sq + aq + [math.log10(max(fp["n_classes"], 1))])


def fingerprint_similarity(a: dict, b: dict) -> float:
    """1.0 = identical; decays with distance in log-feature space."""
    return float(np.exp(-np.linalg.norm(_fp_vec(a) - _fp_vec(b))))


# ── profile store (mirrors signal_gate.json) ─────────────────────────────────
def load_profiles(path: Path = PROFILES_PATH) -> list[dict]:
    if not Path(path).exists():
        return []
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("profiles", [])
    except (OSError, ValueError):
        return []


def save_profile(name: str, fingerprint: dict, policy: dict, metrics: dict,
                 path: Path = PROFILES_PATH) -> Path:
    profs = [p for p in load_profiles(path) if p.get("name") != name]
    profs.append({
        "name": name, "fingerprint": fingerprint, "policy": policy,
        "metrics": metrics, "created": datetime.now().isoformat(timespec="seconds"),
    })
    Path(path).write_text(json.dumps({"profiles": profs}, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return Path(path)


def match_profile(fingerprint: dict, profiles=None):
    """Return (best_profile, similarity) or (None, 0.0)."""
    profiles = load_profiles() if profiles is None else profiles
    best, best_sim = None, 0.0
    for p in profiles:
        s = fingerprint_similarity(fingerprint, p["fingerprint"])
        if s > best_sim:
            best, best_sim = p, s
    return best, best_sim


# ── embedding + metrics ──────────────────────────────────────────────────────
def embed_objects(meta, model, policy, *, cache_path: Path | None = None,
                  progress=None):
    """Embed each object from an in-memory full-res crop under ``policy``.
    Returns (emb [N,D], short_px [N]). Caches to npz keyed by object identity.
    ``progress(done, total)`` is called per object (once at total on a cache hit)."""
    keys = [f"{Path(m['image_path']).stem}__{m['obj_index']}" for m in meta]
    # incremental + checkpoint: reuse already-computed objects (key = stem__obj),
    # save every 200 → a crash resumes from the last checkpoint, and adding data
    # only computes the new objects (not the whole set).
    cached_emb: dict = {}
    cached_short: dict = {}
    if cache_path and Path(cache_path).exists():
        try:
            with np.load(str(cache_path), allow_pickle=False) as d:  # close handle (Windows replace)
                ks = d["keys"].tolist()
                _emb = d["emb"]
                cached_emb = {k: _emb[i] for i, k in enumerate(ks)}
                if "short_px" in d.files:
                    _sp = d["short_px"]
                    cached_short = {k: float(_sp[i]) for i, k in enumerate(ks)}
        except (OSError, ValueError, KeyError):
            cached_emb, cached_short = {}, {}

    def _flush() -> None:
        if not (cache_path and cached_emb):
            return
        ks = list(cached_emb.keys())
        arr = np.stack([cached_emb[k] for k in ks]).astype(np.float32)
        sp = np.array([cached_short.get(k, 0.0) for k in ks])
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_name(cp.stem + ".part.npz")  # 須以 .npz 結尾(np.savez 否則自動補)
        np.savez(str(tmp), emb=arr, keys=np.array(ks), short_px=sp)
        tmp.replace(cp)  # atomic

    embed_fn = None  # loaded lazily → a full cache hit never touches torch
    pad = policy["pad"]
    cur_ip, cur = None, None
    total = len(meta)
    done = sum(1 for k in keys if k in cached_emb)
    if progress:
        progress(done, total)
    since = 0
    for _i, m in enumerate(meta):
        k = keys[_i]
        if k in cached_emb:
            continue
        if embed_fn is None:
            embed_fn = load_model(model, keep_aspect=True,
                                  target_res=policy["target_res"], head=policy["head"])
        ip = m["image_path"]
        if str(ip) != cur_ip:
            try:
                cur = Image.open(ip).convert("RGB")
            except OSError:
                cur = None
            cur_ip = str(ip)
        if cur is None:
            cached_emb[k] = np.zeros(384, dtype=np.float32)
            cached_short[k] = 0.0
        else:
            iw, ih = cur.size
            b = m["bbox"]
            cached_short[k] = float(min(b[2] * iw, b[3] * ih))
            crop = crop_bbox(cur, *b, pad_px=_adaptive_pad_px(b, iw, ih, pad))
            cached_emb[k] = np.asarray(embed_fn(crop), dtype=np.float32)
        done += 1
        since += 1
        if progress:
            progress(done, total)
        if since >= 200:
            _flush()
            since = 0
    if since:
        _flush()
    emb = np.stack([cached_emb[k] for k in keys]) if keys else np.zeros((0, 1))
    short = np.array([cached_short.get(k, 0.0) for k in keys])
    return emb, short


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def score_embeddings(emb: np.ndarray, labels, short_px: np.ndarray, k: int = 5) -> dict:
    """Macro kNN-purity@k (weak label = class) overall + per size bucket. Higher = better."""
    labels = np.asarray(labels)
    n = len(labels)
    k = min(k, max(1, n - 1))
    X = _l2n(emb)
    sims = X @ X.T
    np.fill_diagonal(sims, -np.inf)
    nn = np.argsort(-sims, axis=1)[:, :k]
    per = (labels[nn] == labels[:, None]).mean(axis=1)  # purity per object

    def _macro(mask):
        ls = labels[mask]
        cl = np.unique(ls)
        return float(np.mean([per[mask][ls == c].mean() for c in cl])) if len(cl) else None

    buckets = {}
    for lo, hi, name in SIZE_BUCKETS:
        m = (short_px >= lo) & (short_px < hi)
        buckets[name] = {"n": int(m.sum()),
                         "purity": (float(per[m].mean()) if m.any() else None)}
    return {"macro_purity": _macro(np.ones(n, bool)),
            "purity": float(per.mean()), "k": k, "buckets": buckets}


def neighbor_confusion(emb: np.ndarray, labels, k: int = 10) -> dict:
    """kNN class confusion + per-object suggested label — computed in FULL-D
    (not the 2D/3D projection), so it isn't fooled by a crowded scatter.

    Returns:
      classes    : sorted class names
      confusion  : (C,C); row c = mean fraction of class-c objects' neighbours that
                   are each class. Diagonal = cohesion/separation; off-diagonal =
                   how often class c is surrounded by class c'.
      counts     : objects per class
      per_object : [{idx, own, suggested, suggested_frac, own_frac, disagreement,
                    mismatch}] — mismatch=True ⇒ neighbours mostly a DIFFERENT class
                    (a label-error / hard-case candidate; suggested = that class).
    """
    labels = np.asarray(labels)
    n = len(labels)
    classes = sorted(set(labels.tolist()))
    cidx = {c: i for i, c in enumerate(classes)}
    C = len(classes)
    if n < 2 or C == 0:
        return {"classes": classes, "confusion": np.zeros((C, C)),
                "counts": np.zeros(C, int), "per_object": []}
    k = min(k, n - 1)
    X = _l2n(emb)
    sims = X @ X.T
    np.fill_diagonal(sims, -np.inf)
    nn = np.argsort(-sims, axis=1)[:, :k]
    conf = np.zeros((C, C))
    counts = np.zeros(C)
    per_object = []
    for i in range(n):
        dist = np.zeros(C)
        for nl in labels[nn[i]]:
            dist[cidx[nl]] += 1
        dist /= max(k, 1)
        own = cidx[labels[i]]
        conf[own] += dist
        counts[own] += 1
        sug = int(np.argmax(dist))
        per_object.append({
            "idx": i, "own": str(labels[i]), "suggested": classes[sug],
            "suggested_frac": float(dist[sug]), "own_frac": float(dist[own]),
            "disagreement": float(1.0 - dist[own]),
            "mismatch": classes[sug] != labels[i],
        })
    conf = conf / np.clip(counts[:, None], 1.0, None)
    return {"classes": classes, "confusion": conf,
            "counts": counts.astype(int), "per_object": per_object}


def run_autotune(image_paths, class_names, *, model="dinov2_vits14",
                 candidates=None, cap=800, cache_dir: Path | None = None,
                 progress=None):
    """Embed objects under each candidate policy, rank by macro kNN-purity.
    Returns (rows sorted best-first, best_policy, fingerprint)."""
    candidates = candidates or DEFAULT_CANDIDATES
    image_paths = [Path(p) for p in image_paths]
    fingerprint = dataset_fingerprint(image_paths, class_names)
    meta = discover_yolo_objects(image_paths, class_names)
    if not meta:
        return [], None, fingerprint
    if len(meta) > cap:  # subsample evenly for interactivity
        idx = np.linspace(0, len(meta) - 1, cap).astype(int)
        meta = [meta[i] for i in idx]
    labels = [m["class_id"] for m in meta]
    rows = []
    for ci, pol in enumerate(candidates):
        cache = (Path(cache_dir) / f"autotune_{policy_tag(pol)}_{model}.npz"
                 if cache_dir else None)
        ocb = ((lambda d, t, _ci=ci, _pol=pol: progress(_ci, len(candidates), _pol, d, t))
               if progress else None)
        emb, short = embed_objects(meta, model, pol, cache_path=cache, progress=ocb)
        rows.append({"policy": pol, **score_embeddings(emb, labels, short)})
    rows.sort(key=lambda r: (r["macro_purity"] is not None, r["macro_purity"] or 0.0),
              reverse=True)
    best = rows[0]["policy"] if rows else DEFAULT_POLICY
    return rows, best, fingerprint


# ── CLI ──────────────────────────────────────────────────────────────────────
def _fmt_pct(x):
    return "  –  " if x is None else f"{x * 100:5.1f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="detection dataset dir (has images/+labels/)")
    ap.add_argument("--model", default="dinov2_vits14")
    ap.add_argument("--write-profile", default=None, metavar="NAME")
    ap.add_argument("--match", action="store_true", help="only show closest saved profile")
    ap.add_argument("--cap", type=int, default=800)
    args = ap.parse_args(argv)

    root = Path(args.dataset)
    imgs = list_images(root)
    if not imgs:
        print(f"[err] no images under {root}")
        return 1
    class_names = classes_for(root)
    fp = dataset_fingerprint(imgs, class_names)
    print(f"dataset      : {root}")
    print(f"fingerprint  : {fp['n_objects']} objects, {fp['n_classes']} classes | "
          f"short-px q10/50/90 = {[round(s) for s in fp['size_q']]} | "
          f"AR q10/50/90 = {[round(a, 2) for a in fp['ar_q']]}")

    best_p, sim = match_profile(fp)
    if best_p:
        print(f"closest saved: 『{best_p['name']}』 similarity {sim:.2f} "
              f"→ policy {best_p['policy']}")
    else:
        print("closest saved: (none yet)")
    if args.match:
        return 0

    cache_dir = root / "object_crops" / "_autotune"
    print(f"\nrunning autotune ({len(DEFAULT_CANDIDATES)} policies, cap {args.cap})…")

    def _prog(ci, n, pol, od, ot):
        if od >= ot:  # one line per finished policy
            print(f"  [{ci + 1}/{n}] {policy_tag(pol)}")

    rows, best, fp = run_autotune(imgs, class_names, model=args.model,
                                  cap=args.cap, cache_dir=cache_dir, progress=_prog)
    if not rows:
        print("[err] no objects (labels/ empty?)")
        return 1

    print("\npolicy                         macro   <32  32-96 96-224  >224")
    for r in rows:
        b = r["buckets"]
        print(f"  {policy_tag(r['policy']):<26} {_fmt_pct(r['macro_purity'])}  "
              f"{_fmt_pct(b['<32']['purity'])} {_fmt_pct(b['32-96']['purity'])} "
              f"{_fmt_pct(b['96-224']['purity'])} {_fmt_pct(b['>224']['purity'])}")
    print(f"\nBEST → {policy_tag(best)}  {best}")

    if args.write_profile:
        path = save_profile(args.write_profile, fp, best, rows[0])
        print(f"[ok] saved profile 『{args.write_profile}』 → {path}")
    else:
        print("(pass --write-profile NAME to save this as a reusable profile)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
