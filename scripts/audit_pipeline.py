"""audit_pipeline:資料夾(+metadata CSV)→ 六訊號 → 體檢報告包 + 匯出。

設計:3_Architect_Design/25_audit_pipeline.md(M20,Tier B)。
復用:discover_images(_classifier)/ partition_readable / manifest.update_manifest
(sha256/phash 增量,**只取值不寫檔**)/ find_duplicate_pairs_* / compute_outlier_scores。
embedding 走**內容定址快取**(sha 為鍵,同內容只算一次);匯出絕不寫來源資料夾。
"""
from __future__ import annotations

import csv
import html as _html
import json
from datetime import datetime
from pathlib import Path

import numpy as np

_SPLIT_NAMES = {"train", "val", "valid", "test"}


def _progress(cb, frac: float, text: str) -> None:
    if cb:
        cb(min(float(frac), 1.0), text)


def _discover(root: Path):
    from label_formats import folder_has_annotations
    from visualize_embeddings import discover_images, discover_images_classifier

    subs = [d for d in sorted(root.iterdir())
            if d.is_dir() and d.name.lower() in _SPLIT_NAMES]
    folders = subs if subs else [root]
    records: list[dict] = []
    for f in folders:
        if folder_has_annotations(f) or ((f / "images").is_dir()
                                         and (f / "labels").is_dir()):
            recs = discover_images([f], [])
        else:
            recs = discover_images_classifier([f], allow_flat=True)
        records.extend(recs)
    return folders, records


def run_audit(root, *, model: str = "dinov2_vits14", metadata_csv=None,
              embed_fn=None, use_embedding: bool = True,
              phash_max_hamming: int = 4, emb_max_distance: float = 0.05,
              outlier_k: int = 5, top_n: int = 10, max_pairs: int = 200,
              cache_root=None, progress=None) -> dict:
    from interaction import (compute_outlier_scores,
                             find_duplicate_pairs_embedding,
                             find_duplicate_pairs_phash)
    from manifest import rel_key, update_manifest
    from safe_io import partition_readable

    roots = [Path(p) for p in (root if isinstance(root, (list, tuple, set)) else [root])]
    missing = [str(p) for p in roots if not p.is_dir()]
    if not roots or missing:
        raise ValueError("資料夾不存在:" + "、".join(missing))
    _progress(progress, 0.02, "掃描佈局…")
    folders: list[Path] = []
    records: list[dict] = []
    seen_paths: set[str] = set()
    for source in roots:
        source_folders, source_records = _discover(source)
        folders.extend(source_folders)
        for record in source_records:
            token = str(Path(record["path"]).resolve())
            if token not in seen_paths:
                seen_paths.add(token)
                records.append(record)
    if not records:
        raise ValueError("所選資料夾內找不到影像")

    _progress(progress, 0.05, "檢查影像可讀性…")
    ok_paths, bad_paths = partition_readable([r["path"] for r in records])
    ok_set = {str(Path(p)) for p in ok_paths}
    recs = [r for r in records if str(Path(r["path"])) in ok_set]
    n_unreadable = len(bad_paths)
    if len(recs) < 2:
        raise ValueError(f"可讀影像不足(至少 2 張；現 {len(recs)} 張)")

    # sha256/phash:manifest 增量計算(只取值,不寫檔——不動 viz 的簿記)
    sha_of_path: dict[str, str] = {}
    ph_of_path: dict[str, str | None] = {}
    n_folders = len(folders)
    for fi, f in enumerate(folders):
        frecs = []
        for r in recs:
            try:
                rel_key(f, r["path"])
            except ValueError:
                continue
            frecs.append({"path": r["path"], "split": r["split"],
                          "label": r.get("label", "")})
        if not frecs:
            continue
        entries = update_manifest(
            f, frecs,
            progress_cb=lambda d, t, fi=fi, f=f: _progress(
                progress, 0.08 + 0.27 * (fi + d / max(t, 1)) / n_folders,
                f"內容雜湊 {f.name} {d}/{t}"))
        for r in frecs:
            e = entries.get(rel_key(f, r["path"]))
            if e:
                p = str(Path(r["path"]))
                sha_of_path[p] = e["sha256"]
                ph_of_path[p] = e.get("phash")

    paths = [str(Path(r["path"])) for r in recs]
    splits = [r["split"] for r in recs]
    labels = [r.get("label", "") or "" for r in recs]
    shas = [sha_of_path[p] for p in paths]
    phashes = [ph_of_path.get(p) for p in paths]

    # embedding:內容定址快取(sha 為鍵),同內容只算一次
    emb = None
    if use_embedding:
        cdir = Path(cache_root) if cache_root else None
        if cdir is None:
            from object_eval import dataset_cache_dir
            cdir = dataset_cache_dir(roots[0], f"audit_{model}")
        cdir.mkdir(parents=True, exist_ok=True)
        cfile = cdir / "emb_cache.npz"
        cache: dict[str, np.ndarray] = {}
        if cfile.exists():
            try:
                with np.load(str(cfile), allow_pickle=False) as d:
                    keys = [str(k) for k in d["keys"]]
                    vecs = d["vecs"].astype(np.float32)
                    cache = {k: vecs[i] for i, k in enumerate(keys)}
            except (OSError, ValueError, KeyError):
                cache = {}
        first_path: dict[str, str] = {}
        for p, s in zip(paths, shas):
            first_path.setdefault(s, p)
        todo = [s for s in sorted(set(shas)) if s not in cache]
        if todo:
            if embed_fn is None:
                from _utils import load_model
                embed_fn = load_model(model)
            for k, s in enumerate(todo):
                v = np.asarray(embed_fn(Path(first_path[s])),
                               dtype=np.float32).ravel()
                v /= max(float(np.linalg.norm(v)), 1e-12)
                cache[s] = v
                _progress(progress, 0.36 + 0.40 * (k + 1) / len(todo),
                          f"embedding {k + 1}/{len(todo)}(同內容只算一次)")
            keys = sorted(cache)
            np.savez(str(cfile), keys=np.array(keys),
                     vecs=np.stack([cache[k] for k in keys]).astype(np.float32))
        emb = np.stack([cache[s] for s in shas])

    _progress(progress, 0.80, "重複/洩漏配對…")
    phash_pairs = find_duplicate_pairs_phash(
        phashes, max_hamming=phash_max_hamming, max_pairs=max_pairs)
    leak_pairs = find_duplicate_pairs_phash(
        phashes, max_hamming=phash_max_hamming, splits=splits,
        cross_split_only=True, max_pairs=max_pairs)
    emb_pairs = None
    outlier_scores = None
    if emb is not None:
        emb_pairs = find_duplicate_pairs_embedding(
            emb, max_distance=emb_max_distance, max_pairs=max_pairs)
        _progress(progress, 0.86, "離群分數…")
        outlier_scores = compute_outlier_scores(
            emb, emb, k=min(outlier_k, len(emb) - 1),
            candidates_in_reference=True)

    meta_info = None
    meta_groups = None
    if metadata_csv is not None:
        _progress(progress, 0.90, "掛 metadata(CSV)…")
        from meta_join import group_counts, join_metadata, load_metadata_csv
        rows = load_metadata_csv(metadata_csv)
        meta_info = join_metadata(rows, paths,
                                  {i: s for i, s in enumerate(shas)})
        meta_groups = {f: group_counts(meta_info["per_image"], len(paths), f)
                       for f in meta_info["fields"]}

    _progress(progress, 0.95, "彙總報告…")
    from audit_report import build_report
    report = build_report({
        "paths": paths, "splits": splits, "labels": labels, "shas": shas,
        "n_unreadable": n_unreadable,
        "phash_pairs": phash_pairs, "leak_pairs": leak_pairs,
        "emb_pairs": emb_pairs, "outlier_scores": outlier_scores,
        "meta": meta_info, "meta_groups": meta_groups,
        "params": {"model": model, "phash_max_hamming": phash_max_hamming,
                   "emb_max_distance": emb_max_distance,
                   "outlier_k": outlier_k, "top_n": top_n,
                   "max_pairs": max_pairs, "use_embedding": use_embedding},
    })
    _progress(progress, 1.0, "完成")
    return {"report": report,
            "records": [{"path": p, "split": s, "label": lb, "sha": h}
                        for p, s, lb, h in zip(paths, splits, labels, shas)],
            "n_images": len(paths), "n_unreadable": n_unreadable,
            "root": str(roots[0]), "roots": [str(p) for p in roots],
            "meta_info": meta_info}


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


def export_audit(result: dict, out_dir) -> dict:
    out = Path(out_dir).resolve()
    for source in result.get("roots", [result["root"]]):
        src = Path(source).resolve()
        if out == src or src in out.parents:
            raise ValueError(f"匯出目錄不得在來源資料夾內:{out}(來源 {src})")
    out.mkdir(parents=True, exist_ok=True)
    report = result["report"]
    files: list[str] = []

    jf = out / "report.json"
    jf.write_text(json.dumps({"sections": report["sections"],
                              "totals": report["totals"],
                              "root": result["root"],
                              "roots": result.get("roots", [result["root"]])},
                             ensure_ascii=False, indent=2, default=_jsonable),
                  encoding="utf-8")
    files.append(str(jf))

    recs = {r["path"]: r for r in result["records"]}

    def _rl(p):
        r = recs.get(p, {})
        return r.get("split", ""), r.get("label", "")

    icsv = out / "issues.csv"
    with open(icsv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["issue_type", "path", "split", "label", "detail"])
        for s in report["sections"]:
            if s["key"] == "exact_dup":
                for g in s["items"]:
                    for p in g["paths"][1:]:
                        sp, lb = _rl(p)
                        w.writerow(["exact_dup", p, sp, lb,
                                    f"sha={g['sha'][:12]} 複本"])
            elif s["key"] == "leakage":
                for it in s["items"]:
                    _sp, lb = _rl(it["path_i"])
                    w.writerow(["leakage", it["path_i"], it["split_i"], lb,
                                f"與 {it['path_j']}({it['split_j']})相同/近似"])
            elif s["key"] == "label_conflict":
                for g in s["items"]:
                    for p in g["paths"]:
                        sp, lb = _rl(p)
                        w.writerow(["label_conflict", p, sp, lb,
                                    "/".join(g["labels"])])
            elif s["key"] == "outliers":
                for it in s["items"]:
                    sp, lb = _rl(it["path"])
                    w.writerow(["outlier", it["path"], sp, lb,
                                f"score={it['score']:.4f}"])
            elif s["key"] == "metadata" and result.get("meta_info"):
                per = result["meta_info"]["per_image"]
                for i, r in enumerate(result["records"]):
                    if i not in per:
                        w.writerow(["meta_unmatched", r["path"], r["split"],
                                    r["label"], "CSV 無對應列"])
    files.append(str(icsv))

    hf = out / "report.html"
    style = ("body{font-family:system-ui;margin:24px;max-width:1100px}"
             "table{border-collapse:collapse;margin:6px 0}"
             "td,th{border:1px solid #ccc;padding:2px 8px;font-size:12px}"
             "section{margin-bottom:18px}.m{color:#888;font-size:12px}"
             "h2{margin:4px 0}")
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<title>資料體檢報告</title><style>", style, "</style></head><body>",
             "<h1>🩺 資料體檢報告</h1>",
              f"<p>{_html.escape('、'.join(result.get('roots', [result['root']])))} · "
             f"{datetime.now().isoformat(timespec='seconds')} · "
             f"共 {report['totals']['n_images']} 張可讀影像</p>"]
    for s in report["sections"]:
        val = "—" if s["value"] is None else s["value"]
        parts.append(f"<section><h2>{_html.escape(s['title'])} · <b>{val}</b></h2>"
                     f"<p>{_html.escape(s['text'])}</p>"
                     f"<p class='m'>{_html.escape(s['method'])}</p>")
        items = s["items"][:50]
        if items:
            heads = sorted({k for it in items for k in it})
            parts.append("<table><tr>" + "".join(
                f"<th>{_html.escape(h)}</th>" for h in heads) + "</tr>")
            for it in items:
                parts.append("<tr>" + "".join(
                    f"<td>{_html.escape(str(it.get(h, '')))}</td>"
                    for h in heads) + "</tr>")
            parts.append("</table>")
            if len(s["items"]) > 50:
                parts.append(f"<p class='m'>(僅列前 50/{len(s['items'])})</p>")
        parts.append("</section>")
    parts.append("</body></html>")
    hf.write_text("".join(parts), encoding="utf-8")
    files.append(str(hf))
    return {"out_dir": str(out), "files": files}
