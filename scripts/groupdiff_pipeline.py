"""groupdiff_pipeline:雙資料夾 → patch 特徵(復用)→ 20/21 → 群差異結果包 + 匯出。

設計:3_Architect_Design/22_groupdiff_pipeline.md(M19,Tier B)。
復用:discover_whole_images(M8)/ embed_objects_patch(01,含 .lv_cache 快取)/
render_heatmap(05)/ safe_io / object_eval.list_images、dataset_cache_dir。
grid 對齊走多數決(非多數 grid 跳過 + 對齊警示);匯出絕不寫來源資料夾。
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _progress(cb, frac: float, text: str) -> None:
    if cb:
        cb(min(float(frac), 1.0), text)


def _load_group(folder, name, *, model, target_res, extractor, cache_dir,
                progress, base_frac, span):
    from interaction import discover_whole_images
    from object_eval import list_images
    from patch_features import embed_objects_patch
    from safe_io import partition_readable

    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"資料夾不存在:{folder}")
    readable, bad_files = partition_readable(list_images(folder))
    if len(readable) < 2:
        raise ValueError(f"資料夾可讀影像不足(至少 2 張):{folder}(現 {len(readable)} 張)")
    meta = discover_whole_images(readable, label=name)
    pf = embed_objects_patch(
        meta, model, target_res=target_res, cache_dir=cache_dir, extractor=extractor,
        progress=lambda d, t: _progress(progress, base_frac + span * d / max(t, 1),
                                        f"{name} patch 特徵 {d}/{t}"))
    return meta, pf, len(bad_files)


def _aspect_ratios(meta, keep) -> list[float]:
    out = []
    for i in keep:
        try:
            with Image.open(meta[i]["image_path"]) as im:   # 只讀 header,便宜
                w, h = im.size
            out.append(w / max(h, 1))
        except OSError:
            pass
    return out


def run_groupdiff(good_dir, bad_dir, *, model: str = "dinov2_vits14",
                  target_res: int = 224, n_perm: int = 200, alpha: float = 0.05,
                  stab_q: float = 0.95, top_k: int = 5, n_reps: int = 3,
                  seed: int = 0, extractor=None, cache_root=None,
                  progress=None) -> dict:
    from diff_regions import extract_regions, pick_representatives, region_summary
    from group_patch_stats import group_diff_stats

    good_dir, bad_dir = Path(good_dir), Path(bad_dir)
    if cache_root is None:
        from object_eval import dataset_cache_dir
        cdir_g = dataset_cache_dir(good_dir, f"groupdiff_{model}_{target_res}")
        cdir_b = dataset_cache_dir(bad_dir, f"groupdiff_{model}_{target_res}")
    else:
        cdir_g = Path(cache_root) / "good"
        cdir_b = Path(cache_root) / "bad"

    _progress(progress, 0.01, "列舉影像…")
    meta_g, pf_g, unread_g = _load_group(
        good_dir, "good", model=model, target_res=target_res, extractor=extractor,
        cache_dir=cdir_g, progress=progress, base_frac=0.02, span=0.40)
    meta_b, pf_b, unread_b = _load_group(
        bad_dir, "bad", model=model, target_res=target_res, extractor=extractor,
        cache_dir=cdir_b, progress=progress, base_frac=0.42, span=0.40)

    # grid 多數決:同 grid 才可逐位置比較;壞圖 (1,1) fallback 也在此自然剔除
    grids = [tuple(p["grid"]) for p in pf_g] + [tuple(p["grid"]) for p in pf_b]
    main_grid = Counter(grids).most_common(1)[0][0]
    keep_g = [i for i, p in enumerate(pf_g) if tuple(p["grid"]) == main_grid]
    keep_b = [i for i, p in enumerate(pf_b) if tuple(p["grid"]) == main_grid]
    n_grid_skip = (len(pf_g) - len(keep_g)) + (len(pf_b) - len(keep_b))

    warn_msgs: list[str] = []
    if unread_g + unread_b:
        warn_msgs.append(f"已略過 {unread_g + unread_b} 個壞檔(無法解碼)")
    if n_grid_skip:
        warn_msgs.append(f"{n_grid_skip} 張影像因尺寸/長寬比與多數不一致被跳過"
                         "(patch 網格不相容,位置無法對齊)")
    if len(keep_g) < 2 or len(keep_b) < 2:
        raise ValueError(f"grid 對齊後樣本不足:good {len(keep_g)} / bad {len(keep_b)}"
                         "(需各 ≥2)")

    alignment_warning = bool(n_grid_skip)
    ratios = _aspect_ratios(meta_g, keep_g) + _aspect_ratios(meta_b, keep_b)
    if ratios and max(ratios) / max(min(ratios), 1e-9) > 1.05:
        alignment_warning = True
        warn_msgs.append("兩組影像長寬比差異超過 5%——位置對齊假設可能不成立,結果僅供參考")

    _progress(progress, 0.86, "群組差異統計(permutation 校準)…")
    feats_g = np.stack([pf_g[i]["feats"].astype(np.float32) for i in keep_g])
    feats_b = np.stack([pf_b[i]["feats"].astype(np.float32) for i in keep_b])
    stats = group_diff_stats(feats_g, feats_b, n_perm=n_perm, alpha=alpha,
                             stab_q=stab_q, seed=seed)

    gh, gw = main_grid
    z_map = stats["z"].reshape(gh, gw)
    score_map = stats["score"].reshape(gh, gw)
    stability_map = stats["stability"].reshape(gh, gw)

    _progress(progress, 0.93, "抽取差異區域…")
    good_paths = [str(meta_g[i]["image_path"]) for i in keep_g]
    bad_paths = [str(meta_b[i]["image_path"]) for i in keep_b]
    regions = extract_regions(z_map, stability_map,
                              z_thresh=stats["threshold_z"], top_k=top_k)
    for r in regions:
        reps = pick_representatives(r, stats["dist_good"], stats["dist_bad"],
                                    n_each=n_reps)
        summ = region_summary(r)
        r0, c0, r1, c1 = r["bbox_grid"]
        r.update(text=summ["text"], pct=summ["pct"],
                 good_paths=[good_paths[i] for i in reps["good_idx"]],
                 bad_paths=[bad_paths[i] for i in reps["bad_idx"]],
                 bbox_px=(c0 / gw, r0 / gh, c1 / gw, r1 / gh))

    # 熱圖底圖代表:Good=最典型;Bad=差異區貢獻最大(無區域時退全 patch)
    typical_good = int(np.argmin(stats["dist_good"].mean(axis=1)))
    if regions:
        idxs = sorted({i for r in regions for i in r["patch_idx"]})
        showcase_bad = int(np.argmax(stats["dist_bad"][:, idxs].mean(axis=1)))
    else:
        showcase_bad = int(np.argmax(stats["dist_bad"].mean(axis=1)))

    _progress(progress, 1.0, "完成")
    return {
        "grid": (gh, gw), "score_map": score_map, "z_map": z_map,
        "stability_map": stability_map,
        "verdict": stats["verdict"], "p_global": stats["p_global"],
        "threshold_z": stats["threshold_z"],
        "regions": regions,
        "good_paths": good_paths, "bad_paths": bad_paths,
        "good_dir": str(good_dir), "bad_dir": str(bad_dir),
        "base_good": good_paths[typical_good], "base_bad": bad_paths[showcase_bad],
        "gmean_good": feats_g.mean(axis=1), "gmean_bad": feats_b.mean(axis=1),
        "n_good": len(keep_g), "n_bad": len(keep_b),
        "skipped": {"unreadable": unread_g + unread_b, "grid": n_grid_skip},
        "alignment_warning": alignment_warning, "warn_msgs": warn_msgs,
        "meta": {"model": model, "target_res": target_res, "n_perm": n_perm,
                 "alpha": alpha, "stab_q": stab_q, "seed": seed, "top_k": top_k},
    }


def region_compare_images(result: dict, region: dict, *, n_each: int = 3,
                          margin: float = 0.5) -> dict:
    """代表圖 → 區域+margin 上下文放大圖(紅框標區域本體)。壞路徑跳過。"""
    from safe_io import safe_open_image
    out: dict = {"good": [], "bad": []}
    x0, y0, x1, y1 = region["bbox_px"]
    for kind in ("good", "bad"):
        for p in region.get(f"{kind}_paths", [])[:n_each]:
            im = safe_open_image(p)
            if im is None:
                continue
            W, H = im.size
            bx0, by0, bx1, by1 = x0 * W, y0 * H, x1 * W, y1 * H
            mx, my = (bx1 - bx0) * margin, (by1 - by0) * margin
            cx0 = int(max(0, np.floor(bx0 - mx)))
            cy0 = int(max(0, np.floor(by0 - my)))
            cx1 = int(min(W, np.ceil(bx1 + mx)))
            cy1 = int(min(H, np.ceil(by1 + my)))
            crop = im.convert("RGB").crop((cx0, cy0, cx1, cy1))
            draw = ImageDraw.Draw(crop)
            draw.rectangle([bx0 - cx0, by0 - cy0, bx1 - cx0 - 1, by1 - cy0 - 1],
                           outline=(255, 32, 32),
                           width=max(2, crop.size[0] // 100))
            out[kind].append(crop)
    return out


def pca2d(gmean_good, gmean_bad) -> dict:
    """非監督旁證投影(不吃群標籤——監督投影會人工推開兩群造成假分離)。"""
    g = np.asarray(gmean_good, dtype=float)
    b = np.asarray(gmean_bad, dtype=float)
    X = np.concatenate([g, b], axis=0)
    if X.shape[0] < 3:
        raise ValueError(f"PCA 需至少 3 個樣本(現 {X.shape[0]})")
    Xc = X - X.mean(axis=0, keepdims=True)
    _U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    xy = Xc @ Vt[:2].T
    var = S ** 2
    return {"xy_good": xy[:len(g)], "xy_bad": xy[len(g):],
            "evr": (var / max(var.sum(), 1e-12))[:2]}


def export_report(result: dict, out_dir) -> dict:
    """熱圖 PNG + regions.csv + 對照圖 + summary.json → out_dir。絕不寫來源。"""
    from anomaly_heatmap import render_heatmap
    from safe_io import safe_open_image

    out = Path(out_dir).resolve()
    for label in ("good_dir", "bad_dir"):
        src = Path(result[label]).resolve()
        if out == src or src in out.parents:
            raise ValueError(f"匯出目錄不得在來源資料夾內:{out}(來源 {src})")
    out.mkdir(parents=True, exist_ok=True)

    files: list[str] = []
    suffix = "" if result["verdict"] else "_未達顯著"
    vmax = float(max(float(np.max(result["z_map"])), result["threshold_z"], 1e-9))
    for tag, key in (("good", "base_good"), ("bad", "base_bad")):
        base = safe_open_image(result[key])
        if base is None:
            continue
        img = render_heatmap(result["z_map"], base, alpha=0.55, vmin=0.0, vmax=vmax)
        f = out / f"heatmap_{tag}{suffix}.png"
        img.save(f)
        files.append(str(f))

    fcsv = out / "regions.csv"
    with open(fcsv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "peak_z", "mean_z", "stability_pct", "text",
                    "bbox_px_x0", "bbox_px_y0", "bbox_px_x1", "bbox_px_y1"])
        for r in result["regions"]:
            x0, y0, x1, y1 = r["bbox_px"]
            w.writerow([r["rank"], f"{r['peak_z']:.4f}", f"{r['mean_z']:.4f}",
                        r["pct"], r["text"],
                        f"{x0:.6f}", f"{y0:.6f}", f"{x1:.6f}", f"{y1:.6f}"])
    files.append(str(fcsv))

    for r in result["regions"]:
        imgs = region_compare_images(result, r, n_each=3)
        tiles = imgs["good"] + imgs["bad"]
        if not tiles:
            continue
        h = max(t.size[1] for t in tiles)
        tiles = [t.resize((max(1, round(t.size[0] * h / t.size[1])), h))
                 for t in tiles]
        gap = 6
        canvas = Image.new("RGB",
                           (sum(t.size[0] for t in tiles) + gap * (len(tiles) - 1), h),
                           (245, 245, 245))
        x = 0
        for t in tiles:
            canvas.paste(t, (x, 0))
            x += t.size[0] + gap
        f = out / f"region_{r['rank']:02d}.png"
        canvas.save(f)
        files.append(str(f))

    fsum = out / "summary.json"
    fsum.write_text(json.dumps(
        {"verdict": result["verdict"], "p_global": result["p_global"],
         "threshold_z": result["threshold_z"],
         "n_good": result["n_good"], "n_bad": result["n_bad"],
         "skipped": result["skipped"],
         "alignment_warning": result["alignment_warning"],
         "warn_msgs": result["warn_msgs"], "meta": result["meta"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(str(fsum))
    return {"out_dir": str(out), "files": files}
