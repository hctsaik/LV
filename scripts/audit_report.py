"""audit_report:各訊號原始結果 → 量化報告資料結構(數字+白話+method line)。

設計:3_Architect_Design/24_audit_report.md(M20)。純函式無 I/O;不下總判詞
(v1 無校準基準,誠實地只給數字與出處)。method line 是 G3「數字要帶出處」的契約,
格式逐字釘死;顯示截斷一律明講(no silent caps)。
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def _add(secs, key, title, value, text, method, items=None):
    secs.append({"key": key, "title": title, "value": value,
                 "text": text, "method": method, "items": items or []})


def build_report(inputs: dict) -> dict:
    paths = list(inputs["paths"])
    splits = list(inputs["splits"])
    labels = list(inputs["labels"])
    shas = list(inputs["shas"])
    n = len(paths)
    if not (len(splits) == len(labels) == len(shas) == n):
        raise ValueError(f"輸入長度不一致:paths {n} / splits {len(splits)} / "
                         f"labels {len(labels)} / shas {len(shas)}")
    P = inputs["params"]
    n_unread = int(inputs["n_unreadable"])
    split_counts = dict(Counter(splits))

    def cap_note(pairs) -> str:
        return (f"(已達顯示上限 {P['max_pairs']} 對,實際可能更多)"
                if len(pairs) == P["max_pairs"] else "")

    secs: list[dict] = []
    _add(secs, "overview", "總覽", n,
         f"共 {n} 張可讀影像、{len(split_counts)} 個 split",
         "方法:佈局掃描(split=一級子目錄)",
         [{"split": s, "n": c} for s, c in sorted(split_counts.items())])
    _add(secs, "unreadable", "壞檔", n_unread,
         f"{n_unread} 個檔案無法解碼,已排除於各項統計",
         "方法:完整解碼失敗(safe_io)")

    by_sha: dict[str, list[int]] = {}
    for i, s in enumerate(shas):
        by_sha.setdefault(s, []).append(i)
    dup_groups = sorted(((s, idxs) for s, idxs in by_sha.items() if len(idxs) > 1),
                        key=lambda kv: (-len(kv[1]), kv[1][0]))
    n_extra = sum(len(idxs) - 1 for _, idxs in dup_groups)
    _add(secs, "exact_dup", "完全重複", n_extra,
         f"有 {n_extra} 張影像是其他影像的完全複本"
         f"(含跨 split 與異標者;佔 {100.0 * n_extra / max(n, 1):.1f}%)",
         "方法:SHA-256 內容雜湊完全相同",
         [{"sha": s, "paths": [paths[i] for i in idxs]} for s, idxs in dup_groups])

    ppairs = [(i, j, d) for (i, j, d) in inputs["phash_pairs"] if shas[i] != shas[j]]
    _add(secs, "near_dup", "近重複(phash)", len(ppairs),
         f"{len(ppairs)} 對影像高度相似但內容不完全相同"
         + cap_note(inputs["phash_pairs"]),
         f"方法:perceptual hash(8×8)漢明距離 ≤ {P['phash_max_hamming']},"
         "排除內容完全相同者",
         [{"path_i": paths[i], "path_j": paths[j], "hamming": int(d)}
          for i, j, d in ppairs])

    epairs = inputs["emb_pairs"]
    emb_method = (f"方法:embedding cosine 距離 ≤ {P['emb_max_distance']}"
                  f"(模型 {P['model']})")
    if epairs is None:
        _add(secs, "emb_near_dup", "近重複(embedding)", None,
             "未計算(已略過 embedding)", emb_method)
    else:
        _add(secs, "emb_near_dup", "近重複(embedding)", len(epairs),
             f"{len(epairs)} 對影像在語意特徵上幾乎相同" + cap_note(epairs),
             emb_method,
             [{"path_i": paths[i], "path_j": paths[j], "dist": float(d)}
              for i, j, d in epairs])

    lpairs = list(inputs["leak_pairs"])
    _add(secs, "leakage", "跨 split 洩漏", len(lpairs),
         f"{len(lpairs)} 對相同/近似影像同時出現在不同 split(train/val 汙染)"
         + cap_note(lpairs),
         f"方法:跨 split 的 perceptual hash 配對(漢明距離 ≤ {P['phash_max_hamming']})",
         [{"path_i": paths[i], "path_j": paths[j],
           "split_i": splits[i], "split_j": splits[j], "hamming": int(d)}
          for i, j, d in lpairs])

    conf = []
    for s, idxs in by_sha.items():
        labs = sorted({labels[i] for i in idxs if labels[i]})
        if len(labs) > 1:
            conf.append({"sha": s, "labels": labs,
                         "paths": [paths[i] for i in idxs]})
    _add(secs, "label_conflict", "同內容異標", len(conf),
         f"{len(conf)} 組完全相同的影像被標成不同類別",
         "方法:SHA-256 相同但類別標註不同", conf)

    cov_method = "方法:各 split 的類別集合與最大 split 比對"
    labeled = [(sp, lb) for sp, lb in zip(splits, labels) if lb]
    if len(split_counts) < 2 or not labeled:
        reason = "單一 split" if len(split_counts) < 2 else "資料集無類別標註"
        _add(secs, "class_coverage", "split 類別覆蓋", None,
             f"不適用({reason})", cov_method)
    else:
        per_split: dict[str, set] = {}
        for sp, lb in labeled:
            per_split.setdefault(sp, set()).add(lb)
        base = max(split_counts, key=lambda s: (split_counts[s], s))
        base_classes = per_split.get(base, set())
        items = []
        total = 0
        for sp in sorted(split_counts):
            if sp == base:
                continue
            missing = sorted(base_classes - per_split.get(sp, set()))
            total += len(missing)
            items.append({"split": sp, "missing": missing})
        _add(secs, "class_coverage", "split 類別覆蓋", total,
             f"以最大 split「{base}」為基準,其他 split 共缺 {total} 個類別",
             cov_method, items)

    out_method = (f"方法:embedding k={P['outlier_k']} 近鄰平均 cosine 距離"
                  f"(模型 {P['model']}),取前 {P['top_n']}")
    osc = inputs["outlier_scores"]
    if osc is None:
        _add(secs, "outliers", "離群影像", None, "未計算(已略過 embedding)",
             out_method)
    else:
        osc = np.asarray(osc, dtype=float)
        order = np.argsort(-osc)[: int(P["top_n"])]
        items = [{"path": paths[int(i)], "score": float(osc[int(i)])}
                 for i in order]
        _add(secs, "outliers", "離群影像", len(items),
             f"與其他影像最不像的前 {len(items)} 張(分數越大越可疑)",
             out_method, items)

    meta = inputs.get("meta")
    if meta is not None:
        mg = inputs.get("meta_groups") or {}
        items = [{"field": f,
                  "groups": {v: len(idxs) for v, idxs in g["groups"].items()},
                  "n_missing": len(g["missing"])}
                 for f, g in mg.items()]
        pct = 100.0 * meta["n_matched_images"] / max(n, 1)
        _add(secs, "metadata", "製程 metadata", meta["n_matched_images"],
             f"{meta['n_matched_images']} 張影像對上 metadata(匹配率 {pct:.0f}%);"
             f"CSV 有 {len(meta['unmatched_row_idx'])} 列未匹配、"
             f"{meta['n_unmatched_images']} 張影像無資料",
             "方法:CSV 以 sha256/檔名 對照(未匹配明列)", items)

    return {"sections": secs,
            "totals": {"n_images": n, "n_unreadable": n_unread,
                       "split_counts": split_counts}}
