# -*- coding: utf-8 -*-
"""tools/eval_scenarios.py — 瑕疵偵測全功能「真實資料情境驗證」離線評估(可重現)。

對任一 YOLO 物件偵測資料集(train/ + valid/,各含 images/ labels/、classes.txt)跑 6 情境並輸出
分數卡 JSON,讓 Normal Bank 異常 / bank 跨資料夾 / 分類頭 / 閘控 / 主動學習 / escape-overkill
的真實表現可被機器復現(非只在文件聲明)。對應架構文件:大量 good(常見類)、少數 bad(稀有類)。

用法:
  python tools/eval_scenarios.py --dataset C:/path/to/yolo_root --out report.json \
      --good cabinetDoor refrigeratorDoor door window --bad pole couch openedDoor

設計取捨:物件級 embedding(1 vec/物件)求速度;DINOv2 vits14,首跑算特徵較久、之後 .lv_cache 命中。
注意:本工具「只讀」資料集,所有快取進 .lv_cache(見記憶 no-dataset-writes)。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from active_learning import priority_score, select_for_labeling  # noqa: E402
from anomaly_bank_store import check_compat, load_bank, save_bank  # noqa: E402
from anomaly_project import fit_projector, transform_new  # noqa: E402
from anomaly_tool import run_pipeline  # noqa: E402
from dino_head import gated_predict, predict_head, train_head  # noqa: E402
from object_eval import classes_for, dataset_cache_dir, list_images  # noqa: E402
from sklearn.metrics import (balanced_accuracy_score, recall_score,  # noqa: E402
                             roc_auc_score)
from sklearn.preprocessing import normalize  # noqa: E402


def _ci95(vals):
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def escape_overkill_curve(dist, bad_mask, good_mask, good_self, percentiles=(70, 80, 90, 95, 99)):
    """純函數契約(供 S6 與契約測試共用):給距離分數 + good/bad mask + good 自身分布,
    回各 good 百分位門檻下的 (pct, thr, escape, over_kill)。
    契約:percentile↑ → 門檻↑ → escape(bad<門檻=漏檢)單調不減、over_kill(good≥門檻)單調不增。"""
    bd, gd = dist[bad_mask], dist[good_mask]
    out = []
    for pct in percentiles:
        thr = float(np.percentile(good_self, pct))
        out.append((pct, thr,
                    float((bd < thr).mean()) if len(bd) else None,
                    float((gd >= thr).mean()) if len(gd) else None))
    return out


def _run(folder, model, res, external_ref=None):
    ip, cn = list_images(folder), classes_for(folder)
    cache = dataset_cache_dir(folder, f"anomaly_patch_object_{model}_r{res}")
    return run_pipeline(ip, cn, mode="one_stage", score_mode="object", model=model,
                        target_res=res, cache_dir=cache, external_ref=external_ref)


def evaluate(dataset: Path, good: set, bad: set, rare: set, model="dinov2_vits14", res=224, eval_split="valid"):
    train, valid = dataset / "train", dataset / eval_split
    rep = {"data": {}, "scenarios": {}}

    t0 = time.time()
    rt = _run(train, model, res)
    labels_tr = np.array([r["label"] for r in rt["records"]])
    emb_tr = np.asarray(rt["obj_emb"], np.float32)
    scores_tr = np.asarray(rt["scores"], float)
    rep["data"] = {"train_objects": len(labels_tr), "train_secs": round(time.time() - t0, 1),
                   "class_counts": {c: int((labels_tr == c).sum()) for c in sorted(set(labels_tr.tolist()))}}

    # S1:Normal Bank 無監督異常 → 稀有類應較異常(開集 novelty)+ per-class AUROC + bootstrap CI
    is_rare = np.array([1 if x in rare else 0 for x in labels_tr])
    s1 = {}
    if 0 < is_rare.sum() < len(is_rare):
        top = np.argsort(-scores_tr)[: max(1, len(scores_tr) // 10)]
        s1 = {"auroc_rare_vs_anomaly": round(float(roc_auc_score(is_rare, scores_tr)), 3),
              "rare_rate_overall": round(float(is_rare.mean()), 3),
              "rare_rate_in_top10pct_anomaly": round(float(is_rare[top].mean()), 3),
              "per_rare_class_auroc": {c: round(float(roc_auc_score((labels_tr == c).astype(int), scores_tr)), 3)
                                       for c in sorted(rare) if 0 < (labels_tr == c).sum() < len(labels_tr)}}
        _b = []
        for i in range(500):
            ix = np.random.default_rng(i).integers(0, len(scores_tr), len(scores_tr))
            y = is_rare[ix]
            if 0 < y.sum() < len(y):
                _b.append(float(roc_auc_score(y, scores_tr[ix])))
        s1["auroc_rare_ci95"] = _ci95(_b)
    rep["scenarios"]["S1_normal_bank_novelty"] = s1

    rv = _run(valid, model, res)
    labels_va = np.array([r["label"] for r in rv["records"]])
    emb_va = np.asarray(rv["obj_emb"], np.float32)
    sc_va = np.asarray(rv["scores"], float)
    thr_va = float(rv["threshold"])

    # S2:bank 存讀 + 跨資料夾投影 + 數值保真
    basis = fit_projector(emb_tr)
    bank_dir = dataset_cache_dir(train, "anomaly_bank/_eval")
    save_bank(bank_dir, projection=basis, meta={"model": model, "target_res": res, "patch_dim": None,
              "obj_dim": int(emb_tr.shape[1]), "score_mode": "object"})
    lb = load_bank(bank_dir)
    s2 = {"saved_files": sorted(p.name for p in Path(bank_dir).iterdir())}
    try:
        check_compat(lb["meta"], model=model, target_res=res, obj_dim=int(emb_tr.shape[1]))
        s2["compat_ok"] = True
    except Exception as e:
        s2["compat_ok"] = False; s2["compat_err"] = str(e)
    try:
        check_compat(lb["meta"], model=model + "_x", target_res=res); s2["incompat_rejected"] = False
    except Exception:
        s2["incompat_rejected"] = True
    probe = emb_tr[:500]
    s2["proj_roundtrip_max_abs_diff"] = round(float(np.max(np.abs(
        transform_new(basis, probe) - transform_new(lb["projection"], probe)))), 6)
    s2["proj_roundtrip_equal"] = bool(s2["proj_roundtrip_max_abs_diff"] < 1e-4)
    rv_ext = _run(valid, model, res, external_ref=lb["projection"]["good_obj_emb"])
    s2["valid_scored_with_train_ref"] = rv_ext["bank"] is None and len(rv_ext["records"]) == len(labels_va)
    s2["train_ref_scores_nondegenerate"] = bool(float(np.asarray(rv_ext["scores"], float).std()) > 1e-6)
    rep["scenarios"]["S2_bank_persist_crossfolder"] = s2

    # S3:分類頭 closed-set(train 訓 → valid 測)+ per-class recall + 信心×對錯校準
    head = train_head(emb_tr, labels_tr)
    pred_va, conf_va, proba_va = predict_head(head, emb_va)
    present = sorted(set(labels_va.tolist()))
    pcr = {c: round(float(recall_score(labels_va == c, pred_va == c, zero_division=0)), 3) for c in present}
    rare_present = [c for c in present if c in rare]
    correct = (pred_va == labels_va)
    s3 = {"valid_balanced_acc": round(float(balanced_accuracy_score(labels_va, pred_va)), 3),
          "valid_acc": round(float(correct.mean()), 3),
          "majority_baseline": round(float(max((labels_va == c).mean() for c in present)), 3),
          "balanced_chance": round(1.0 / len(present), 3),
          "per_class_recall": pcr,
          "rare_class_mean_recall": (round(float(np.mean([pcr[c] for c in rare_present])), 3) if rare_present else None),
          "mean_conf_correct": round(float(conf_va[correct].mean()), 3) if correct.any() else None,
          "mean_conf_wrong": round(float(conf_va[~correct].mean()), 3) if (~correct).any() else None,
          "rare_class_valid_n": {c: int((labels_va == c).sum()) for c in rare_present}}
    if s3["mean_conf_correct"] is not None and s3["mean_conf_wrong"] is not None:
        s3["calibration_gap_correct_minus_wrong"] = round(s3["mean_conf_correct"] - s3["mean_conf_wrong"], 3)
    # bootstrap CI for balanced_acc + rare recall(對齊 S1/S5/S6);rare recall 同時報 macro 與 support-weighted
    _nv = len(labels_va)
    _bacc, _rrec = [], []
    for i in range(500):
        ix = np.random.default_rng(i).integers(0, _nv, _nv)
        yv, pv = labels_va[ix], pred_va[ix]
        _bacc.append(float(balanced_accuracy_score(yv, pv)))
        _rp = [c for c in rare_present if (yv == c).any()]
        if _rp:
            _rrec.append(float(np.mean([recall_score(yv == c, pv == c, zero_division=0) for c in _rp])))
    s3["balanced_acc_ci95"] = _ci95(_bacc)
    s3["rare_recall_macro_ci95"] = _ci95(_rrec) if _rrec else None
    _wn = {c: int((labels_va == c).sum()) for c in rare_present}
    _wt = sum(_wn.values())
    s3["rare_class_recall_weighted"] = round(sum(pcr[c] * _wn[c] for c in rare_present) / _wt, 3) if _wt else None
    rep["scenarios"]["S3_classification_head"] = s3

    # S4:閘控級聯 + contamination 掃描(recall-first tradeoff:良品/缺陷用 good/bad)
    vg, vb = np.isin(labels_va, list(good)), np.isin(labels_va, list(bad))
    s4 = {"contamination_scan": {}}
    _nb, _ng = int(vb.sum()), int(vg.sum())
    for c in (0.02, 0.05, 0.10, 0.20, 0.30):
        t = float(np.quantile(sc_va, 1.0 - c))
        g = np.array(gated_predict(head, emb_va, sc_va, anomaly_threshold=t, min_conf=0.5))
        nrm = (g == "正常")
        _esc_b = ([float(nrm[vb][np.random.default_rng(i).integers(0, _nb, _nb)].mean()) for i in range(500)]
                  if _nb else [])
        _ovk_b = ([float((~nrm)[vg][np.random.default_rng(i).integers(0, _ng, _ng)].mean()) for i in range(500)]
                  if _ng else [])
        s4["contamination_scan"][f"c{c}"] = {
            "normal_share": round(float(nrm.mean()), 3),
            "defect_escape": round(float(nrm[vb].mean()), 3) if vb.any() else None,
            "defect_escape_ci95": _ci95(_esc_b) if _esc_b else None,
            "good_overkill": round(float((~nrm)[vg].mean()), 3) if vg.any() else None,
            "good_overkill_ci95": _ci95(_ovk_b) if _ovk_b else None}
    s4["normal_share_note"] = "正常占比 ≈ 1-contamination 為門檻設計恆等,非偵測訊號"
    rep["scenarios"]["S4_gated_cascade"] = s4

    # S5:主動學習取樣(三模式 + bootstrap CI + 真標籤多樣性)
    is_rare_va = np.array([1 if x in rare else 0 for x in labels_va])
    div = np.asarray(pred_va)
    K = 30
    modes = {"novelty": (1, 0, 0), "blend_default": (1, 0.4, 0.4), "balanced_old": (1, 1, 1)}
    sel_by = {m: select_for_labeling(priority_score(sc_va, head_proba=proba_va, anomaly_threshold=thr_va,
              w_novelty=w[0], w_boundary=w[1], w_disagreement=w[2]), k=K, cluster_labels=div, max_per_cluster=3)
              for m, w in modes.items()}
    sel = sel_by["blend_default"]
    rr = [float(is_rare_va[np.random.default_rng(s).choice(len(labels_va), min(K, len(labels_va)), replace=False)].mean())
          for s in range(20)]
    cr = [float(conf_va[np.random.default_rng(s).choice(len(labels_va), min(K, len(labels_va)), replace=False)].mean())
          for s in range(20)]
    boot = [float(is_rare_va[np.array(sel)[np.random.default_rng(b).integers(0, len(sel), len(sel))]].mean())
            for b in range(2000)]
    s5 = {"k": K, "default_mode": "blend(1,0.4,0.4)",
          "rare_rate_in_AL_queue": round(float(is_rare_va[sel].mean()), 3),
          "rare_rate_AL_ci95": _ci95(boot),
          "rare_rate_random_mean20seed": round(float(np.mean(rr)), 3),
          "rare_rate_by_mode": {m: round(float(is_rare_va[s].mean()), 3) for m, s in sel_by.items()},
          "mean_conf_in_AL_queue": round(float(conf_va[sel].mean()), 3),
          "mean_conf_random_mean20seed": round(float(np.mean(cr)), 3),
          "true_label_groups_covered_AL": int(len(set(labels_va[sel].tolist()))),
          "true_label_groups_total": int(len(set(labels_va.tolist())))}
    rep["scenarios"]["S5_active_learning"] = s5

    # S6:忠實 escape/over-kill(只用 good 建 Normal Bank,train→valid 無 self-match)+ tradeoff CI
    gtr = normalize(emb_tr[np.isin(labels_tr, list(good))])
    dist_va = 1.0 - (normalize(emb_va) @ gtr.T).max(axis=1)
    gg = gtr @ gtr.T
    np.fill_diagonal(gg, -np.inf)
    good_self = 1.0 - gg.max(axis=1)
    vbd = dist_va[vb]
    vgd = dist_va[vg]
    # AUROC + bootstrap CI(valid_bad_n 小 → 標頭也附 CI,對齊 S1/S3)
    _yb, _db = vb[vg | vb], dist_va[vg | vb]
    _ab = []
    for i in range(500):
        ix = np.random.default_rng(i).integers(0, len(_yb), len(_yb))
        if 0 < _yb[ix].sum() < len(ix):
            _ab.append(float(roc_auc_score(_yb[ix], _db[ix])))
    s6 = {"valid_good_n": int(vg.sum()), "valid_bad_n": int(vb.sum()),
          "auroc_good_vs_bad": round(float(roc_auc_score(_yb, _db)), 3),
          "auroc_good_vs_bad_ci95": _ci95(_ab) if _ab else None, "tradeoff": {}}
    for pct, thr, esc_pt, ovk_pt in escape_overkill_curve(dist_va, vb, vg, good_self):
        esc = (vbd < thr).astype(float)
        ovk = (vgd >= thr).astype(float)
        eb = [float(esc[np.random.default_rng(pct * 1000 + b).integers(0, len(esc), len(esc))].mean()) for b in range(1000)]
        ob = [float(ovk[np.random.default_rng(pct * 1000 + 7 + b).integers(0, len(ovk), len(ovk))].mean()) for b in range(1000)]
        s6["tradeoff"][f"p{pct}"] = {"escape_rate": round(float(esc_pt), 3), "escape_ci95": _ci95(eb),
                                     "over_kill": round(float(ovk_pt), 3), "over_kill_ci95": _ci95(ob)}
    s6["note"] = f"稀有類視覺相似常見物件時(非強離群)escape 偏高屬資料特性;valid_bad_n={int(vb.sum())}"
    rep["scenarios"]["S6_escape_overkill"] = s6
    return rep


def main():
    ap = argparse.ArgumentParser(description="瑕疵偵測全功能真實資料情境驗證")
    ap.add_argument("--dataset", required=True, help="YOLO 根目錄(含 train/ 與評估 split)")
    ap.add_argument("--eval-split", default="valid", help="held-out 評估 split 資料夾名(valid/test)")
    ap.add_argument("--out", default="eval_report.json")
    ap.add_argument("--good", nargs="+", default=["cabinetDoor", "refrigeratorDoor", "door", "window"])
    ap.add_argument("--bad", nargs="+", default=["pole", "couch", "openedDoor"])
    ap.add_argument("--rare", nargs="+", default=["pole", "openedDoor", "couch", "cabinet", "table"])
    ap.add_argument("--model", default="dinov2_vits14")
    ap.add_argument("--res", type=int, default=224)
    a = ap.parse_args()
    rep = evaluate(Path(a.dataset), set(a.good), set(a.bad), set(a.rare), a.model, a.res, a.eval_split)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    # cp950 安全:不直印含 unicode 的 JSON,只印路徑
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
