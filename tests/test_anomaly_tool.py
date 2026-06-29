"""驗收:anomaly_tool 核心 run_pipeline(設計 06_anomaly_tool.md,單元部分)。
先紅:scripts/anomaly_tool.py 未實作。用注入 fake → 不需真模型。
約定:records 依 discover_yolo_objects 順序;ranking 為 records 索引,分數由高到低。"""
import numpy as np
import pytest
from pathlib import Path
from sklearn.metrics import roc_auc_score

from anomaly_tool import run_pipeline
from interaction import discover_yolo_objects
from object_eval import list_images, classes_for

REQUIRED = {"path", "image_path", "bbox", "obj_index", "label", "split", "score", "verdict"}


def _ds(ds):
    root = ds["root"]
    ip, cn = list_images(root), classes_for(root)
    meta = discover_yolo_objects(ip, cn)
    defect_idx = {i for i, m in enumerate(meta)
                  if Path(str(m["image_path"])).stem.startswith("defect")}
    normal_idx = [i for i in range(len(meta)) if i not in defect_idx]
    return ip, cn, meta, defect_idx, normal_idx


def test_schema_and_original_path(synthetic_yolo_dataset, color_patch_extractor, color_object_embed):  # AC1
    ip, cn, meta, _, _ = _ds(synthetic_yolo_dataset)
    r = run_pipeline(ip, cn, mode="one_stage", score_mode="patch",
                     extractor=color_patch_extractor(), embed_fn=color_object_embed)
    assert len(r["records"]) == len(meta)
    for rec in r["records"]:
        assert REQUIRED <= set(rec)
        assert rec["path"] == rec["image_path"]      # 匯出原圖 [[object-export-crop-bug]]


def test_subtle_defects_top_and_auroc(synthetic_yolo_dataset, color_patch_extractor, color_object_embed):  # AC2 + AC3
    ip, cn, meta, defect_idx, normal_idx = _ds(synthetic_yolo_dataset)
    confirmed = {i: "good" for i in normal_idx[:8]}  # few-shot 乾淨正常集
    r = run_pipeline(ip, cn, mode="two_stage", score_mode="patch",
                     confirmed=confirmed, extractor=color_patch_extractor(),
                     embed_fn=color_object_embed)
    assert set(r["ranking"][:len(defect_idx)]) == defect_idx           # AC2 細微瑕疵全在前段
    scores = np.array([rec["score"] for rec in r["records"]])
    y = np.array([1 if i in defect_idx else 0 for i in range(len(meta))])
    assert roc_auc_score(y, scores) >= 0.85                            # AC3 餵 G1


def test_bank_pollution_drops_score(synthetic_yolo_dataset, color_patch_extractor, color_object_embed):  # AC4(反向驗證設計修正)
    ip, cn, meta, defect_idx, normal_idx = _ds(synthetic_yolo_dataset)
    clean = {i: "good" for i in normal_idx[:8]}
    d = sorted(defect_idx)[0]
    s_clean = run_pipeline(ip, cn, mode="two_stage", score_mode="patch", confirmed=clean,
                           extractor=color_patch_extractor(),
                           embed_fn=color_object_embed)["records"][d]["score"]
    polluted = dict(clean); polluted[d] = "good"      # 把一個瑕疵誤標進乾淨集 → 污染 bank
    s_pol = run_pipeline(ip, cn, mode="two_stage", score_mode="patch", confirmed=polluted,
                         extractor=color_patch_extractor(),
                         embed_fn=color_object_embed)["records"][d]["score"]
    assert s_pol < 0.5 * s_clean


def _diffuse_embed():
    """注入式:每物件回一個正交(L2 後彼此等距)的嵌入 → HDBSCAN 找不到正常密群、normal_set 空。
    用來確定性重現「diffuse 少樣本」(使用者實際踩的 N=5/8 靜默 None)。"""
    c = {"i": 0}
    def f(crop):
        v = np.zeros(384, dtype=np.float32)
        v[c["i"] % 384] = 1000.0
        c["i"] += 1
        return v
    return f


def test_patch_build_diffuse_normal_builds_bank(synthetic_yolo_dataset, color_patch_extractor):  # AC-F4a:diffuse 少樣本 patch 建模仍建出 bank(修 silent-wrong)
    """diffuse 參考(N≥5 不崩潰,但分群找不到正常密群 → normal_set 空)。
    舊行為:落 leave-one-out、bank 靜默 = None → _anomaly_build_model 假顯示「模型已建立」、
    悄悄退化成物件級(silent-wrong)。修後:整批參考視為正常 → 用全體 patch 建出 Normal Bank。"""
    ip, cn, meta, _, normal_idx = _ds(synthetic_yolo_dataset)
    few = [ip[i] for i in normal_idx[:6]]                 # N=6 ≥5(不崩潰),正交嵌入 → 不分群
    r = run_pipeline(few, cn, mode="two_stage", score_mode="patch",
                     extractor=color_patch_extractor(), embed_fn=_diffuse_embed())
    bank = r["bank"]
    assert bank is not None, "diffuse 少樣本 patch 建模應建出 Normal Bank,不可靜默回 None(假成功)"
    assert getattr(bank, "vectors", None) is not None and len(bank.vectors) > 0
    assert len(r["records"]) == 6


def test_patch_build_tiny_n_no_crash(synthetic_yolo_dataset, color_patch_extractor, color_object_embed):  # AC-F4b:物件數 < 分群最小群(5)不可崩潰
    """物件數 N<5 時,舊行為 HDBSCAN min_samples(5)>N 直接拋 ValueError(GUI 顯示晦澀錯誤)。
    修後:分群退化但不崩潰,patch 建模仍建出 bank。"""
    ip, cn, meta, _, normal_idx = _ds(synthetic_yolo_dataset)
    few = [ip[i] for i in normal_idx[:4]]                 # N=4 < 5 → 舊:HDBSCAN 崩潰
    r = run_pipeline(few, cn, mode="two_stage", score_mode="patch",
                     extractor=color_patch_extractor(), embed_fn=color_object_embed)
    assert r["bank"] is not None and getattr(r["bank"], "vectors", None) is not None
    assert len(r["records"]) == 4


def test_object_mode_and_one_stage_gross(synthetic_yolo_dataset):  # AC5
    ip, cn, meta, defect_idx, _ = _ds(synthetic_yolo_dataset)
    A = np.zeros(384, np.float32); A[0] = 1.0
    B = np.zeros(384, np.float32); B[1] = 1.0

    def gross_ext(crop):  # 任一格紅 → 整個物件全 B(物件級可分,明顯缺陷)
        arr = np.asarray(crop.convert("RGB")).astype(float)
        red = float((arr[..., 0] > 1.2 * (arr[..., 1] + 1)).mean()) > 0.01
        w, h = crop.size; gh, gw = max(1, h // 14), max(1, w // 14)
        return np.repeat((B if red else A)[None, :], gh * gw, axis=0).astype(np.float32), (gh, gw)

    def gross_emb(crop):
        t, _ = gross_ext(crop); v = t.mean(0)
        return (v / (np.linalg.norm(v) or 1.0)).astype(np.float32)

    r_obj = run_pipeline(ip, cn, mode="one_stage", score_mode="object",
                         extractor=gross_ext, embed_fn=gross_emb)
    assert all("verdict" in rec for rec in r_obj["records"])           # object 模式 schema
    r = run_pipeline(ip, cn, mode="one_stage", score_mode="patch",
                     extractor=gross_ext, embed_fn=gross_emb)
    assert set(r["ranking"][:len(defect_idx)]) == defect_idx           # 明顯缺陷 1-stage 仍排前段


def test_no_dataset_pollution(synthetic_yolo_dataset, color_patch_extractor, color_object_embed, tmp_path):  # AC6
    ip, cn, *_ = _ds(synthetic_yolo_dataset)
    root = synthetic_yolo_dataset["root"]
    before = set(root.rglob("*"))
    run_pipeline(ip, cn, mode="one_stage", score_mode="patch",
                 extractor=color_patch_extractor(), embed_fn=color_object_embed,
                 cache_dir=tmp_path / "cache")
    assert set(root.rglob("*")) == before


# ── M6 統一畫面路由純函式(設計 3_Architect_Design/M6_unified_al_screen.md)──
from anomaly_tool import (gate_phase, gate_threshold, head_unlock_state,  # noqa: E402
                          label_semantic_hint, per_class_counts)


def test_per_class_counts_boundaries():  # AC-F1a/b/c
    assert per_class_counts([]) == {}
    assert per_class_counts(["", "", None]) == {}
    assert per_class_counts(["a", "a", "b"]) == {"a": 2, "b": 1}


def test_head_unlock_object_semantics_never_unlocks():  # AC-F2a:物件語義永不解鎖(鎖死 silent-wrong 修復)
    u = head_unlock_state(label_semantic="object", labels=["c%d" % (i % 5) for i in range(500)])
    assert u["unlocked"] is False and u["reason"] == "object_semantic"


def test_head_unlock_undeclared_semantics():  # AC-F2b
    for sem in (None, "", "bogus"):
        u = head_unlock_state(label_semantic=sem, labels=["a"] * 9 + ["b"] * 9)
        assert u["unlocked"] is False and u["reason"] == "undeclared_semantics"


def test_head_unlock_lt2_classes():  # AC-F2c:空/單類
    assert head_unlock_state(label_semantic="defect", labels=[])["reason"] == "lt_2_classes"
    assert head_unlock_state(label_semantic="defect", labels=["a"] * 50)["reason"] == "lt_2_classes"


def test_head_unlock_below_nmin_stays_bank_only():  # AC-F2d
    u = head_unlock_state(label_semantic="defect", labels=["scratch"] * 7 + ["stain"] * 9, n_min=8)
    assert u["unlocked"] is False and u["reason"] == "classes_below_nmin"
    assert u["insufficient_classes"] == {"scratch": 7} and u["eligible_classes"] == ["stain"]


def test_head_unlock_exactly_nmin_unlocks():  # AC-F2e:等號邊界
    u = head_unlock_state(label_semantic="defect", labels=["a"] * 8 + ["b"] * 8, n_min=8)
    assert u["unlocked"] is True and u["reason"] == "ok" and set(u["eligible_classes"]) == {"a", "b"}


def test_head_unlock_min_class_rule_not_total():  # AC-F2f:最小類規則非總數
    u = head_unlock_state(label_semantic="defect", labels=["good"] * 500 + ["scratch"] * 3, n_min=8)
    assert u["unlocked"] is False and u["insufficient_classes"] == {"scratch": 3}


def test_head_unlock_single_eligible_no_unlock():  # AC-F2g:只 1 類達標 → 不解鎖
    u = head_unlock_state(label_semantic="defect", labels=["a"] * 8 + ["b"] * 3, n_min=8)
    assert u["unlocked"] is False and u["eligible_classes"] == ["a"]


def test_gate_phase_auto():  # AC-F3a
    assert gate_phase(has_confirmed_good=False, external_bank=False, head_ready=False)["phase"] == "phase0"
    assert gate_phase(has_confirmed_good=True, external_bank=False, head_ready=False)["phase"] == "phase1"
    assert gate_phase(has_confirmed_good=False, external_bank=True, head_ready=False)["phase"] == "phase1"
    assert gate_phase(has_confirmed_good=True, external_bank=False, head_ready=True)["phase"] == "phase2"


def test_bank_always_active():  # AC-F3b:窮舉 phase/override/head_ready → bank_active 恆 True(顯示鎖)
    for hcg in (False, True):
        for ext in (False, True):
            for hr in (False, True):
                for ov in (None, "phase0", "phase1", "phase2"):
                    assert gate_phase(has_confirmed_good=hcg, external_bank=ext,
                                      head_ready=hr, override=ov)["bank_active"] is True


def test_gate_phase_demote_fake_phase2():  # AC-F3c:override phase2 但無 head → 降級 + overridden
    g = gate_phase(has_confirmed_good=False, external_bank=False, head_ready=False, override="phase2")
    assert g["phase"] == "phase0" and g["overridden"] is True


def test_label_semantic_hint_object_vs_defect():  # AC-F4a/b
    obj = [{"image_path": f"img{i}.jpg", "label": "door"} for i in range(5)
           for _ in range(3)]   # 每圖 3 個同類框 → 像物件偵測
    assert label_semantic_hint(obj)["suggested"] == "object"
    dfc = [{"image_path": f"img{i}.jpg", "label": "scratch"} for i in range(5)]  # 每圖單框
    assert label_semantic_hint(dfc)["suggested"] != "object"


def test_label_semantic_hint_unknown_on_tiny():  # AC-F4c
    assert label_semantic_hint([])["suggested"] == "unknown"
    assert label_semantic_hint([{"image_path": "a.jpg", "label": "x"}])["suggested"] == "unknown"


def test_quantile_gate_is_identity():  # AC-F5a:正常占比≈1-contam 是設計恆等
    scores = np.linspace(0, 1, 100)
    thr = gate_threshold(scores, contamination=0.05)["threshold"]
    assert 93 <= int((scores < thr).sum()) <= 97          # ~95% 落門檻下(與真實瑕疵數無關)


def test_quantile_gate_escapes_when_defect_rate_exceeds_contam():  # AC-F5b:相對門檻會漏檢
    # 80 良品(0)+ 20 真瑕疵(分數各異 1~5);contamination 5% → 門檻只抓前 ~5% → 多數真瑕疵漏
    scores = np.concatenate([np.zeros(80), np.linspace(1.0, 5.0, 20)])
    thr = gate_threshold(scores, contamination=0.05, mode="quantile")["threshold"]
    escaped = int((scores[80:] < thr).sum())              # 真瑕疵被判正常(< 門檻)
    assert escaped >= 10


def test_confirmed_gate_does_not_escape():  # AC-F5c:良品分布校準門檻 → 真瑕疵不漏
    scores = np.concatenate([np.zeros(80), np.linspace(1.0, 5.0, 20)])
    confirmed = {i: "good" for i in range(10)}            # 10 個確認良品(分數 0)
    g = gate_threshold(scores, contamination=0.05, confirmed=confirmed, mode="confirmed")
    assert g["calibrated"] is True
    assert int((scores[80:] < g["threshold"]).sum()) == 0  # 真瑕疵全在門檻上 → 0 漏


def test_confirmed_insufficient_falls_back():  # AC-F5d:良品不足 → 退回 quantile、誠實不假裝校準
    scores = np.linspace(0, 1, 100)
    g = gate_threshold(scores, confirmed={0: "good", 1: "good"}, mode="confirmed", min_confirmed=5)
    assert g["calibrated"] is False and g["reason"] == "insufficient_confirmed"
    assert g["threshold"] == pytest.approx(gate_threshold(scores, contamination=0.05)["threshold"])


def test_run_pipeline_always_returns_scores(synthetic_yolo_dataset, color_patch_extractor, color_object_embed):
    # 結構不變量(Bank 骨幹的真鎖):注入可分 fake → 每分支都算過 scores(長度 N 且非佔位零)
    ip, cn, meta, *_ = _ds(synthetic_yolo_dataset)
    for kw in ({"score_mode": "object"}, {"score_mode": "patch"}):
        r = run_pipeline(ip, cn, mode="one_stage", extractor=color_patch_extractor(),
                         embed_fn=color_object_embed, **kw)
        s = np.asarray(r["scores"], float)
        assert s.shape[0] == len(meta) and float(s.std()) > 0.0
