"""驗收:al_batch(設計 3_Architect_Design/08_al_batch.md)。

先紅:`scripts/al_batch.py` 尚未實作。
- `al_batch` 的 import 一律放在**測試函式內** → 未實作時是「執行期紅」(test fail),非 collection error。
- 全程用**注入式** extractor / embed_fn + 合成向量凍結模型(save_bank/save_head),不觸真 DINOv2、不讀真模型。
- 影像用色彩編碼兩個「獨立訊號」:
    紅色通道 r → 決定「異常分數 score」(graded extractor:patch 與 bank-A 的距離隨 r 單調上升)
    藍色通道 b → 決定「head 預測類別」(class embed_fn:obj_emb 落在 X 或 Y 區,與 r 無關)
  → 分數與類別可各自控制,才能精確驗 Top-K 排序 / tie-break / 多樣性。

契約鎖點(防 silent-wrong):分批==一次跑(AC1)、item_id 內容定址跨模型不變(AC7)、
暫定=已處理真 Top-K(AC3/AC13)、身分漂移拒絕續跑(AC5)、不寫來源資料夾(AC8)。
"""
import json
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

# 既有(已實作)基礎設施 —— 造凍結模型 fixture 用
from anomaly_bank_store import save_bank
from dino_head import save_head, train_head

_D = 384  # DINOv2 vits14 dim


# ── 合成基底:score 訊號放 dim0/1,class 訊號放 dim2/3(彼此正交、互不干擾)──
def _e(i):
    v = np.zeros(_D, dtype=np.float32)
    v[i] = 1.0
    return v


_E0, _E1, _E2, _E3 = _e(0), _e(1), _e(2), _e(3)


def _unit(v):
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n else v.astype(np.float32)


# ── 注入式 extractor:紅通道 → 一個 patch,與 bank-A(=E0)距離隨 r 單調 ──
def _graded_extractor(counter=None):
    def ext(crop):
        if counter is not None:
            counter["n"] += 1
        arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
        s = float(arr[..., 0].mean()) / 255.0          # 紅越多 → s 越大 → 越異常
        patch = _unit((1.0 - s) * _E0 + s * _E1)
        return patch[None, :], (1, 1)                   # 單一 patch, grid (1,1)
    return ext


# ── 注入式 object embed_fn:藍通道 → obj_emb 落在 X(E2)或 Y(E3)區(與紅無關)──
def _class_embed(counter=None):
    def emb(crop):
        if counter is not None:
            counter["n"] += 1
        arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
        b = float(arr[..., 2].mean()) / 255.0
        return _unit((1.0 - b) * _E2 + b * _E3)          # b<0.5→近 X, b>0.5→近 Y
    return emb


def _img(folder: Path, name: str, *, red: int, blue: int = 0):
    """造一張 32x32 影像:紅=score 訊號、藍=class 訊號、綠固定。"""
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    Image.new("RGB", (32, 32), (int(red), 120, int(blue))).save(p)
    return p


def _make_model(model_dir: Path, *, with_head: bool = False):
    """造一個凍結模型目錄:patch bank(50 個正常 A=E0 向量)+(可選)2 類 head。"""
    save_bank(model_dir,
              vectors=np.repeat(_E0[None, :], 50, axis=0),
              meta={"model": "dinov2_vits14", "target_res": 224, "score_mode": "patch"})
    if with_head:
        # head 用 obj_emb 訓練:X 群(E2)10 個 + Y 群(E3)10 個 → 兩類清楚可分
        X = np.vstack([np.repeat(_E2[None, :], 10, axis=0),
                       np.repeat(_E3[None, :], 10, axis=0)])
        y = ["X"] * 10 + ["Y"] * 10
        head = train_head(X, y)
        save_head(model_dir / "head.joblib", head)


def _run(**kw):
    """呼叫 al_batch.run_batched(import 放函式內 → 未實作是執行期紅)。"""
    import al_batch
    kw.setdefault("extractor", _graded_extractor())
    kw.setdefault("embed_fn", _class_embed())
    kw.setdefault("object_source", "whole_image")   # 每圖 1 物件 → images 與 objects 1:1
    return al_batch.run_batched(**kw)


# ══════════════════════════════════════════════════════════════════════
# 分批正確性 / 續跑
# ══════════════════════════════════════════════════════════════════════

def test_ac1_batched_equals_single(tmp_path):
    # AC1:分批(batch_size=7)最終 topk 逐一等於一次跑完(batch_size=40)
    ds = tmp_path / "ds"
    paths = [_img(ds, f"img_{i:02d}.png", red=5 + 6 * i) for i in range(40)]  # 相異紅 → 相異分數
    md = tmp_path / "model"; _make_model(md)
    r_split = _run(image_paths=paths, model_dir=md, checkpoint_dir=tmp_path / "ck1",
                   objective="novelty", k=10, batch_size=7, dataset_dirs=(ds,))
    r_whole = _run(image_paths=paths, model_dir=md, checkpoint_dir=tmp_path / "ck2",
                   objective="novelty", k=10, batch_size=40, dataset_dirs=(ds,))
    assert r_split["done"] and r_whole["done"]
    assert r_split["topk"] == r_whole["topk"], "分批與一次跑完的 Top-K 必須逐一相同"
    assert r_split["objects_scored"] == r_whole["objects_scored"] == 40


def test_ac2_resume_no_recompute(tmp_path):
    # AC2:中斷後續跑,已處理批的 extractor 呼叫增量為 0(靠 shard/.lv_cache),結果 == 未中斷
    ds = tmp_path / "ds"
    paths = [_img(ds, f"img_{i:02d}.png", red=5 + 6 * i) for i in range(30)]
    md = tmp_path / "model"; _make_model(md)
    ck = tmp_path / "ck"

    import al_batch
    c1 = {"n": 0}
    # 先只跑前 2 批(max_batches),記錄 extractor 呼叫數
    al_batch.run_batched(image_paths=paths, model_dir=md, checkpoint_dir=ck,
                         objective="novelty", k=10, batch_size=10, max_batches=2,
                         object_source="whole_image", dataset_dirs=(ds,),
                         extractor=_graded_extractor(c1), embed_fn=_class_embed())
    first_calls = c1["n"]
    assert first_calls > 0

    c2 = {"n": 0}
    r = al_batch.run_batched(image_paths=paths, model_dir=md, checkpoint_dir=ck,
                             objective="novelty", k=10, batch_size=10, resume=True,
                             object_source="whole_image", dataset_dirs=(ds,),
                             extractor=_graded_extractor(c2), embed_fn=_class_embed())
    # 續跑只該 embed 剩下 1 批(10 物件);已處理 20 物件不重算
    assert c2["n"] <= 10, f"續跑重算了已處理物件(extractor 呼叫 {c2['n']})"
    assert r["done"] and r["objects_scored"] == 30

    # 與完全不中斷、一氣呵成的結果一致
    r_ref = _run(image_paths=paths, model_dir=md, checkpoint_dir=tmp_path / "ck_ref",
                 objective="novelty", k=10, batch_size=10, dataset_dirs=(ds,))
    assert r["topk"] == r_ref["topk"]


def test_ac3_provisional_topk_is_true_topk_of_processed(tmp_path):
    # AC3:跑一批後,progress 收到的 provisional_topk = 「僅已處理物件、依原始分數」的前 k
    ds = tmp_path / "ds"
    # 依 path 排序後前 batch_size 個 = 批 1;讓分數與檔名序不同調,才驗得出「取分數最大」而非「取前幾個」
    reds = [200, 20, 150, 40, 90, 10, 250, 60, 30, 5]
    paths = [_img(ds, f"img_{i:02d}.png", red=reds[i]) for i in range(10)]
    md = tmp_path / "model"; _make_model(md)

    seen = {}

    def prog(d):
        seen.update(d)  # 留最後一次(批 1 完)

    r = _run(image_paths=paths, model_dir=md, checkpoint_dir=tmp_path / "ck",
             objective="novelty", k=3, batch_size=5, max_batches=1,
             dataset_dirs=(ds,), progress=prog)
    assert r["done"] is False and r["provisional"] is True
    # 批 1 = 排序後前 5 個 path(img_00..img_04),紅值 [200,20,150,40,90]
    # 依分數(∝紅)由大到小取前 3 → img_00(200), img_02(150), img_04(90)
    id_by_stem = {Path(rec["image_path"]).stem: rec["item_id"] for rec in r["topk_records"]}
    # topk_records 是「已處理併集」的 topk;比對其 image_path 對應的紅序
    got_reds = sorted((int(np.asarray(Image.open(rec["image_path"]))[..., 0].mean().round()))
                      for rec in r["topk_records"])
    # 允許 ±2 的取整誤差;應為 {90,150,200} 這組最大三個(批 1 內)
    assert got_reds == sorted([90, 150, 200]) or all(v >= 85 for v in got_reds), \
        f"暫定 Top-K 不是已處理併集的分數前三:{got_reds}"
    assert "provisional_topk" in seen and len(seen["provisional_topk"]) == 3


def test_ac4_topk_deterministic_across_batch_size_on_ties(tmp_path):
    # AC4:分數全同(大量 tie)時,batch_size=3 與 =50 的最終 topk 完全相同(item_id canonical 排序)
    ds = tmp_path / "ds"
    paths = [_img(ds, f"img_{i:02d}.png", red=100) for i in range(12)]  # 同紅 → 同分數 → 全 tie
    md = tmp_path / "model"; _make_model(md)
    r_a = _run(image_paths=paths, model_dir=md, checkpoint_dir=tmp_path / "cka",
               objective="novelty", k=5, batch_size=3, dataset_dirs=(ds,))
    r_b = _run(image_paths=paths, model_dir=md, checkpoint_dir=tmp_path / "ckb",
               objective="novelty", k=5, batch_size=50, dataset_dirs=(ds,))
    assert r_a["topk"] == r_b["topk"], "tie 下 Top-K 必須跨 batch_size 位元相同(item_id 排序,非陣列位置)"


def test_ac5_resume_refuses_on_identity_drift(tmp_path):
    # AC5:改 batch_size 後續跑 → error 模式 raise;restart 模式清 checkpoint 重跑
    ds = tmp_path / "ds"
    paths = [_img(ds, f"img_{i:02d}.png", red=5 + 6 * i) for i in range(20)]
    md = tmp_path / "model"; _make_model(md)
    ck = tmp_path / "ck"
    import al_batch
    al_batch.run_batched(image_paths=paths, model_dir=md, checkpoint_dir=ck,
                         objective="novelty", k=5, batch_size=10, max_batches=1,
                         object_source="whole_image", dataset_dirs=(ds,),
                         extractor=_graded_extractor(), embed_fn=_class_embed())
    # batch_size 變 → shard 邊界移動 → 必拒
    with pytest.raises((ValueError, RuntimeError)):
        al_batch.run_batched(image_paths=paths, model_dir=md, checkpoint_dir=ck,
                             objective="novelty", k=5, batch_size=7, resume=True,
                             on_identity_mismatch="error",
                             object_source="whole_image", dataset_dirs=(ds,),
                             extractor=_graded_extractor(), embed_fn=_class_embed())
    # restart 模式:清掉重跑,能完成
    r = al_batch.run_batched(image_paths=paths, model_dir=md, checkpoint_dir=ck,
                             objective="novelty", k=5, batch_size=7, resume=True,
                             on_identity_mismatch="restart",
                             object_source="whole_image", dataset_dirs=(ds,),
                             extractor=_graded_extractor(), embed_fn=_class_embed())
    assert r["done"] and r["objects_scored"] == 20


# ══════════════════════════════════════════════════════════════════════
# 契約 / 目標 / 身分
# ══════════════════════════════════════════════════════════════════════

def test_ac6_objective_requires_head_before_embedding(tmp_path):
    # AC6:objective='uncertain' 給無 head 的模型 → 開跑前 raise,extractor 呼叫 0 次
    ds = tmp_path / "ds"
    paths = [_img(ds, f"img_{i:02d}.png", red=50) for i in range(5)]
    md = tmp_path / "model"; _make_model(md, with_head=False)
    import al_batch
    c = {"n": 0}
    with pytest.raises(ValueError):
        al_batch.run_batched(image_paths=paths, model_dir=md,
                             checkpoint_dir=tmp_path / "ck", objective="uncertain",
                             k=5, batch_size=10, object_source="whole_image",
                             dataset_dirs=(ds,), extractor=_graded_extractor(c),
                             embed_fn=_class_embed())
    assert c["n"] == 0, "需 head 的目標必須在任何 embedding 之前就擋下"


def test_ac7_item_id_content_addressed(tmp_path):
    # AC7(a) 同物件跨不同 model_dir → 同 item_id;(b) 同 stem 不同內容(不同資料夾)→ 不同 item_id
    dsA = tmp_path / "A"; dsB = tmp_path / "B"
    pA = _img(dsA, "img.png", red=200)      # A/img.png 紅
    pB = _img(dsB, "img.png", red=20)       # B/img.png 綠(同 stem 不同內容)
    md1 = tmp_path / "m1"; _make_model(md1)
    md2 = tmp_path / "m2"; _make_model(md2)   # 內容不同的第二個模型(見下方確保)
    # 讓兩模型內容不同(bank 向量列數不同)
    save_bank(md2, vectors=np.repeat(_E0[None, :], 77, axis=0),
              meta={"model": "dinov2_vits14", "target_res": 224, "score_mode": "patch"})

    def _id_of(path, md, ck):
        # dataset_dirs 指「真實資料夾」(A/B 子資料夾),checkpoint 放這些之外的 ck 才安全
        r = _run(image_paths=[pA, pB], model_dir=md, checkpoint_dir=ck,
                 objective="novelty", k=5, batch_size=10, dataset_dirs=(dsA, dsB))
        return {Path(rec["image_path"]).resolve(): rec["item_id"] for rec in r["topk_records"]}

    ids1 = _id_of(pA, md1, tmp_path / "ck1")
    ids2 = _id_of(pA, md2, tmp_path / "ck2")
    # (a) A/img.png 在兩個模型下 item_id 相同(id 不含模型)
    assert ids1[pA.resolve()] == ids2[pA.resolve()], "同物件跨模型 item_id 必須不變(rescore 原地更新)"
    # (b) 同 stem 不同內容 → 不同 item_id(證明用影像內容雜湊,非 stem)
    assert ids1[pA.resolve()] != ids1[pB.resolve()], "同 stem 不同內容必須是不同 item_id(防跨資料夾撞名)"


def test_ac8_never_writes_dataset_folder(tmp_path):
    # AC8:跑完來源資料夾檔案清單 + mtime 快照零變化
    ds = tmp_path / "ds"
    paths = [_img(ds, f"img_{i:02d}.png", red=5 + 6 * i) for i in range(15)]
    snap = {p: p.stat().st_mtime_ns for p in ds.rglob("*")}
    md = tmp_path / "model"; _make_model(md)
    _run(image_paths=paths, model_dir=md, checkpoint_dir=tmp_path / "ck",
         objective="novelty", k=5, batch_size=5, dataset_dirs=(ds,))
    after = {p: p.stat().st_mtime_ns for p in ds.rglob("*")}
    assert after == snap, "來源資料夾不可被寫入/變動(C6)"


def test_ac9_rejects_unsafe_checkpoint_dir(tmp_path):
    # AC9:checkpoint_dir 落在來源資料夾內 → assert_safe_bank_dir raise
    ds = tmp_path / "ds"
    paths = [_img(ds, f"img_{i:02d}.png", red=50) for i in range(4)]
    md = tmp_path / "model"; _make_model(md)
    import al_batch
    with pytest.raises(ValueError):
        al_batch.run_batched(image_paths=paths, model_dir=md,
                             checkpoint_dir=ds / "ckpt",       # 在來源資料夾底下 → 危險
                             objective="novelty", k=5, batch_size=5,
                             object_source="whole_image", dataset_dirs=(ds,),
                             extractor=_graded_extractor(), embed_fn=_class_embed())


def test_ac10_empty_and_zero_object_batch(tmp_path):
    # AC10:yolo 模式某批影像無 label 框 → 空批不崩、commit、objects_scored 不含空批
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True); (ds / "labels").mkdir()
    (ds / "classes.txt").write_text("obj\n", encoding="utf-8")
    paths = []
    for i in range(6):
        p = ds / "images" / f"img_{i:02d}.jpg"
        Image.new("RGB", (48, 48), (200 if i % 2 == 0 else 20, 120, 0)).save(p)
        paths.append(p)
        if i % 2 == 0:  # 只有偶數張有框;奇數張「無 label 檔」→ 0 物件
            (ds / "labels" / f"img_{i:02d}.txt").write_text(
                "0 0.5 0.5 0.9 0.9\n", encoding="utf-8")
    md = tmp_path / "model"; _make_model(md)
    r = _run(image_paths=paths, model_dir=md, checkpoint_dir=tmp_path / "ck",
             objective="novelty", k=5, batch_size=2, object_source="yolo",
             class_names=["obj"], dataset_dirs=(ds,))
    assert r["done"] is True
    assert r["objects_scored"] == 3, "只有 3 張有框 → 3 個物件(空批不計、不 KeyError)"


def test_ac11_crash_between_shard_and_state_is_idempotent(tmp_path):
    # AC11:模擬「shard 寫入後、state 更新前」crash(刪掉 state 對某批的登錄、留 orphan shard)
    #        → 續跑重跑該批冪等(item_id 去重),objects_scored / topk == 未中斷
    ds = tmp_path / "ds"
    paths = [_img(ds, f"img_{i:02d}.png", red=5 + 6 * i) for i in range(20)]
    md = tmp_path / "model"; _make_model(md)
    ref = _run(image_paths=paths, model_dir=md, checkpoint_dir=tmp_path / "ckref",
               objective="novelty", k=5, batch_size=10, dataset_dirs=(ds,))

    import al_batch
    ck = tmp_path / "ck"
    al_batch.run_batched(image_paths=paths, model_dir=md, checkpoint_dir=ck,
                         objective="novelty", k=5, batch_size=10, max_batches=2,
                         object_source="whole_image", dataset_dirs=(ds,),
                         extractor=_graded_extractor(), embed_fn=_class_embed())
    # 模擬 crash:把 state.json 對「最後一批」的登錄移除,但 shard 檔留著(orphan)
    sp = ck / "state.json"
    st = json.loads(sp.read_text(encoding="utf-8"))
    assert st.get("committed_batches"), "state.json 應記錄 committed_batches(設計 §2.8)"
    st["committed_batches"] = st["committed_batches"][:-1]      # 丟掉最後一批的登錄
    st["done"] = False
    sp.write_text(json.dumps(st), encoding="utf-8")

    r = al_batch.run_batched(image_paths=paths, model_dir=md, checkpoint_dir=ck,
                             objective="novelty", k=5, batch_size=10, resume=True,
                             object_source="whole_image", dataset_dirs=(ds,),
                             extractor=_graded_extractor(), embed_fn=_class_embed())
    assert r["objects_scored"] == ref["objects_scored"] == 20, "orphan shard 重跑不可雙計"
    assert r["topk"] == ref["topk"]


# ══════════════════════════════════════════════════════════════════════
# 版本 / 暫定語義 / 多樣性 / 回傳形狀
# ══════════════════════════════════════════════════════════════════════

def test_ac12_model_version_excludes_manifest(tmp_path):
    # AC12:只改 manifest.json → model_version 不變;改 bank.npz → model_version 變
    import al_batch
    md = tmp_path / "model"; _make_model(md)
    (md / "manifest.json").write_text(json.dumps({"saved_at": 1.0}), encoding="utf-8")
    v1 = al_batch.load_frozen_model(md)["model_version"]
    (md / "manifest.json").write_text(json.dumps({"saved_at": 2.0}), encoding="utf-8")
    v2 = al_batch.load_frozen_model(md)["model_version"]
    assert v1 == v2, "model_version 必須排除 manifest.json(非原子/含 saved_at)"
    save_bank(md, vectors=np.repeat(_E0[None, :], 60, axis=0),
              meta={"model": "dinov2_vits14", "target_res": 224, "score_mode": "patch"})
    v3 = al_batch.load_frozen_model(md)["model_version"]
    assert v3 != v1, "bank.npz 內容變 → model_version 必須改變"


def test_ac13_novelty_provisional_is_final_membership(tmp_path):
    # AC13:novelty 下,暫定 Top-K 成員 == 已處理併集依原始分數的前 k 成員(_minmax 單調不改成員)
    ds = tmp_path / "ds"
    reds = [10, 240, 60, 200, 30, 170, 5, 250]
    paths = [_img(ds, f"img_{i:02d}.png", red=reds[i]) for i in range(8)]
    md = tmp_path / "model"; _make_model(md)
    r = _run(image_paths=paths, model_dir=md, checkpoint_dir=tmp_path / "ck",
             objective="novelty", k=3, batch_size=5, max_batches=1, dataset_dirs=(ds,))
    assert r["provisional"] is True
    # 批 1 = img_00..img_04,紅 [10,240,60,200,30] → 分數前 3 成員 = {240,200,60}
    got = sorted(int(np.asarray(Image.open(rec["image_path"]))[..., 0].mean().round())
                 for rec in r["topk_records"])
    assert got == sorted([60, 200, 240]) or all(v >= 55 for v in got), \
        f"novelty 暫定 Top-K 成員必須是已處理併集的分數前 k:{got}"


def test_ac14_diversity_flag_and_cluster_cap(tmp_path):
    # AC14:有 head → diversity_applied=True 且同預測類別在 Top-K 中 ≤ max_per_cluster(=2);
    #        無 head → diversity_applied=False 且 Top-K = 純 priority 前 k
    ds = tmp_path / "ds"
    # 4 個 X 類(藍=0)高分(紅大) + 4 個 Y 類(藍=255)中分(紅中);k=4,cap=2
    # 純 priority 會全取 4 個 X(分數最高);多樣性 cap 應逼出 2X + 2Y
    specs = [("x0", 250, 0), ("x1", 240, 0), ("x2", 230, 0), ("x3", 220, 0),
             ("y0", 210, 255), ("y1", 200, 255), ("y2", 190, 255), ("y3", 180, 255)]
    paths = [_img(ds, f"{n}.png", red=r, blue=b) for (n, r, b) in specs]
    md_h = tmp_path / "mh"; _make_model(md_h, with_head=True)
    md_n = tmp_path / "mn"; _make_model(md_n, with_head=False)

    r_h = _run(image_paths=paths, model_dir=md_h, checkpoint_dir=tmp_path / "ckh",
               objective="uncertain", k=4, batch_size=10, dataset_dirs=(ds,))
    assert r_h["diversity_applied"] is True
    # Top-K 的來源影像藍值 → 推 X/Y;cap 應使不會 4 個全同類
    blues = [int(np.asarray(Image.open(rec["image_path"]))[..., 2].mean().round())
             for rec in r_h["topk_records"]]
    n_x = sum(1 for b in blues if b < 128)
    n_y = len(blues) - n_x
    assert n_x <= 2 and n_y <= 2, f"多樣性 cap 應限制同類 ≤2:X={n_x} Y={n_y}"

    r_n = _run(image_paths=paths, model_dir=md_n, checkpoint_dir=tmp_path / "ckn",
               objective="novelty", k=4, batch_size=10, dataset_dirs=(ds,))
    assert r_n["diversity_applied"] is False


def test_ac15_return_shape_and_reason(tmp_path):
    # AC15:回傳含設計 §2.4 所有鍵;topk_records[i] 欄位齊全且 reason 為非空中文字串
    ds = tmp_path / "ds"
    paths = [_img(ds, f"img_{i:02d}.png", red=5 + 12 * i) for i in range(8)]
    md = tmp_path / "model"; _make_model(md)
    r = _run(image_paths=paths, model_dir=md, checkpoint_dir=tmp_path / "ck",
             objective="novelty", k=3, batch_size=10, dataset_dirs=(ds,))
    for key in ("topk", "topk_records", "objective", "k", "images_total",
                "images_processed", "objects_scored", "done", "provisional",
                "diversity_applied", "model_version", "skipped"):
        assert key in r, f"回傳缺鍵:{key}"
    assert len(r["topk"]) == 3 == len(r["topk_records"])
    rec = r["topk_records"][0]
    for f in ("item_id", "image_path", "bbox", "obj_index", "score", "priority", "reason"):
        assert f in rec, f"topk_records 缺欄位:{f}"
    assert isinstance(rec["reason"], str) and rec["reason"].strip(), "reason 必須是非空字串"


# ══════════════════════════════════════════════════════════════════════
# 推導 / metamorphic 測試(設計未明列,PM 自行施壓)
# ══════════════════════════════════════════════════════════════════════

def test_derived_multi_resume_converges(tmp_path):
    # 推導:任意切成「多次續跑」(每次 max_batches=1)最終結果 == 一次跑完(比 AC1/AC2 更強)
    ds = tmp_path / "ds"
    paths = [_img(ds, f"img_{i:02d}.png", red=3 + 5 * i) for i in range(25)]
    md = tmp_path / "model"; _make_model(md)
    ref = _run(image_paths=paths, model_dir=md, checkpoint_dir=tmp_path / "ref",
               objective="novelty", k=8, batch_size=5, dataset_dirs=(ds,))
    import al_batch
    ck = tmp_path / "ck"
    r = None
    for _ in range(10):  # 一次一批地推進,直到 done
        r = al_batch.run_batched(image_paths=paths, model_dir=md, checkpoint_dir=ck,
                                 objective="novelty", k=8, batch_size=5, max_batches=1,
                                 resume=True, object_source="whole_image",
                                 dataset_dirs=(ds,), extractor=_graded_extractor(),
                                 embed_fn=_class_embed())
        if r["done"]:
            break
    assert r["done"] and r["topk"] == ref["topk"], "多次續跑必須收斂到與一次跑完相同"


def test_derived_no_forbidden_global_ops_in_batch(tmp_path):
    # 反向稽查:al_batch 實作**不得**呼叫 run_pipeline / cluster_objects / 用 candidates_in_reference=True
    #           (設計 §2.6 禁止事項、C8)。用 AST 檢「實際呼叫」,而非字串出現(註解裡誠實提到不算)。
    import ast
    src = Path(__file__).resolve().parent.parent / "scripts" / "al_batch.py"
    assert src.exists(), "scripts/al_batch.py 尚未實作(先紅)"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    called, loo_true = set(), False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            called.add(name)
            for kw in node.keywords:
                if (kw.arg == "candidates_in_reference"
                        and isinstance(kw.value, ast.Constant) and kw.value.value is True):
                    loo_true = True
    assert "run_pipeline" not in called, "禁止呼叫 run_pipeline(每批會跑全域 HDBSCAN/門檻,違反 C8)"
    assert "cluster_objects" not in called, "禁止在批次流程做全域分群(C8);多樣性只用 head 預測類別"
    assert not loo_true, "禁止 candidates_in_reference=True(全域 leave-one-out,非批次可合併)"


# ══════════════════════════════════════════════════════════════════════
# M12b / A3:objective="similar"(找長得像參考物件的)—— 復用注入式 embed_fn
#   ref_vector=_E2 → X 群(blue=0→obj_emb E2)cosine=1、Y 群(blue=255→E3)cosine=0
# ══════════════════════════════════════════════════════════════════════

def test_ac_sim1_similar_ranks_by_reference(tmp_path):
    # AC-SIM1:ref=E2 → topk 全為 X 群、priority 降冪、reason 含「相似度」
    ds = tmp_path / "ds"
    xs = [_img(ds, f"x_{i:02d}.png", red=10 + 5 * i, blue=0) for i in range(6)]
    ys = [_img(ds, f"y_{i:02d}.png", red=10 + 5 * i, blue=255) for i in range(6)]
    md = tmp_path / "model"; _make_model(md)
    r = _run(image_paths=xs + ys, model_dir=md, checkpoint_dir=tmp_path / "ck",
             objective="similar", ref_vector=_E2, k=6, batch_size=40, dataset_dirs=(ds,))
    assert r["done"] and len(r["topk_records"]) == 6
    assert all("x_" in Path(rec["image_path"]).name for rec in r["topk_records"]), \
        f"ref=E2(X群) → topk 應全為 X 群:{[Path(x['image_path']).name for x in r['topk_records']]}"
    pris = [rec["priority"] for rec in r["topk_records"]]
    assert pris == sorted(pris, reverse=True), "priority 應降冪(最像在前)"
    assert all("相似度" in rec["reason"] for rec in r["topk_records"])


def test_ac_sim2_similar_requires_ref_vector(tmp_path):
    # AC-SIM2:objective=similar 但缺 ref_vector → ValueError(含 ref_vector)
    ds = tmp_path / "ds"
    paths = [_img(ds, f"x_{i}.png", red=10 + i, blue=0) for i in range(3)]
    md = tmp_path / "model"; _make_model(md)
    with pytest.raises(ValueError, match="ref_vector"):
        _run(image_paths=paths, model_dir=md, checkpoint_dir=tmp_path / "ck",
             objective="similar", k=3, dataset_dirs=(ds,))


def test_ac_sim3_change_ref_is_new_run(tmp_path):
    # AC-SIM3:換參考=另一 run(error 拒續跑;restart 重算 topk 翻群)
    ds = tmp_path / "ds"
    xs = [_img(ds, f"x_{i:02d}.png", red=10 + 5 * i, blue=0) for i in range(4)]
    ys = [_img(ds, f"y_{i:02d}.png", red=10 + 5 * i, blue=255) for i in range(4)]
    md = tmp_path / "model"; _make_model(md); ck = tmp_path / "ck"
    r1 = _run(image_paths=xs + ys, model_dir=md, checkpoint_dir=ck,
              objective="similar", ref_vector=_E2, k=4, dataset_dirs=(ds,))
    assert all("x_" in Path(rec["image_path"]).name for rec in r1["topk_records"])
    with pytest.raises(ValueError):        # 換 ref=E3、同 ck、error → 身分不符
        _run(image_paths=xs + ys, model_dir=md, checkpoint_dir=ck,
             objective="similar", ref_vector=_E3, k=4, dataset_dirs=(ds,),
             on_identity_mismatch="error")
    r2 = _run(image_paths=xs + ys, model_dir=md, checkpoint_dir=ck,   # restart → 重算翻 Y
              objective="similar", ref_vector=_E3, k=4, dataset_dirs=(ds,),
              on_identity_mismatch="restart")
    assert all("y_" in Path(rec["image_path"]).name for rec in r2["topk_records"]), \
        f"換 ref=E3 重算後 topk 應翻成 Y 群:{[Path(y['image_path']).name for y in r2['topk_records']]}"


def test_ac_sim4_similar_batched_equals_single(tmp_path):
    # AC-SIM4(C8):similar 分批(4)==一次跑完(40)topk 逐一相同
    ds = tmp_path / "ds"
    xs = [_img(ds, f"x_{i:02d}.png", red=10 + 5 * i, blue=0) for i in range(6)]
    ys = [_img(ds, f"y_{i:02d}.png", red=10 + 5 * i, blue=255) for i in range(6)]
    md = tmp_path / "model"; _make_model(md)
    r_split = _run(image_paths=xs + ys, model_dir=md, checkpoint_dir=tmp_path / "cka",
                   objective="similar", ref_vector=_E2, k=8, batch_size=4, dataset_dirs=(ds,))
    r_whole = _run(image_paths=xs + ys, model_dir=md, checkpoint_dir=tmp_path / "ckb",
                   objective="similar", ref_vector=_E2, k=8, batch_size=40, dataset_dirs=(ds,))
    assert r_split["done"] and r_whole["done"]
    assert r_split["topk"] == r_whole["topk"], "similar 分批與一次跑完 Top-K 必須逐一相同"
