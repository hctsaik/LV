"""驗收:al_service(設計 3_Architect_Design/10_al_service.md)。

先紅:`scripts/al_service.py` 尚未實作(import 放函式內 → 執行期紅)。
合成資料 + 注入 graded extractor(紅→分數)+ 凍結 bank(save_bank);now 注入固定 run_id;不觸真 DINOv2。全 tmp 目錄。

契約鎖點:增量只 embed 新圖(AC2)、標註閉環移出(AC3)、換版重評分 item_id 原地(AC4)、
鎖釋放/error 不殘留(AC5)、不寫來源(AC6)、單實例(AC7)。
"""
import json
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from anomaly_bank_store import save_bank

_D = 384


def _e(i):
    v = np.zeros(_D, dtype=np.float32)
    v[i] = 1.0
    return v


_E0, _E1 = _e(0), _e(1)


def _unit(v):
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n else v.astype(np.float32)


def _graded_extractor(counter=None):
    """紅通道 → 一個 patch,與 bank-A(E0)距離隨紅單調(紅越多越異常)。"""
    def ext(crop):
        if counter is not None:
            counter["n"] += 1
        arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
        s = float(arr[..., 0].mean()) / 255.0
        return _unit((1.0 - s) * _E0 + s * _E1)[None, :], (1, 1)
    return ext


def _make_model(model_dir, *, n_vec=50):
    save_bank(model_dir, vectors=np.repeat(_E0[None, :], n_vec, axis=0),
              meta={"model": "dinov2_vits14", "target_res": 224, "score_mode": "patch",
                    "object_source": "whole_image"})


def _img(folder: Path, name: str, red: int):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    Image.new("RGB", (32, 32), (int(red), 120, 0)).save(p)
    return p


def _svc():
    import al_service
    return al_service


# ══════════════════════════════════════════════════════════════════════

def test_ac1_first_round(tmp_path):
    # AC1:首輪 → queue 有 min(k,objects) 筆、高紅排最前、欄位齊全、runs/<id> 存在
    svc = _svc()
    import al_workspace as ws
    src = tmp_path / "src"
    for i, r in enumerate([20, 250, 40, 240, 60, 30]):   # 2 張高紅(250,240)
        _img(src, f"o{i}.jpg", r)
    mdl = tmp_path / "mdl"; _make_model(mdl)
    wsd = tmp_path / "wsd"
    svc.init_workspace(wsd, name="p1", watch_folders=[str(src)], model_dir=str(mdl), k=4)
    res = svc.run_once(wsd, extractor=_graded_extractor(), now=1000.0)
    assert res["status"] == "ok", res
    q = ws.read_queue(wsd)
    assert len(q) == 4
    reds = [int(np.asarray(Image.open(r["image_path"]))[..., 0].mean().round()) for r in q]
    assert reds[0] >= 235 and reds[1] >= 235, f"高紅應排最前:{reds}"
    for f in ("id", "score", "reason", "model_version", "run_id", "status"):
        assert f in q[0], f"queue 項缺欄位:{f}"
    assert q[0]["status"] == "pending"
    assert (wsd / "runs" / f"{res['run_id']}.json").exists()


def test_ac2_incremental_only_embeds_new(tmp_path):
    # AC2:第二輪只 embed 新圖(舊圖靠 .lv_cache)→ extractor 增量 == 新圖數
    svc = _svc()
    src = tmp_path / "src"
    for i, r in enumerate([20, 250, 40, 240, 60, 30]):
        _img(src, f"o{i}.jpg", r)
    mdl = tmp_path / "mdl"; _make_model(mdl)
    wsd = tmp_path / "wsd"
    svc.init_workspace(wsd, name="p", watch_folders=[str(src)], model_dir=str(mdl), k=10)
    svc.run_once(wsd, extractor=_graded_extractor(), now=1000.0)   # 首輪 embed 6
    for i, r in enumerate([200, 210, 190]):
        _img(src, f"n{i}.jpg", r)                                  # 加 3 新圖
    c = {"n": 0}
    res = svc.run_once(wsd, extractor=_graded_extractor(c), now=2000.0)
    assert res["status"] == "ok"
    assert c["n"] <= 3, f"第二輪只該 embed 3 張新圖(舊圖快取),實際 extractor 呼叫 {c['n']}"
    assert res["new"] == 3


def test_ac3_labels_remove_from_queue(tmp_path):
    # AC3:標註前 2 筆 → 下一輪那 2 id 不在 queue
    svc = _svc()
    import al_workspace as ws
    src = tmp_path / "src"
    for i, r in enumerate([20, 250, 40, 240, 60, 30, 200, 90]):
        _img(src, f"o{i}.jpg", r)
    mdl = tmp_path / "mdl"; _make_model(mdl)
    wsd = tmp_path / "wsd"
    svc.init_workspace(wsd, name="p", watch_folders=[str(src)], model_dir=str(mdl), k=5)
    svc.run_once(wsd, extractor=_graded_extractor(), now=1000.0)
    q1 = ws.read_queue(wsd)
    labeled = [q1[0]["id"], q1[1]["id"]]
    ws.append_label(wsd, {"id": labeled[0], "decision": "good"})
    ws.append_label(wsd, {"id": labeled[1], "decision": "defect"})
    svc.run_once(wsd, extractor=_graded_extractor(), now=2000.0)
    q2_ids = {r["id"] for r in ws.read_queue(wsd)}
    assert not (set(labeled) & q2_ids), "已標註的物件不得再出現在佇列"


def test_ac4_model_version_rescore_inplace(tmp_path):
    # AC4:換內容不同但相容的 model_dir → model_version 變、同物件 id 不變(原地)
    svc = _svc()
    import al_workspace as ws
    src = tmp_path / "src"
    for i, r in enumerate([20, 250, 40, 240]):
        _img(src, f"o{i}.jpg", r)
    mdl = tmp_path / "mdl"; _make_model(mdl, n_vec=50)
    wsd = tmp_path / "wsd"
    svc.init_workspace(wsd, name="p", watch_folders=[str(src)], model_dir=str(mdl), k=4)
    svc.run_once(wsd, extractor=_graded_extractor(), now=1000.0)
    q1 = ws.read_queue(wsd)
    v1 = q1[0]["model_version"]
    ids1 = {r["id"] for r in q1}
    _make_model(mdl, n_vec=77)     # 內容變(bank 列數變)→ model_version 變
    svc.run_once(wsd, extractor=_graded_extractor(), now=2000.0)
    q2 = ws.read_queue(wsd)
    assert q2[0]["model_version"] != v1, "換版後 model_version 應改變"
    assert {r["id"] for r in q2} == ids1, "同物件 id 應不變(內容定址、原地更新,非新增)"


def test_ac5_bad_input(tmp_path):
    # AC5:(a) 無 profile → error、無鎖/queue/state;(b) model 缺 → error、鎖已釋放
    svc = _svc()
    wsd = tmp_path / "wsd"     # 未 init → 無 profile
    res = svc.run_once(wsd, extractor=_graded_extractor(), now=1000.0)
    assert res["status"] == "error"
    assert not (wsd / "lock").exists() and not (wsd / "queue.jsonl").exists()
    # (b) profile 有但 model_dir 缺
    src = tmp_path / "src"; _img(src, "a.jpg", 100)
    svc.init_workspace(wsd, name="p", watch_folders=[str(src)],
                       model_dir=str(tmp_path / "nonexist_mdl"), k=4)
    res2 = svc.run_once(wsd, extractor=_graded_extractor(), now=1000.0)
    assert res2["status"] == "error"
    assert not (wsd / "lock").exists(), "error 後鎖必須已釋放(不殘留)"


def test_ac6_never_writes_source(tmp_path):
    # AC6:run_once 跑完來源快照零變化
    svc = _svc()
    src = tmp_path / "src"
    for i, r in enumerate([20, 250, 40]):
        _img(src, f"o{i}.jpg", r)
    mdl = tmp_path / "mdl"; _make_model(mdl)
    wsd = tmp_path / "wsd"
    svc.init_workspace(wsd, name="p", watch_folders=[str(src)], model_dir=str(mdl), k=4)
    snap = {p: p.stat().st_mtime_ns for p in src.rglob("*")}
    svc.run_once(wsd, extractor=_graded_extractor(), now=1000.0)
    after = {p: p.stat().st_mtime_ns for p in src.rglob("*")}
    assert after == snap, "不可寫/動來源資料夾"


def test_ac7_single_instance_lock(tmp_path):
    # AC7:新鮮 lock(他人 pid)存在 → skipped、不動 queue/state;移除後 → ok
    svc = _svc()
    import al_workspace as ws
    import time as _t
    src = tmp_path / "src"
    for i, r in enumerate([20, 250, 40]):
        _img(src, f"o{i}.jpg", r)
    mdl = tmp_path / "mdl"; _make_model(mdl)
    wsd = tmp_path / "wsd"
    svc.init_workspace(wsd, name="p", watch_folders=[str(src)], model_dir=str(mdl), k=4)
    ws.acquire_lock(wsd, pid=999999, started=_t.time())     # 他人新鮮鎖
    res = svc.run_once(wsd, extractor=_graded_extractor(), now=_t.time())
    assert res["status"] == "skipped"
    assert not (wsd / "queue.jsonl").exists(), "被鎖時不得寫佇列"
    ws.release_lock(wsd, pid=999999)
    res2 = svc.run_once(wsd, extractor=_graded_extractor(), now=_t.time())
    assert res2["status"] == "ok"


def test_ac8_cli_exit_codes(tmp_path):
    # AC8:main run ok→0、run 無 profile→非0、init 建 profile.yaml
    svc = _svc()
    src = tmp_path / "src"; _img(src, "a.jpg", 250)
    mdl = tmp_path / "mdl"; _make_model(mdl)
    wsd = tmp_path / "wsd"
    rc_init = svc.main(["init", "--profile", str(wsd), "--model-dir", str(mdl),
                        "--watch", str(src), "--k", "4"])
    assert rc_init == 0 and (wsd / "profile.yaml").exists()
    # CLI run 走真實 DINOv2(無注入)——本測只驗 init + 壞路徑退出碼,避免真模型;
    # run 的完整行為由 AC1–AC7(注入)涵蓋。
    rc_bad = svc.main(["run", "--profile", str(tmp_path / "no_such_ws")])
    assert rc_bad != 0, "無 profile 的 run 應回非0"


# ══════════════════════════════════════════════════════════════════════
# M12b / A3:離線監看服務 similar(profile 存參考向量 + run_once 依相似排序)
# ══════════════════════════════════════════════════════════════════════

def _img_rb(folder: Path, name: str, red: int, blue: int):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    Image.new("RGB", (32, 32), (int(red), 120, int(blue))).save(p)
    return p


def _class_embed():
    """藍通道 → obj_emb 落 E2(blue=0)或 E3(blue=255)。"""
    def emb(crop):
        arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
        b = float(arr[..., 2].mean()) / 255.0
        return _unit((1.0 - b) * _e(2) + b * _e(3))
    return emb


def test_ac_svc_sim_similar_profile_and_ranking(tmp_path):
    # AC-SVC-SIM:init 存 reference.npy + profile.reference_vector_file;run_once similar → 佇列以參考同群為主
    svc = _svc()
    import al_workspace as ws
    src = tmp_path / "src"
    for i in range(3):
        _img_rb(src, f"x_{i}.jpg", red=100, blue=0)     # X 群(E2)
    for i in range(3):
        _img_rb(src, f"y_{i}.jpg", red=100, blue=255)   # Y 群(E3)
    mdl = tmp_path / "mdl"; _make_model(mdl)
    wsd = tmp_path / "wsd"
    svc.init_workspace(wsd, name="sim", watch_folders=[str(src)], model_dir=str(mdl),
                       k=3, objective="similar", reference_vector=_e(2))
    prof = ws.load_profile(wsd)
    assert prof.get("reference_vector_file") == "reference.npy", prof
    assert (wsd / "reference.npy").exists()
    res = svc.run_once(wsd, extractor=_graded_extractor(), embed_fn=_class_embed(), now=1000.0)
    assert res["status"] == "ok", res
    q = ws.read_queue(wsd)
    assert len(q) == 3
    xs = sum(1 for it in q if "x_" in Path(it["image_path"]).name)
    assert xs >= 2, f"參考=X 群 → 佇列應以 X 為主;實際 {[Path(it['image_path']).name for it in q]}"
