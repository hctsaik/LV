"""anomaly_bank_store:bank profile 目錄的序列化/反序列化 + 相容鍵 + few-shot 內容定址。

一個 bank profile = 一個目錄,四個檔(meta 最後寫當完整性哨兵):
  <bank_dir>/
    bank.npz         # patch coreset 向量(評分用;object 模式可無)
    projection.npz   # 投影器(anomaly_project 格式;投影/密度用)
    fewshot.json     # few-shot good/bad 標記(內容定址,跨資料夾可對齊)
    meta.json        # 相容鍵 + 來源指紋(最後寫,存在=profile 完整)

鐵則:絕不寫使用者資料集 —— assert_safe_bank_dir 正向白名單防呆。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
_REPO = Path(__file__).resolve().parent.parent
_BBOX_TOL = 0.02      # 量化後歸一化 bbox 的 L∞ 容忍(吸收重存 label 的浮點差)


# ── 序列化 ────────────────────────────────────────────────────────────────
def save_bank(bank_dir, *, vectors=None, projection=None, meta: dict,
              fewshot=None) -> None:
    """寫一個 bank profile 目錄。meta.json 最後寫:任何資料檔半途失敗 → 無 meta → 視為不完整。"""
    d = Path(bank_dir)
    d.mkdir(parents=True, exist_ok=True)

    # 空 bank 早擋(在清舊檔前,避免清掉舊 profile 卻沒寫新的)
    if vectors is not None:
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        if v.ndim != 2 or v.shape[0] == 0:
            raise ValueError("空 bank 拒存(0 列向量)")

    # 重存先清舊 profile:先移除 meta.json(哨兵失效 → 期間視為不完整),再清舊 data 檔。
    # 否則上一版殘留檔(例如 patch→object 重存後殘留的 bank.npz)會被新 meta 收編 → silent mismatch。
    for _fn in ("meta.json", "bank.npz", "projection.npz", "fewshot.json"):
        try:
            (d / _fn).unlink()
        except FileNotFoundError:
            pass

    if vectors is not None:
        _atomic_npz(d / "bank.npz", vectors=v)

    if projection is not None:
        from anomaly_project import save_projector
        save_projector(d / "projection.npz", projection)

    if fewshot is not None:
        _atomic_text(d / "fewshot.json",
                     json.dumps(list(fewshot), ensure_ascii=False, indent=2))

    m = dict(meta)
    m.setdefault("schema_version", SCHEMA_VERSION)
    _atomic_text(d / "meta.json", json.dumps(m, ensure_ascii=False, indent=2))  # 最後寫


def load_bank(bank_dir) -> dict:
    """讀回 {meta, vectors(or None), projection(or None), fewshot(or None)}。
    缺 meta.json → 視為不完整,raise。"""
    d = Path(bank_dir)
    mp = d / "meta.json"
    if not mp.exists():
        raise FileNotFoundError(f"bank profile 不完整(缺 meta.json):{d}")
    meta = json.loads(mp.read_text(encoding="utf-8"))

    vectors = None
    if (d / "bank.npz").exists():
        vectors = np.load(d / "bank.npz")["vectors"].astype(np.float32)

    projection = None
    if (d / "projection.npz").exists():
        from anomaly_project import load_projector
        projection = load_projector(d / "projection.npz")

    fewshot = None
    if (d / "fewshot.json").exists():
        fewshot = json.loads((d / "fewshot.json").read_text(encoding="utf-8"))

    return {"meta": meta, "vectors": vectors, "projection": projection,
            "fewshot": fewshot}


def check_compat(meta: dict, *, model: str, target_res, patch_dim=None,
                 obj_dim=None) -> bool:
    """硬鍵不符即拒載(擋 silent-wrong / 維度 broadcasting):model、target_res 必相等;
    patch_dim / obj_dim 若 meta 有記且不符也拒。軟鍵(budget/seed/fingerprint)不在此檢。"""
    if str(meta.get("model")) != str(model):
        raise ValueError(f"bank 不相容:model 期望 {meta.get('model')} 但當前 {model}")
    if str(meta.get("target_res")) != str(target_res):
        raise ValueError(f"bank 不相容:target_res 期望 {meta.get('target_res')} 但當前 {target_res}")
    if patch_dim is not None and meta.get("patch_dim") not in (None, patch_dim):
        raise ValueError(f"bank 不相容:patch_dim {meta.get('patch_dim')} vs {patch_dim}")
    if obj_dim is not None and meta.get("obj_dim") not in (None, obj_dim):
        raise ValueError(f"bank 不相容:obj_dim {meta.get('obj_dim')} vs {obj_dim}")
    return True


# ── few-shot 內容定址 ──────────────────────────────────────────────────────
def _stem(p) -> str:
    return Path(str(p)).stem


def confirmed_to_fewshot(confirmed: dict, records: list) -> list:
    """session 的 {物件整數索引: 'good'/'bad'} → 內容定址 fewshot list。
    主鍵 (stem, 量化 bbox, label):跨資料夾/重掃才對得回去(int 索引綁當次掃描順序,跨檔無意義)。"""
    out = []
    n = len(records)
    for i, verdict in confirmed.items():
        i = int(i)
        if not (0 <= i < n):              # 殘留/跨資料夾索引超界 → 跳過(不崩、也不誤把別物件當錨點)
            continue
        r = records[i]
        out.append({
            "stem": _stem(r["image_path"]),
            "bbox": [round(float(x), 3) for x in r["bbox"]],
            "label": str(r.get("label", "")),
            "verdict": str(verdict),
        })
    return out


def match_fewshot_to_indices(fewshot: list, records: list):
    """把 fewshot 對齊到 records 的整數索引,回 ({index: verdict}, unmatched 數)。
    先用 (stem, label) 候選、再用量化 bbox 反查最近者(救回行序變動);
    bbox 差超過容忍 → 視為對不到(計入 unmatched,不靜默貼錯)。"""
    by_key: dict = {}
    for idx, r in enumerate(records):
        by_key.setdefault((_stem(r["image_path"]), str(r.get("label", ""))), []).append(
            (idx, [float(x) for x in r["bbox"]]))

    matched: dict = {}
    unmatched = 0
    for fs in fewshot:
        cands = by_key.get((fs["stem"], str(fs.get("label", ""))), [])
        target = [float(x) for x in fs["bbox"]]
        best_idx, best_d = None, None
        for idx, bb in cands:
            if idx in matched:
                continue
            d = max(abs(a - b) for a, b in zip(bb, target))     # L∞ on 歸一化 bbox
            if best_d is None or d < best_d:
                best_idx, best_d = idx, d
        if best_idx is not None and best_d <= _BBOX_TOL:
            matched[best_idx] = fs["verdict"]
        else:
            unmatched += 1
    return matched, unmatched


# ── 白名單防呆(絕不寫使用者資料集)──────────────────────────────────────────
def _under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_safe_bank_dir(bank_dir, selected_folders=()) -> bool:
    """允許落點:(a) .lv_cache(LV_CACHE_DIR 或 <repo>/.lv_cache)下,或
    (b) 不含 images//labels/ 子目錄、且非任一已選資料夾之祖先/後代/本身的乾淨目錄。否則 raise。"""
    p = Path(bank_dir).resolve()
    cache_root = Path(os.environ.get("LV_CACHE_DIR") or (_REPO / ".lv_cache")).resolve()
    if _under(p, cache_root):
        return True
    if (p / "images").exists() or (p / "labels").exists():
        raise ValueError(f"目標看起來是資料集(含 images//labels/),拒寫:{p}")
    # 扁平 YOLO 佈局(影像 + .txt 標籤同層,無 images//labels/ 子目錄)也是資料集 → 拒寫
    if p.exists() and any(p.glob("*.txt")) and any(
            p.glob(e) for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp")):
        raise ValueError(f"目標看起來是扁平 YOLO 資料集(同層有影像+標籤 .txt),拒寫:{p}")
    for f in selected_folders:
        fp = Path(f).resolve()
        if p == fp or _under(p, fp) or _under(fp, p):
            raise ValueError(f"目標與資料集 {fp} 重疊(祖先/後代/本身),拒寫:{p}")
    return True


# ── atomic 寫(tmp 同目錄 → os.replace)────────────────────────────────────
def _atomic_text(final: Path, text: str) -> None:
    tmp = final.parent / ("_tmp_" + final.name)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, final)


def _atomic_npz(final: Path, **arrays) -> None:
    tmp = final.parent / ("_tmp_" + final.name)        # 以 .npz 結尾 → savez 不再加副檔
    np.savez_compressed(str(tmp), **arrays)
    os.replace(tmp, final)
