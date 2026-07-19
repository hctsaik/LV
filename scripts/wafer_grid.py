"""晶圓 die 格點整形與點陣渲染(M22 模組 29,Tier A)。

metadata 列 + per-item 判定 → 逐片晶圓 die 格點結構(對不上逐項列名)+
該結構的 PIL 點陣地圖(匯出用,像素可驗)。
設計:3_Architect_Design/29_wafer_grid.md。純函式、無 I/O、無 streamlit。
G6 鐵則:零欄位假設——缺 wafer 欄且未獲使用者明示「同屬一片」→ ValueError,
絕不默默當同一片;對不上/缺座標逐項列名,絕不默默少畫。
"""
from __future__ import annotations

from typing import Mapping, Sequence

_WAFER_ALIASES = ("wafer", "wafer_id", "waferid", "wafer_no", "晶圓", "片號")
_X_ALIASES = ("die_x", "x", "col", "column", "diex")
_Y_ALIASES = ("die_y", "y", "row", "diey")
_LOT_ALIASES = ("lot", "lot_id", "lotid", "lot_no", "批號")

_COLOR_BAD = (220, 60, 60)
_COLOR_GOOD = (120, 160, 220)
_COLOR_BG = (245, 245, 245)

SINGLE_WAFER_ID = "(單片)"


def guess_fields(fieldnames: Sequence[str]) -> dict:
    """欄名 casefold 後與別名表完全相等才中;同表多欄取先出現者;沒中=None。"""
    def pick(aliases):
        for f in fieldnames:
            if str(f).casefold() in aliases:
                return f
        return None
    return {"wafer": pick(_WAFER_ALIASES), "x": pick(_X_ALIASES),
            "y": pick(_Y_ALIASES), "lot": pick(_LOT_ALIASES)}


def _parse_coord(row: Mapping, field: str):
    if field not in row:
        return None
    v = row[field]
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_wafer_grids(
    items: list, meta_lookup: Mapping[str, dict], *,
    wafer_field: str | None, x_field: str, y_field: str,
    lot_field: str | None = None,
    assume_single_wafer: bool = False,
) -> dict:
    if x_field is None or y_field is None:
        raise ValueError("缺座標欄位:x/y 欄必須指定")
    if wafer_field is None and not assume_single_wafer:
        raise ValueError("無 wafer 欄位:請指定欄位,或明示勾選「這批同屬一片」(工具不猜)")

    # die 聚合:一張影像=一顆 die;flag/score 取 max(score 全 None → None)
    per_path: dict = {}
    for it in items:
        p = str(it["path"])
        cur = per_path.setdefault(p, {"flag": 0, "score": None})
        cur["flag"] = max(cur["flag"], int(it["flag"]))
        s = it.get("score")
        if s is not None:
            cur["score"] = s if cur["score"] is None else max(cur["score"], float(s))

    unmatched: set = set()
    missing_coord: set = set()
    missing_wafer: set = set()
    groups: dict = {}       # wafer_id → {"dies": [...], "lots": set}
    for p in sorted(per_path):
        row = meta_lookup.get(p)
        if row is None:
            unmatched.add(p)
            continue
        x = _parse_coord(row, x_field)
        y = _parse_coord(row, y_field)
        if x is None or y is None:
            missing_coord.add(p)
            continue
        if wafer_field is not None:
            wv = row.get(wafer_field)
            if wv is None or str(wv).strip() == "":
                missing_wafer.add(p)
                continue
            wid = str(wv)
        else:
            wid = SINGLE_WAFER_ID
        g = groups.setdefault(wid, {"dies": [], "lots": set()})
        g["dies"].append({"x": x, "y": y, "flag": per_path[p]["flag"],
                          "path": p, "score": per_path[p]["score"]})
        if lot_field is not None:
            lv = row.get(lot_field)
            if lv is not None and str(lv).strip() != "":
                g["lots"].add(str(lv))

    wafers = []
    for wid in sorted(groups):
        dies = sorted(groups[wid]["dies"], key=lambda d: d["path"])
        lots = sorted(groups[wid]["lots"])
        wafers.append({"wafer_id": wid,
                       "lot": (lots[0] if lots else None),  # 同片多值取字典序最小
                       "dies": dies, "n": len(dies),
                       "k": sum(d["flag"] for d in dies)})
    return {"wafers": wafers,
            "unmatched": sorted(unmatched),
            "missing_coord": sorted(missing_coord),
            "missing_wafer": sorted(missing_wafer)}


def _axis_slots(values: list) -> dict:
    """座標值 → 格位:step=相鄰唯一值最小正差(唯一值 1 個時 step=1)。"""
    uniq = sorted(set(values))
    if len(uniq) == 1:
        step = 1.0
    else:
        step = min(b - a for a, b in zip(uniq, uniq[1:]))
    lo = uniq[0]
    return {v: round((v - lo) / step) for v in uniq}


def wafer_map_png(wafer: dict, *, cell: int = 24, pad: int = 2):
    from PIL import Image, ImageDraw
    dies = wafer.get("dies") or []
    if not dies:
        raise ValueError("dies 為空,無法渲染晶圓地圖")
    xs = _axis_slots([d["x"] for d in dies])
    ys = _axis_slots([d["y"] for d in dies])
    ncol = max(xs.values()) + 1
    nrow = max(ys.values()) + 1
    # 同格重複 die → flag 取 max 後畫一次
    cells: dict = {}
    for d in dies:
        key = (xs[d["x"]], ys[d["y"]])
        cells[key] = max(cells.get(key, 0), int(d["flag"]))
    img = Image.new("RGB", (ncol * cell + 2 * pad, nrow * cell + 2 * pad), _COLOR_BG)
    draw = ImageDraw.Draw(img)
    for (cx, cy), flag in cells.items():
        x0 = pad + cx * cell
        y0 = pad + cy * cell
        draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1],
                       fill=(_COLOR_BAD if flag else _COLOR_GOOD))
    return img
