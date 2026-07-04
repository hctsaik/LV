"""非 YOLO-txt 標註格式的解析(COCO JSON / Pascal VOC XML / LabelMe / NDJSON)。

**依賴契約:pure stdlib + PIL(+safe_io,同為 stdlib+PIL)。**
export_subset 與外部宿主(如 Labeling sidecar)會在沒有 numpy/sklearn/hnswlib
的環境 import 本模組——不要在此加重依賴;GUI/embedding 邏輯留在 interaction.py
(interaction 會 re-export 本模組的公開名,app 端沿用 interaction 入口即可)。

統一入口 `annotation_boxes_for_image()`:依 COCO→VOC→LabelMe→NDJSON 順序查詢,
回 rows [(cid|None, cx, cy, w, h, score, name|None), …](0-1 正規化);
找不到任何標註**來源**回 None(有來源但該影像無框回 [],與 YOLO「檔在但空」一致)。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from PIL import Image

from safe_io import safe_read_text

# ── COCO JSON ────────────────────────────────────────────────────────────────
_COCO_FILENAMES = ("_annotations.coco.json", "annotations.json", "coco.json")


def coco_file_for_root(root: Path) -> Path | None:
    """資料夾根層的 COCO 標註檔(Roboflow `_annotations.coco.json` 等常見名)。"""
    for name in _COCO_FILENAMES:
        p = Path(root) / name
        if p.exists():
            return p
    return None


@lru_cache(maxsize=16)
def _load_coco_cached(json_path_str: str, mtime_ns: int) -> dict:
    """解析 COCO JSON → {"by_name": {檔名: [(cid,cx,cy,w,h,score,name),…]},
    "names_by_id": {id: name}}。bbox 由絕對像素 [x,y,w,h] 正規化(用 images 的
    width/height;缺尺寸的影像條目略過)。壞 JSON 回空(不 raise)。"""
    try:
        data = json.loads(safe_read_text(Path(json_path_str)))
    except (json.JSONDecodeError, OSError):
        return {"by_name": {}, "names_by_id": {}}
    cats = {int(c["id"]): str(c.get("name", f"class_{c['id']}"))
            for c in data.get("categories", []) if "id" in c}
    imgs = {im.get("id"): im for im in data.get("images", [])}
    by_name: dict[str, list] = {}
    for a in data.get("annotations", []):
        im = imgs.get(a.get("image_id"))
        bb = a.get("bbox")
        if not im or not bb or len(bb) < 4:
            continue
        W, H = im.get("width") or 0, im.get("height") or 0
        if not W or not H:
            continue
        try:
            x, y, w, h = (float(v) for v in bb[:4])
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        cid = int(a.get("category_id", -1))
        score = float(a["score"]) if "score" in a else None
        fname = Path(str(im.get("file_name", ""))).name
        by_name.setdefault(fname, []).append(
            (cid, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H,
             score, cats.get(cid)))
    return {"by_name": by_name, "names_by_id": cats}


def _coco_for_image(image_path: Path) -> dict | None:
    """影像所屬資料夾(本層或上一層,涵蓋平鋪與 images/ 兩種佈局)的 COCO 標註。"""
    for root in (image_path.parent, image_path.parent.parent):
        jf = coco_file_for_root(root)
        if jf:
            try:
                return _load_coco_cached(str(jf), jf.stat().st_mtime_ns)
            except OSError:
                return None
    return None


def coco_boxes_for_image(image_path: Path) -> list[tuple] | None:
    """該影像的 COCO 框 [(cid,cx,cy,w,h,score,name),…];資料夾無 COCO 檔回 None
    (有檔但該影像無標註回 [],語義與 YOLO txt「檔在但空」一致)。"""
    coco = _coco_for_image(Path(image_path))
    if coco is None:
        return None
    return coco["by_name"].get(Path(image_path).name, [])


# ── VOC XML / LabelMe JSON / NDJSON(名稱制標註:無數字 id,cid 由
#    discover_yolo_objects 以類名穩定補號)────────────────────────────────

def _voc_xml_for_image(image_path: Path) -> Path | None:
    """該影像的 Pascal VOC XML:同層同名(Roboflow VOC)或 Annotations/ 慣例位置
    (含經典 VOC 的 JPEGImages/ 佈局 → 上一層的 Annotations/)。"""
    ip = Path(image_path)
    for cand in (
        ip.with_suffix(".xml"),
        ip.parent / "Annotations" / f"{ip.stem}.xml",
        ip.parent / "annotations" / f"{ip.stem}.xml",
        ip.parent.parent / "Annotations" / f"{ip.stem}.xml",
        ip.parent.parent / "annotations" / f"{ip.stem}.xml",
    ):
        if cand.exists():
            return cand
    return None


def voc_xml_for_image(image_path: Path) -> Path | None:
    """公開別名(export_subset 的原格式標註保留用)。"""
    return _voc_xml_for_image(image_path)


def parse_voc_boxes(xml_path: Path, img_path: Path | None = None) -> list[tuple]:
    """Pascal VOC XML → [(None, cx, cy, w, h, None, name), …](0-1 正規化)。
    <size> 缺席/為 0 時讀影像標頭補尺寸;壞 XML/壞影像回 [](不 raise)。"""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(safe_read_text(Path(xml_path)))
    except ET.ParseError:
        return []
    W = H = 0
    size = root.find("size")
    if size is not None:
        try:
            W = int(float(size.findtext("width") or 0))
            H = int(float(size.findtext("height") or 0))
        except (TypeError, ValueError):
            W = H = 0
    if (not W or not H) and img_path is not None:
        try:
            with Image.open(img_path) as im:  # 只讀標頭取尺寸
                W, H = im.size
        except (OSError, Image.DecompressionBombError):
            return []
    if not W or not H:
        return []
    out: list[tuple] = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip() or "unknown"
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x0 = float(bb.findtext("xmin") or "")
            y0 = float(bb.findtext("ymin") or "")
            x1 = float(bb.findtext("xmax") or "")
            y1 = float(bb.findtext("ymax") or "")
        except (TypeError, ValueError):
            continue
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        out.append((None, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H,
                    w / W, h / H, None, name))
    return out


def parse_labelme_boxes(json_path: Path) -> list[tuple] | None:
    """LabelMe 同名 .json → [(None, cx, cy, w, h, None, label), …]。
    rectangle 用兩角;polygon 等多點 shape 取外接框。
    非 LabelMe 結構(無 shapes list)回 None(讓呼叫端試其他來源)。"""
    try:
        data = json.loads(safe_read_text(Path(json_path)))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
        return None
    W = data.get("imageWidth") or 0
    H = data.get("imageHeight") or 0
    if not W or not H:
        return []
    out: list[tuple] = []
    for s in data["shapes"]:
        pts = s.get("points") or []
        if not isinstance(pts, list) or len(pts) < 2:
            continue
        try:
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
        except (TypeError, ValueError, IndexError):
            continue
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        out.append((None, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H,
                    w / W, h / H, None, str(s.get("label") or "unknown")))
    return out


_NDJSON_IMG_KEYS = ("image", "image_path", "file_name", "filename", "source-ref", "path")
_NDJSON_BOX_KEYS = ("boxes", "annotations", "objects", "bboxes")
_NDJSON_LABEL_KEYS = ("label", "class", "name", "category")
_NDJSON_EXCLUDE = {"manifest.jsonl"}  # 本 app 的資料契約檔(path/sha…),非標註


@lru_cache(maxsize=32)
def _ndjson_qualifies(path_str: str, mtime_ns: int) -> bool:
    """抽樣前 50 行:至少一行同時有影像鍵與「含 dict 的框清單」才算標註檔。
    擋掉只有 path/sha 的資料契約類 jsonl(如本 app 的 manifest.jsonl)誤判。"""
    try:
        text = safe_read_text(Path(path_str))
    except OSError:
        return False
    for line in text.splitlines()[:50]:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if not any(isinstance(row.get(k), str) and row.get(k)
                   for k in _NDJSON_IMG_KEYS):
            continue
        if any(isinstance(row.get(k), list)
               and any(isinstance(x, dict) for x in row[k])
               for k in _NDJSON_BOX_KEYS):
            return True
    return False


def ndjson_file_for_root(root: Path) -> Path | None:
    """資料夾根層的 NDJSON/JSONL **標註**檔(內容驗證過;取字典序第一個合格者)。"""
    root = Path(root)
    if not root.is_dir():
        return None
    for pat in ("*.ndjson", "*.jsonl"):
        for p in sorted(root.glob(pat)):
            if p.name.lower() in _NDJSON_EXCLUDE:
                continue
            try:
                if _ndjson_qualifies(str(p), p.stat().st_mtime_ns):
                    return p
            except OSError:
                continue
    return None


@lru_cache(maxsize=16)
def _load_ndjson_cached(json_path_str: str, mtime_ns: int) -> dict:
    """NDJSON(一行一影像)→ {"by_name": {檔名: [(None,cx,cy,w,h,score,label),…]}}。
    寬容 schema:影像鍵 image/file_name/filename/image_path/source-ref/path;
    框清單鍵 boxes/annotations/objects/bboxes;每框 label 鍵 label/class/name/category
    (或 class_id → class_<id>);bbox 收四種:bbox=[x,y,w,h](像素)、
    xmin/ymin/xmax/ymax、left/top/width/height(SageMaker 風)、cx/cy/w/h(0-1)。
    像素形需行內 width/height(或 image_size:[{width,height}])。壞行跳過。"""
    by_name: dict[str, list] = {}
    try:
        text = safe_read_text(Path(json_path_str))
    except OSError:
        return {"by_name": {}}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        fname = next((Path(str(row[k]).replace("\\", "/")).name
                      for k in _NDJSON_IMG_KEYS
                      if isinstance(row.get(k), str) and row.get(k)), None)
        if not fname:
            continue
        W, H = row.get("width") or 0, row.get("height") or 0
        if not W or not H:
            isz = row.get("image_size")
            if isinstance(isz, list) and isz and isinstance(isz[0], dict):
                W, H = isz[0].get("width") or 0, isz[0].get("height") or 0
        boxes = next((row[k] for k in _NDJSON_BOX_KEYS
                      if isinstance(row.get(k), list)), [])
        rows: list[tuple] = []
        for b in boxes:
            if not isinstance(b, dict):
                continue
            label = next((str(b[k]) for k in _NDJSON_LABEL_KEYS if b.get(k)), None)
            if label is None and b.get("class_id") is not None:
                label = f"class_{b['class_id']}"
            label = label or "unknown"
            got = None
            try:
                bb = b.get("bbox")
                if isinstance(bb, list) and len(bb) >= 4 and W and H:
                    x, y, w, h = (float(v) for v in bb[:4])
                    if w > 0 and h > 0:
                        got = ((x + w / 2) / W, (y + h / 2) / H, w / W, h / H)
                if got is None and W and H and all(
                        k in b for k in ("xmin", "ymin", "xmax", "ymax")):
                    x0, y0 = float(b["xmin"]), float(b["ymin"])
                    x1, y1 = float(b["xmax"]), float(b["ymax"])
                    if x1 > x0 and y1 > y0:
                        got = ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H,
                               (x1 - x0) / W, (y1 - y0) / H)
                if got is None and W and H and all(
                        k in b for k in ("left", "top", "width", "height")):
                    x, y = float(b["left"]), float(b["top"])
                    w, h = float(b["width"]), float(b["height"])
                    if w > 0 and h > 0:
                        got = ((x + w / 2) / W, (y + h / 2) / H, w / W, h / H)
                if got is None and all(k in b for k in ("cx", "cy", "w", "h")):
                    cx, cy = float(b["cx"]), float(b["cy"])
                    w, h = float(b["w"]), float(b["h"])
                    if 0 < w <= 1 and 0 < h <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1:
                        got = (cx, cy, w, h)
            except (TypeError, ValueError):
                continue
            if got:
                rows.append((None, *got, None, label))
        by_name[fname] = rows  # 空也記:有來源但該影像無框
    return {"by_name": by_name}


# ── 統一入口 ─────────────────────────────────────────────────────────────────
def annotation_boxes_for_image(image_path: Path) -> list[tuple] | None:
    """非 YOLO-txt 標註的統一查詢:COCO JSON → VOC XML → LabelMe JSON → NDJSON。
    回 [(cid|None, cx, cy, w, h, score, name|None), …];找不到任何標註**來源**回 None
    (有來源但該影像無框回 [],與 YOLO「檔在但空」語義一致)。"""
    ip = Path(image_path)
    rows = coco_boxes_for_image(ip)
    if rows is not None:
        return rows
    xml = _voc_xml_for_image(ip)
    if xml is not None:
        return parse_voc_boxes(xml, ip)
    lm = ip.with_suffix(".json")
    if lm.exists():
        rows = parse_labelme_boxes(lm)
        if rows is not None:
            return rows
    for root in (ip.parent, ip.parent.parent):
        nd = ndjson_file_for_root(root)
        if nd:
            try:
                data = _load_ndjson_cached(str(nd), nd.stat().st_mtime_ns)
            except OSError:
                return None
            return data["by_name"].get(ip.name, [])
    return None


def annotation_image_label(image_path: Path) -> str | None:
    """整張影像級的標註語義(COCO/VOC/LabelMe/NDJSON 共用):單一類→類名、
    多類→'mix'、無標註→'unknown';無任何標註來源回 None(走 YOLO txt 邏輯)。"""
    boxes = annotation_boxes_for_image(image_path)
    if boxes is None:
        return None
    names = {(b[6] or (f"class_{b[0]}" if b[0] is not None else "unknown"))
             for b in boxes}
    if not names:
        return "unknown"
    return names.pop() if len(names) == 1 else "mix"


def folder_has_annotations(folder: Path, probe: int = 25) -> bool:
    """資料夾是否帶任一支援的**非 YOLO-txt** 標註(COCO/NDJSON 根層檔、
    Annotations/*.xml、或抽樣影像的同名 XML/LabelMe JSON sidecar)。"""
    folder = Path(folder)
    if coco_file_for_root(folder) or ndjson_file_for_root(folder):
        return True
    for sub in ("Annotations", "annotations"):
        d = folder / sub
        if d.is_dir() and next(iter(d.glob("*.xml")), None) is not None:
            return True
    base = next((folder / s for s in ("images", "JPEGImages")
                 if (folder / s).is_dir()), folder)
    n = 0
    if base.is_dir():
        for p in sorted(base.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if annotation_boxes_for_image(p) is not None:
                return True
            n += 1
            if n >= probe:
                break
    return False
