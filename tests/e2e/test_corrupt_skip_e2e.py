"""壞檔防呆驗收:任一影像或其 YOLO label 損壞/格式錯時,各功能『跳過』壞檔而
不崩潰,好檔照常運作,並回報略過數。實作統一走 scripts/safe_io.py。

**只讀不寫**:壞檔被略過,絕不刪除/改寫使用者資料(見 safe_io 模組說明)。

兩層驗收:
- **headless 廣度**(不標 e2e,進快速 gate):直接驅動各工具真實的讀取/管線函式
  —— 覆蓋『各個功能』的讀取層(safe_io 基元、物件發現、瑕疵管線、近重複 phash、
  匯出、Compare 影像比對)。確定性、快、真實行為斷言。
- **真實 GUI E2E**(標 @e2e):對 scripts/app.py 起真實 Streamlit + Playwright,瑕疵
  偵測工具吃含壞檔的資料夾 → 不得出現 stException、好檔仍排序、顯示「已略過」提示。

灰階讀取路徑(completeness.image_stats / signal_strength.signal_level_for_image)與
label-parse(visualize_embeddings.parse_label_file)同走 safe_io,其壞檔行為由
test_safe_io_primitives 的共用基元 + 各模組 py_compile 覆蓋。
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from .conftest import _add_folder, load_app, wait_idle

W = H = 128
BOX = (0.5, 0.5, 0.92, 0.92)          # 物件 ≈ 整張圖(與 tests/conftest 同款)
DEFECT_PX = (10, 10, 74, 74)          # 紅色缺陷塊 ≈ 25% 面積(真實 DINOv2 分得開)


@pytest.fixture
def corrupt_yolo_dataset(tmp_path):
    """好的 YOLO 資料集(8 normal + 3 defect)+ 注入壞檔:

      - corrupt_bytes.jpg:垃圾 bytes 偽裝 .jpg(UnidentifiedImageError)
      - truncated.jpg:真 JPEG 砍半(.load() 時 OSError)
      - badlabel.jpg:**好影像**但 label 是二進位垃圾(壞編碼)

    回傳每類壞檔的 stem,供斷言「壞檔被略過、好檔保留」。
    """
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("good\nbad\n", encoding="utf-8")
    good_stems: list[str] = []

    def write_good(stem: str, cls: int, defect: bool = False) -> None:
        img = Image.new("RGB", (W, H), (0, 170, 0))
        if defect:
            ImageDraw.Draw(img).rectangle(list(DEFECT_PX), fill=(200, 0, 0))
        img.save(root / "images" / f"{stem}.jpg", quality=90)
        (root / "labels" / f"{stem}.txt").write_text(
            f"{cls} {BOX[0]} {BOX[1]} {BOX[2]} {BOX[3]}\n", encoding="utf-8")
        good_stems.append(stem)

    for i in range(8):
        write_good(f"normal_{i:02d}", 0)
    for i in range(3):
        write_good(f"defect_{i:02d}", 1, defect=True)

    # 壞檔 1:垃圾 bytes 偽裝成 JPEG
    (root / "images" / "corrupt_bytes.jpg").write_bytes(b"\x00\x01not-a-jpeg\xff\xd8junk")
    (root / "labels" / "corrupt_bytes.txt").write_text(
        f"0 {BOX[0]} {BOX[1]} {BOX[2]} {BOX[3]}\n", encoding="utf-8")

    # 壞檔 2:真 JPEG 砍半 → 截斷(header 看似有效,解碼到一半才爆)
    buf = io.BytesIO()
    Image.new("RGB", (W, H), (0, 170, 0)).save(buf, format="JPEG", quality=90)
    raw = buf.getvalue()
    (root / "images" / "truncated.jpg").write_bytes(raw[: len(raw) // 2])
    (root / "labels" / "truncated.txt").write_text(
        f"0 {BOX[0]} {BOX[1]} {BOX[2]} {BOX[3]}\n", encoding="utf-8")

    # 壞檔 3:好影像 + 壞編碼/二進位垃圾 label(預設 read_text 會 UnicodeDecodeError)
    Image.new("RGB", (W, H), (0, 170, 0)).save(root / "images" / "badlabel.jpg", quality=90)
    (root / "labels" / "badlabel.txt").write_bytes(b"\xff\xfe\x00\x80\x81 \x90garbage label\n")

    return {
        "root": root,
        "good_stems": good_stems,                       # 11 個帶有效 label 的好檔
        "corrupt_imgs": {"corrupt_bytes.jpg", "truncated.jpg"},
        "bad_label_stem": "badlabel",
        "n_readable_imgs": len(good_stems) + 1,         # +badlabel.jpg(影像本身可讀)
    }


# ── headless 廣度:各功能的真實讀取/管線函式吃壞檔不崩潰 ──────────────────

def test_safe_io_primitives(corrupt_yolo_dataset):
    """統一入口 safe_io:壞影像→None、壞 label→不拋例外、partition 正確分流。"""
    from safe_io import (is_readable_image, partition_readable,
                         safe_open_image, safe_read_text)
    root = corrupt_yolo_dataset["root"]
    assert safe_open_image(root / "images" / "corrupt_bytes.jpg") is None
    assert safe_open_image(root / "images" / "truncated.jpg") is None
    assert safe_open_image(root / "images" / "normal_00.jpg") is not None
    assert is_readable_image(root / "images" / "defect_00.jpg")
    assert not is_readable_image(root / "images" / "truncated.jpg")

    # 壞編碼 label → 回字串、不拋例外(逐行解析會自然略過壞行)
    txt = safe_read_text(root / "labels" / "badlabel.txt")
    assert isinstance(txt, str)

    all_imgs = sorted((root / "images").glob("*.jpg"))
    good, bad = partition_readable(all_imgs)
    assert {p.name for p in bad} == corrupt_yolo_dataset["corrupt_imgs"]
    assert len(good) == corrupt_yolo_dataset["n_readable_imgs"]


def test_discover_yolo_survives_bad_label(corrupt_yolo_dataset):
    """物件/散點等工具的共用發現層:壞 label 不崩潰,好檔物件照常,壞 label 0 框被略過。"""
    from interaction import discover_yolo_objects
    from object_eval import classes_for, list_images
    root = corrupt_yolo_dataset["root"]
    ip, cn = list_images(root), classes_for(root)
    meta = discover_yolo_objects(ip, cn)        # 不得拋例外
    stems = {Path(str(m["image_path"])).stem for m in meta}
    assert {"normal_00", "defect_00"} <= stems
    assert "badlabel" not in stems              # 壞 label → 0 框 → 不在 meta


def test_anomaly_pipeline_skips_corrupt(corrupt_yolo_dataset,
                                        color_patch_extractor, color_object_embed):
    """瑕疵偵測 run_pipeline:壞影像前置過濾掉、回報 skipped、好檔仍產出排序,不崩潰。"""
    from anomaly_tool import run_pipeline
    from object_eval import classes_for, list_images
    root = corrupt_yolo_dataset["root"]
    ip, cn = list_images(root), classes_for(root)
    r = run_pipeline(ip, cn, mode="one_stage", score_mode="object",
                     extractor=color_patch_extractor(), embed_fn=color_object_embed)
    assert r["records"], "好檔仍應產出排序結果"
    assert {Path(s).name for s in r["skipped"]} == corrupt_yolo_dataset["corrupt_imgs"]
    rec_stems = {Path(str(rec["image_path"])).stem for rec in r["records"]}
    assert "corrupt_bytes" not in rec_stems and "truncated" not in rec_stems
    assert "normal_00" in rec_stems             # 好檔物件保留


def test_manifest_phash_skips_corrupt(corrupt_yolo_dataset):
    """近重複偵測 compute_phash:壞影像→None(略過),好影像→雜湊。"""
    from manifest import compute_phash
    root = corrupt_yolo_dataset["root"]
    assert compute_phash(root / "images" / "corrupt_bytes.jpg") is None
    assert compute_phash(root / "images" / "normal_00.jpg") is not None


def test_compare_ssim_skips_corrupt(corrupt_yolo_dataset):
    """Compare Distributions 影像比對:含壞檔的配對被跳過,仍算出有限分數,不崩潰。"""
    from compare_distributions import compute_ssim_score
    root = corrupt_yolo_dataset["root"]
    good = [root / "images" / f"normal_{i:02d}.jpg" for i in range(4)]
    withbad = [root / "images" / "corrupt_bytes.jpg",
               root / "images" / "truncated.jpg",
               root / "images" / "normal_05.jpg",
               root / "images" / "normal_06.jpg"]
    score = compute_ssim_score(good, withbad, n_pairs=4)   # 不得拋例外
    assert not math.isnan(score)                # 至少有可比對的好對 → 有限分數


def test_export_subset_skips_corrupt(corrupt_yolo_dataset, tmp_path):
    """匯出子集 crop-out:壞影像進 corrupt 帳本被略過、好檔照常匯出,不崩潰。"""
    from export_subset import ExportItem, export_subset
    root = corrupt_yolo_dataset["root"]
    items = [
        ExportItem(image_path=root / "images" / f"{stem}.jpg",
                   label_path=root / "labels" / f"{stem}.txt",
                   class_names=["good", "bad"], level="object", object_ids=[0])
        for stem in ("normal_00", "defect_00", "corrupt_bytes", "truncated")
    ]
    report = export_subset(items, tmp_path / "export", layout="crop-out",
                           source_roots=[root])            # 不得拋例外
    corrupt_srcs = {Path(c["src"]).stem for c in report.corrupt}
    assert {"corrupt_bytes", "truncated"} <= corrupt_srcs
    assert report.exported >= 1                 # 好檔仍匯出


# ── 真實 GUI E2E:瑕疵偵測工具吃含壞檔資料夾(使用者真的能用)──────────────

@pytest.mark.e2e
def test_anomaly_gui_skips_corrupt(app_server, browser, corrupt_yolo_dataset):
    """真實 Streamlit + Playwright:含壞檔資料夾 → 不崩潰、好檔仍排序、顯示「已略過」。"""
    from playwright.sync_api import expect
    ds = corrupt_yolo_dataset
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    load_app(page, app_server)
    page.locator('.st-key-tool_switch').get_by_text("瑕疵偵測", exact=True).click()
    wait_idle(page)
    _add_folder(page, "anomaly_folder", str(ds["root"]))
    wait_idle(page)
    page.locator('.st-key-anomaly_run button').click()

    container = page.locator('[data-testid="stAppViewContainer"]')
    # 1) 好檔仍算出排序:8 normal + 3 defect = 11 個好物件全部進管線(corrupt_bytes / truncated
    #    兩張壞影像被略過、badlabel 的壞 label 0 框)。原可見排序清單已移除 → 改驗「共 N 個物件」
    #    摘要:N==11 ⟹ 缺陷檔也被正常處理、沒被當壞檔丟掉。長 timeout 等冷啟動模型載入。
    page.wait_for_function(
        r"() => { const m = document.body.innerText.match(/共\s*(\d+)\s*個物件/);"
        r" return m && +m[1] === 11; }", timeout=180000)
    # 2) 壞檔沒讓工具崩潰:無原始 traceback
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    # 3) 顯示略過提示(2 個壞影像被跳過並告知使用者)
    expect(container).to_contain_text("已略過", timeout=30000)
    ctx.close()
