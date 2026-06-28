import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:  # matplotlib is optional — only some legacy tests use it
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# ── anomaly-tool acceptance fixtures (PM-owned; used by test_anomaly_*.py + e2e) ──
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

_AB_D = 384  # DINOv2 vits14 dim (matches Dinov2Extractor / embed_objects)


@pytest.fixture
def ab_vectors():
    """Two orthonormal unit vectors A,B in R^384 (cos(A,B)=0 → distance 1.0).
    A = 'normal patch', B = 'anomalous patch'. Returns (A, B, D)."""
    A = np.zeros(_AB_D, dtype=np.float32); A[0] = 1.0
    B = np.zeros(_AB_D, dtype=np.float32); B[1] = 1.0
    return A, B, _AB_D


@pytest.fixture
def color_patch_extractor(ab_vectors):
    """Factory → a fake DINOv2 patch extractor for patch_features/anomaly_tool
    tests, so no real model/weights are needed.

    The returned ``extractor(crop_PIL) -> (patchtokens (N,384), grid (gh,gw))``
    splits the crop into a gh×gw grid (gh=H//14, gw=W//14, min 1) and emits
    B (e1) for cells whose mean is RED-dominant, else A (e0). Thus a small red
    patch on a green object becomes exactly the few anomalous patches that
    object-level pooling would wash out — the subtle-defect case (A)."""
    A, B, D = ab_vectors

    def make(red_thresh: float = 1.2):
        def extractor(crop):
            crop = crop.convert("RGB")
            w, h = crop.size
            gh, gw = max(1, h // 14), max(1, w // 14)
            arr = np.asarray(crop.resize((gw, gh))).astype(np.float32)  # (gh,gw,3)
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            is_red = r > red_thresh * (g + 1.0)  # red-dominant cell
            toks = np.where(is_red.reshape(-1, 1), B[None, :], A[None, :])
            return toks.astype(np.float32), (gh, gw)
        return extractor

    return make


@pytest.fixture
def color_object_embed(color_patch_extractor):
    """Object-level fake embed_fn(crop)->(384,): mean of the patch fake, L2-norm.
    Defect objects (1 red patch) end up ≈A (subtle at object level)."""
    make_ext = color_patch_extractor()

    def embed_fn(crop):
        toks, _ = make_ext(crop)
        v = toks.mean(axis=0)
        n = np.linalg.norm(v)
        return (v / n if n else v).astype(np.float32)

    return embed_fn


@pytest.fixture(scope="session")
def synthetic_yolo_dataset(tmp_path_factory):
    """A YOLO detection dataset (images/ + labels/ + classes.txt) with a NORMAL
    majority (solid-green object, class 0='good') and a few SUBTLE DEFECTS
    (green object with a small red patch in a known top-left region,
    class 1='bad'). class_id is GROUND TRUTH for AUROC only — the pipeline
    does not read it to decide normal/bad.

    Returns dict: {root, normal_keys, defect_keys, n_normal, n_defect,
    defect_box_norm (the red region as normalized coords inside the image)}.
    """
    root = tmp_path_factory.mktemp("yolo_ds")
    (root / "images").mkdir(); (root / "labels").mkdir()
    (root / "classes.txt").write_text("good\nbad\n", encoding="utf-8")
    n_normal, n_defect = 20, 5
    W = H = 128
    box = (0.5, 0.5, 0.92, 0.92)  # object ≈ whole image
    # 紅色缺陷區塊(x0,y0,x1,y1)放在物件左上。佔約 1/4 面積 → 真實 DINOv2 物件級
    # 也分得開(供 E2E 用預設流程驗證 GUI 接線 + 真實模型分離)。
    # 「極細微瑕疵 + few-shot 乾淨 bank」的硬路徑由單元測試(fake extractor)驗證。
    defect_px = (10, 10, 74, 74)  # 64x64 ≈ 25% 面積
    normal_keys, defect_keys = [], []

    def _write(stem, cls, defect):
        img = Image.new("RGB", (W, H), (0, 170, 0))
        if defect:
            for x in range(defect_px[0], defect_px[2]):
                for y in range(defect_px[1], defect_px[3]):
                    img.putpixel((x, y), (220, 0, 0))
        img.save(root / "images" / f"{stem}.jpg", quality=92)
        cx, cy, w, h = box
        (root / "labels" / f"{stem}.txt").write_text(
            f"{cls} {cx} {cy} {w} {h}\n", encoding="utf-8")

    for i in range(n_normal):
        s = f"normal_{i:02d}"; _write(s, 0, False); normal_keys.append((s, 0))
    for i in range(n_defect):
        s = f"defect_{i:02d}"; _write(s, 1, True); defect_keys.append((s, 0))
    return {
        "root": root, "normal_keys": normal_keys, "defect_keys": defect_keys,
        "n_normal": n_normal, "n_defect": n_defect,
        "defect_box_norm": (defect_px[0] / W, defect_px[1] / H,
                            defect_px[2] / W, defect_px[3] / H),
    }


# ── M6 統一主動學習畫面:label 語義 gate 的 E2E fixtures ──
# 都是「每圖單框」資料(每張圖一個 YOLO box)→ label_semantic_hint 預設偏 'defect'
# (multi_same_class_ratio=0),好讓 head 解鎖只由「語義=defect AND ≥2 類各達 N_min」決定。
# 沿用上面 synthetic_yolo_dataset 的 _write 慣例:128x128 綠底、缺陷加色塊、寫
# images/<stem>.jpg + labels/<stem>.txt + classes.txt。每個 fixture 回 {"root": root}。

def _build_yolo_single_box(root, classes: list[str], specs: list[tuple]):
    """寫一個「每圖單框」YOLO 偵測資料夾。
    classes: classes.txt 的類別名(行序=class_id)。
    specs: [(stem, class_id, patch_rgb_or_None, corner), ...];patch_rgb=None → 純綠底;
           corner ∈ {"tl","bl"} 決定色塊放左上/左下(讓兩瑕疵類視覺可分)。
    每張圖整框為物件(cx,cy,w,h≈whole image),class_id 寫進 label。"""
    (root / "images").mkdir(); (root / "labels").mkdir()
    (root / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    W = H = 128
    box = (0.5, 0.5, 0.92, 0.92)
    for stem, cls, patch_rgb, corner in specs:
        img = Image.new("RGB", (W, H), (0, 170, 0))
        if patch_rgb is not None:
            # 64x64 ≈ 25% 面積色塊;tl=左上(10..74)、bl=左下(10..74 x、54..118 y)
            x0 = 10
            y0 = 10 if corner == "tl" else 54
            for x in range(x0, x0 + 64):
                for y in range(y0, y0 + 64):
                    img.putpixel((x, y), patch_rgb)
        img.save(root / "images" / f"{stem}.jpg", quality=92)
        cx, cy, w, h = box
        (root / "labels" / f"{stem}.txt").write_text(
            f"{cls} {cx} {cy} {w} {h}\n", encoding="utf-8")
    return {"root": root}


@pytest.fixture(scope="session")
def yolo_defect_at_nmin(tmp_path_factory):
    """每圖單框瑕疵資料:scratch(class0,紅塊左上)×8 + stain(class1,藍塊左下)×8。
    預設語義啟發式 → 'defect',兩類各 8 達預設 N_min=8 → 分類頭解鎖。兩類視覺可分
    (紅 vs 藍、左上 vs 左下)讓真實 DINOv2 head 學得起來。"""
    root = tmp_path_factory.mktemp("yolo_defect_at_nmin")
    specs = []
    for i in range(8):
        specs.append((f"scratch_{i:02d}", 0, (220, 0, 0), "tl"))
    for i in range(8):
        specs.append((f"stain_{i:02d}", 1, (0, 0, 220), "bl"))
    return _build_yolo_single_box(root, ["scratch", "stain"], specs)


@pytest.fixture(scope="session")
def yolo_defect_below_nmin(tmp_path_factory):
    """每圖單框瑕疵資料:scratch(class0,紅塊)×7 + stain(class1,藍塊)×9。
    預設語義 'defect',但 scratch=7 < 預設 N_min=8 → 達標類 <2 → head 不解鎖。
    狀態列應列『待補樣本:scratch(7)』。"""
    root = tmp_path_factory.mktemp("yolo_defect_below_nmin")
    specs = []
    for i in range(7):
        specs.append((f"scratch_{i:02d}", 0, (220, 0, 0), "tl"))
    for i in range(9):
        specs.append((f"stain_{i:02d}", 1, (0, 0, 220), "bl"))
    return _build_yolo_single_box(root, ["scratch", "stain"], specs)


@pytest.fixture(scope="session")
def yolo_object_2class(tmp_path_factory):
    """每圖單框物件偵測資料:door(class0)×10 + window(class1)×10,都純綠底無瑕疵
    (視覺相同即可,只測語義 gate 不測準度)。兩類各 10 ≥ N_min=8,所以一旦語義設成
    'object' 仍不訓 head 就證明是『語義守門』非『資料量不足』。"""
    root = tmp_path_factory.mktemp("yolo_object_2class")
    specs = []
    for i in range(10):
        specs.append((f"door_{i:02d}", 0, None, "tl"))
    for i in range(10):
        specs.append((f"window_{i:02d}", 1, None, "tl"))
    return _build_yolo_single_box(root, ["door", "window"], specs)
