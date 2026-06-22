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
