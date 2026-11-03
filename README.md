# VisualLatent — Local Image Analysis

VisualLatent analyses image dataset distributions with local feature models. **M24 React Web migration is in progress.** The Streamlit interface is retained only as the behavioral reference until all existing use cases have matching React cases and evidence.

## React Web migration (M24)

The React migration has a verified first slice for **Visualize Embeddings → Image Classifier → whole image** (`LV-WEB-01-IC-01`), including real DINOv2-small extraction, PCA, filters, image selection, cancellation, and visible errors. **Image Classifier fixed UMAP reference** (`LV-WEB-01-IC-03`) has React controls and worker support; deterministic feature and visible UI checks cover first fit, repeat runs, data growth, manual rebuild, and app-cache placement. **Image Classifier 3D KMeans navigation** (`LV-WEB-01-IC-04`) labels 3D blocks, opens a block in 2D for box/lasso selection, and preserves accumulated selection when switching dimensions. **Object Detector whole-image analysis** (`LV-WEB-01-OD-01`) accepts YOLO/VOC image folders, multiple models, selected-file → dataset → manual class-name priority, and reports missing class names explicitly. **Object Detector object-level analysis** (`LV-WEB-01-OD-02`) expands labeled boxes into ordered points, applies bbox padding / input-size / feature-head policy, supports class filtering and 2D/3D views, and previews the selected crop. OD-02 API and visible React evidence use deterministic features; real-model parity and NativeApp integration remain pending for OD-01/02, IC-03/04. **Compare Distributions → whole-image A/B** (`LV-WEB-02-IMG-01`) has a React page and local worker API for the current Streamlit flow: LV feature embeddings, A/B centroid cosine drift, projection controls, point/box selection, and image galleries. That slice also remains partial: deterministic feature functions cover the React/API wiring, while real-model numerical parity and NativeApp folder grants are pending; object-by-class comparison, cart handoff, and the remaining app journeys are not migrated yet. The skipped FID/KID/LPIPS Streamlit scenario is not an acceptance baseline for the current Compare flow.

```powershell
cd web
npm install
npm run build
cd ..
python scripts/react_web_shell.py
```

The API binds to `127.0.0.1:8765`. Set `LV_MODELS_DIR` to a local LV model directory and `LV_CACHE_DIR` to an app-owned writable cache location when needed. A built `web/dist` is served by the local worker for standalone development; NativeApp is expected to serve the same static build through its public integration host.

Migration source of truth: [user need](1_user_needs/react_web_migration.md), [PRD](2_PO_PRD/react_web_migration_prd.md), [architecture](3_Architect_Design/31_react_web_shell.md), and [AC-to-test map](4_PM_Feedback/react_web_shell.md).

> 📖 **怎麼使用？** 新手先看 **[使用指南](docs/usage_guide.md)**（逐工具教學）＋ **[五工具總覽圖](docs/tools_overview.html)**；完整文件索引見 **[文件中心](docs/index.html)**。

## Features

| Tool | Description |
|---|---|
| **Visualize Embeddings** | Extract features from one or more image folders, reduce to 2-D with PCA / t-SNE / UMAP, colour by class label |
| **Compare Distributions** | Compare two groups of image folders and visualise their joint embedding space |

Both tools run **fully offline** — model architectures and weights are loaded from local files only.

---

## Requirements

- Python 3.10+
- Node.js 18+ (React Web build)
- Conda (recommended) or virtualenv

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Model weights

Weights are **not** in the repository (large binaries). The full list of what's
needed — which **feature** uses which model, where it lives, where it's fetched from —
is the single source of truth in **[MODELS.md](MODELS.md)**; `setup_models.py` reads it
to provision, and runtime errors quote it so a missing weight tells you exactly which
feature is blocked.

```bash
python scripts/setup_models.py                # core: DINOv2 + Chinese-CLIP
python scripts/setup_models.py --with-compare # legacy CLI FID/LPIPS weights; not needed for current whole-image Compare
python scripts/model_manifest.py              # check what's present / missing (✅/❌)
```

Everything lives under a single **`models/`** root — one same-named folder per model
(`models/dinov2_vits14/`, `models/lpips/`, …; the full map is in [MODELS.md](MODELS.md)).
Set `LV_MODELS_DIR` to relocate the whole root (so a host platform can point at one
writable model-house and keep its checkout thin).

The script skips anything already present. The legacy metric CLI may provision FID/KID Inception and LPIPS weights on first use, so `--with-compare` is only needed to pre-seed an offline machine for that CLI. The current Streamlit whole-image Compare uses the selected LV feature model instead. For
ResNet, drop `resnet*.pth` into `models/` (or `models/<name>/<name>.pth`).

Supported model name prefixes:

| Prefix | Architecture |
|---|---|
| `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` | torchvision ResNet |
| `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14` | DINOv2 ViT (Meta AI) |

The DINOv2 architecture source is bundled at `scripts/dinov2_hub/` and loaded locally — no internet connection required.

---

## Dataset structure

Each folder passed to the app must follow this layout:

```
dataset/
  train/
    images/       ← .jpg / .jpeg / .png
    labels/       ← YOLO .txt files (optional, used for class labels)
  test/
    images/
    labels/
  valid/
    images/
    labels/
```

**Class names** are auto-detected from `classes.txt` in the parent directory of the first folder:

```
dataset/
  classes.txt     ← one class name per line
  train/
  test/
```

If `classes.txt` is not found, class names can be entered manually in the sidebar.

---

## Legacy Streamlit behavioral baseline

This launcher remains temporarily available for use-case comparison while migration continues. It is not the target interface for new work; parity is tracked by case ID above.

```bash
streamlit run scripts/app.py
```

App opens at `http://localhost:8501`.

---

## Tool 1 — Visualize Embeddings

All image-source selectors support multiple folders: choose or enter folders repeatedly to
append them, and use the per-row ✕ button to remove one source. Output, model, and workspace
directories remain single-value destinations.

1. Enter one or more folder paths (one per line) in the sidebar, e.g.:
   ```
   dataset/train
   dataset/test
   dataset/valid
   ```
2. Select one or more models.
3. Click **▶ Run**.
4. Use the **Model / Method / Split** dropdowns to switch views interactively.
5. Click **⬇ Download HTML (all views)** to save a fully interactive standalone HTML.

---

## Tool 2 — Compare Distributions

Add one or more folders to **Folder A** and **Folder B**; each side is merged into one image group. Select a local LV feature model and choose **whole-image** or **object-by-class (YOLO)** comparison.

- **Whole-image:** labels are not required. The app extracts image embeddings and reports one A/B centroid cosine drift score; larger values mean the group centroids differ more. Choose a projection method and 2D/3D view to inspect the joint embedding plot. In 2D, click or box/lasso-select points; the galleries show selected images or the first 12 from each side.
- **Object-by-class:** select YOLO dataset folders with `images/` and `labels/`. The app compares shared classes and reports classes found on only one side. Object settings are selected automatically and applied to both sides.

Projection distances are for visual exploration; use the drift score for the numeric whole-image comparison. FID/KID/LPIPS/SSIM/PSNR/IS, Coverage Gap and metric exports are not part of the current Streamlit Compare screen. The skipped `test_s05_compare_distributions_metrics` scenario describes an older flow and is not a migration acceptance case.

React migration status: `LV-WEB-02-IMG-01` currently covers only whole-image A/B. Its deterministic UI/API contract does not yet prove real-model parity; object-by-class behavior and NativeApp folder-grant validation remain pending.

---

## Embedding cache

Embeddings are cached per folder per model to avoid re-extraction:

```
dataset/train/embeddings_resnet18/embeddings.npz
dataset/train/embeddings_dinov2_vits14/embeddings.npz
```

Delete a cache folder to force re-extraction.

---

## CLI usage

Both scripts also expose a command-line interface:

```bash
# Visualize embeddings
python scripts/visualize_embeddings.py \
  --folders dataset/train dataset/test \
  --models resnet18 \
  --classes apple banana orange \
  --output-dir output/

# Compare distributions
python scripts/compare_distributions.py \
  --folder-a dataset/train/images \
  --folder-b goal/images \
  --model resnet18 \
  --name my_comparison \
  --n-pairs 500 \
  --output-dir output/
```

---

## Run tests

```bash
pytest tests/ -v
```

---

## Companion script: Region Synthesis

`utils_region_synthesis.py` is a standalone **Tkinter desktop tool** (not part of the
Streamlit app) for compositing a reference image onto a target image at a chosen
position — handy for generating synthetic `goal/` images.

```bash
python utils_region_synthesis.py
```

It sits **upstream** of this app: images it produces can be fed straight into
**Compare Distributions** (e.g. synthesized set as Folder A vs. real set as
Folder B) to quantify how close the synthetic distribution is to the real one.
