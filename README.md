# VisualLatent — Local Image Analysis

VisualLatent analyses image dataset distributions with local feature models. **M24 React Web migration is in progress.** The Streamlit interface is retained only as the behavioral reference until all existing use cases have matching React cases and evidence.

## React Web migration (M24)

The selection-to-curation flow (`LV-WEB-01-CART-01`) now has a React cart persisted in `LV_CACHE_DIR`, with CSV/JSON/ZIP export and visible readback after a different worker PID starts. It accepts Visualize and Compare selections; Compare object selections are collapsed to unique original images, matching the legacy cart. Importing the Streamlit session cart and NativeApp cache-root integration are still unverified.

The independent image-level embedding coverage flow (`LV-WEB-03-IMG-01`) now runs from its own folder/model controls, computes sparsity in the original high-dimensional cosine space, supports projection and color-mode switching, and provides 2D box selection, 3D point selection, thumbnail readback, and explicit clearing. Fixture API and visible browser evidence pass; real-model numeric parity with Streamlit and NativeApp folder-grant/package evidence remain unverified.

Object-level coverage (`LV-WEB-03-OBJ-01`) now runs from the same independent workspace using YOLO object crops. Class, confidence, and source short-edge filters preserve original object indices while keeping labels, embeddings, boxes, and thumbnail crops aligned; the UI reports per-class counts/sparsity, guards tiny classes, and lets developers inspect class blocks. Fixture API and visible Chromium evidence pass. The separate `LV-WEB-03-REF-01` workflow now also has a visible fixture journey for candidate-pool ranking, external-reference recall, reordered class-name alignment, 2D/3D candidate selection, object-crop readback, and curation provenance. Real-model numeric parity with Streamlit and NativeApp folder-grant/package evidence remain unverified.

The image-level attribute completeness grid (`LV-WEB-03-ATTR-01`) now reuses the existing Python binning and diversity-health calculations. React exposes class/split and image-statistic axes, quantile bins, per-cell sample targets, calibrated frequency priors, missing-cell worklists, source-image readback, cell candidate mining, and CSV export. API fixture tests and a visible Chromium journey cover calibration and candidate mining; real-model numeric parity and NativeApp folder-grant/package evidence remain unverified.

The diagnostics flow (`LV-WEB-04-CTX-01` / `HEAD-01` / `EXPL-01`) now rechecks the selected source and run, prepares an exact-resolution classifier head, and runs DINO Attention or target-class Grad-CAM only after an explicit action. The pinned real DINOv2-small API journey and visible fixture controls pass, including signed and positive evidence views, PNG overlay and JSON downloads, and exact map comparisons with the shared explanation helpers. Full Streamlit/MVTec UI parity and NativeApp folder-grant/package evidence remain unverified.

The first false-positive case slice (`LV-WEB-05-CASE-01`) now supports React case creation/reopen, original and screenshot sources stored as SHA-256 named local copies, six-column YOLO prediction import, source and region navigation, and separate region-level verdicts with stale-revision rejection. Fixture API tests and a visible Chromium journey pass, including readback of mixed normal/defect region decisions without creating an image-level label. Retrieval, cited reports, Active Learning handoff, broader Streamlit parity, and NativeApp package/folder-grant evidence remain unverified.

React currently implements **Visualize Embeddings → Image Classifier → whole image** (`LV-WEB-01-IC-01`) and **Object Detector → whole image/object** slices (`LV-WEB-01-OD-01/02`). Fixture-backed visible React flows also cover image-query search, Chinese-CLIP text-query search, dHash and embedding duplicate/leakage review, outlier/disagreement ranking with rank-ordered image navigation and CSV export, and the selection viewer's box/lasso selection, ordered thumbnail gallery, image navigation, and detector annotation overlay. SEARCH-02 is only fixture-verified: real Chinese-CLIP weights are unavailable in this checkout, and the legacy cross-tool Labeling/candidate handoff is not yet implemented in React. Ranking scores reuse the existing Python helpers and are explicitly presented as review clues, not quality or mislabel verdicts; fixed-model numeric comparison with the Streamlit UI remains unverified. These flows do not yet have NativeApp package evidence or full real-model parity evidence. Visible React runs with the pinned real DINOv2-small model pass on Windows ARM64 for the classifier, whole-image detector, and per-object detector fixtures. Object mode uses YOLO boxes, adjustable crop policy, and content-aware cache; the visible fixture journey also carries two selected objects into curation with their original object indices and bounding boxes intact. The classifier's fixed UMAP reference-frame slice has deterministic API and visible UI coverage for fit, reuse, added/removed images, and explicit rebuild; real `umap-learn` integration also passes in an isolated CPython 3.11 AMD64 environment. Windows ARM64 runtime parity and numerical comparison with Streamlit remain unverified. Exact Streamlit numerical comparison, NativeApp folder-grant evidence, NativeApp packaging, and the remaining nested flows are not complete.

`LV-WEB-01-IC-02` now also has a visible fixture journey covering two model pipelines, all five projection selectors, class/split filters, and selection identity through model, projection, and 2D/3D changes. Its UMAP path used real `umap-learn 0.5.12` on fixture embeddings; the second real model weights are absent, so multi-model numerical parity remains unverified.

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
python scripts/setup_models.py --with-compare # + Compare Distributions (Inception, LPIPS)
python scripts/model_manifest.py              # check what's present / missing (✅/❌)
```

Everything lives under a single **`models/`** root — one same-named folder per model
(`models/dinov2_vits14/`, `models/lpips/`, …; the full map is in [MODELS.md](MODELS.md)).
Set `LV_MODELS_DIR` to relocate the whole root (so a host platform can point at one
writable model-house and keep its checkout thin).

The script skips anything already present. FID/KID Inception and LPIPS also auto-download
on first use, so `--with-compare` is only needed to pre-seed an offline machine. For
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

1. Enter **Folder A** and **Folder B** (direct image directories, not split folders), e.g.:
   ```
   Folder A: dataset/train/images
   Folder B: goal/images
   ```
2. Select a model, set **Pairwise metric samples** (used for LPIPS and SSIM), click **▶ Run**.
3. Metrics (FID ↓, KID ↓, LPIPS ↓, SSIM ↑) appear above the plot.
4. Use the **Method** dropdown to switch between PCA / t-SNE / UMAP.
5. Download HTML or JSON metrics with the **⬇** buttons.

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
