# VisualLatent — Local Image Analysis

VisualLatent analyses image dataset distributions with local feature models. **M24 React Web migration is in progress.** The Streamlit interface is retained only as the behavioral reference until all existing use cases have matching React cases and evidence.

## React Web migration (M24)

The first NativeApp folder-grant slice for `LV-WEB-01-IC-01` is implemented on the app side: React forwards the opaque picker grant, and the Python API resolves it through the public loopback grant endpoint and binds it to the selected folder. On 2026-11-02, a built and verified NativeApp `0.2.1-dev` package completed a managed React/DINOv2 journey with a NativeApp test grant: three synthetic images produced finite, nonzero 3×384 embeddings, three points rendered, and the selected 64×64 image read back; fixture hashes stayed unchanged and the App stopped cleanly. Playwright ran headless Chromium with a test shim for the grant request. The NativeApp desktop window and OS picker, React save/reopen, native ARM64 Python, and release/update recovery remain unverified. See the [managed package receipt](https://github.com/hctsaik/nativeApp_v2/blob/codex/lv-react-http-runtime-fix/evidence/app-integration/lv-react-managed-package-20261102/result.json).

`LV-WEB-02-IMG-01` whole-image A/B comparison now has a visible real-model React journey with pinned DINOv2-small. On the same eight images, React matches the legacy Streamlit centroid-cosine drift and PCA coordinates within `1e-5`, displays the drift and eight points, reads the original selected image bytes, and leaves source hashes unchanged. This ran with Windows CPython 3.11.9 AMD64; NativeApp managed-host and native ARM64 Python evidence remain unverified.

`LV-WEB-10-EVAL-01` now has a React evaluation workspace for YOLO ground truth, prediction CSV, optional consensus CSV, IoU/confidence thresholds, class-aware matching, per-class metrics, false-negative crops, confusion summaries, and `escapes.csv` export. Its explicit sample journey demonstrates one missed image and one gray-band image without implying model predictions. A real result can be explicitly attached as one path-free event to an existing active-learning round; the Curation timeline reads back the report and uses recall, precision, FN, and FP as round metrics. Sample results cannot be attached. It reuses the existing Python evaluation core and prediction parser. The old Streamlit `_evaluation_ui` path had no official toolbar entry; React now exposes the behavior. API and visible Chromium fixture journeys pass, and source hashes stay unchanged. A real NativeApp picker/package journey and a side-by-side legacy UI comparison remain unverified.

The selection-to-curation flow (`LV-WEB-01-CART-01`) now has a React cart persisted in `LV_CACHE_DIR`, with CSV/JSON/ZIP export and visible readback after a different worker PID starts. It accepts Visualize and Compare selections; Compare object selections are collapsed to unique original images, matching the legacy cart. Importing the Streamlit session cart and NativeApp cache-root integration are still unverified.

The independent image-level embedding coverage flow (`LV-WEB-03-IMG-01`) now runs from its own folder/model controls, computes sparsity in the original high-dimensional cosine space, supports projection and color-mode switching, and provides 2D box selection, 3D point selection, thumbnail readback, and explicit clearing. Fixture API and visible browser evidence pass; real-model numeric parity with Streamlit and NativeApp folder-grant/package evidence remain unverified.

Object-level coverage (`LV-WEB-03-OBJ-01`) now runs from the same independent workspace using YOLO object crops. Class, confidence, and source short-edge filters preserve original object indices while keeping labels, embeddings, boxes, and thumbnail crops aligned; the UI reports per-class counts/sparsity, guards tiny classes, and lets developers inspect class blocks. Fixture API and visible Chromium evidence pass. The separate `LV-WEB-03-REF-01` workflow now also has a visible fixture journey for candidate-pool ranking, external-reference recall, reordered class-name alignment, 2D/3D candidate selection, object-crop readback, and curation provenance. Real-model numeric parity with Streamlit and NativeApp folder-grant/package evidence remain unverified.

The image-level attribute completeness grid (`LV-WEB-03-ATTR-01`) now reuses the existing Python binning and diversity-health calculations. React exposes class/split and image-statistic axes, quantile bins, per-cell sample targets, calibrated frequency priors, missing-cell worklists, source-image readback, cell candidate mining, and CSV export. API fixture tests and a visible Chromium journey cover calibration and candidate mining; real-model numeric parity and NativeApp folder-grant/package evidence remain unverified.

The diagnostics flow (`LV-WEB-04-CTX-01` / `HEAD-01` / `EXPL-01`) now rechecks the selected source and run, prepares an exact-resolution classifier head, and runs DINO Attention or target-class Grad-CAM only after an explicit action. On 2026-11-02, the visible React controls and pinned real DINOv2-small Attention/Grad-CAM API cases passed on native Windows ARM64 CPython 3.14.3 with CPU PyTorch 2.12.0; the test also checks exact maps against the shared explanation helpers. Full Streamlit/MVTec UI parity and NativeApp-managed runtime, folder-grant, and package evidence remain unverified.

The false-positive case workspace (`LV-WEB-05-CASE-01` / `RETRIEVE-01` / `REPORT-01` / `AL-01`) now supports React case creation/reopen, original and screenshot sources stored as SHA-256 named local copies, six-column YOLO prediction import, source and region navigation, and separate region-level verdicts with stale-revision rejection. Explicit DINOv2-small retrieval compares the selected ROI with user-supplied normal/defect folders and manually reviewed history, records model and source fingerprints, retains displayed images as case-owned copies, and adds references only as pending human-review candidates. Engineers can add and edit evidence-cited facts, hypotheses, and actions, track user-reply status separately from cause status, and export Markdown and path-safe JSON snapshots. Review-only batches now deduplicate image assets by SHA-256, link idempotently to the round ledger, display candidate ROI overlays, support individual and partial JSON readback with case/revision/source/ROI checks, and never write training labels or model metrics. Fixture API tests and visible Chromium journeys cover case intake, explicit search, empty categories, skipped files, claim citations and edit history, exports, cache reuse, image readback, candidate handoff, review ZIPs, partial readback, and manual candidate review. Broader Streamlit parity and NativeApp package/folder-grant evidence remain unverified.

The first `LV-WEB-05-EXT-01` React slice now supports content-addressed ONNX model uploads, immutable profile revisions, explicit classification/YOLO output contracts, local CPU inference, full historical output comparison, and JSON/ZIP evidence export with input/output tensor hashes and previews. API and visible Chromium fixture journeys pass for both classification and detection. Install the optional local inference/test dependencies with `pip install -r requirements-external-models.txt`. Controlled experiments, paired holdout evaluation, reviewed training ZIP export, real deployment-model parity, actual Windows ARM64 worker/package validation, and NativeApp folder-grant evidence remain incomplete; inference does not create human-reviewed labels.

`LV-WEB-06-BANK-01` now exposes the existing anomaly pipeline through React: choose YOLO or whole-image input, patch or object scoring, model resolution and label semantics, build a Normal Bank, then save and reload the bank, fixed projector, few-shot records and optional classifier head from `LV_CACHE_DIR`. The visible React build ran the pinned DINOv2-small weights on four YOLO objects and produced 384D patch/object features; the object label semantic kept the classifier head locked, while defect semantics plus the configured two-sample threshold trained it. A separate visible persistence journey stopped the first worker and loaded the saved fixture bank in a new worker PID.

`LV-WEB-06-SCORE-01` now has a partial React slice for applying the active bank to target folders, risk ranking and filtering, object-crop preview, Patch heatmap, and confirmation-based threshold reclassification. The real DINOv2 visible journey scored four objects, rendered a heatmap, reapplied a confirmation, and kept the model-bank SHA-256 and source dataset unchanged; fixture API coverage checks filtered rows, path-safe responses, and stale revisions. Scoring results stay in the current worker. `LV-WEB-06-REVIEW-01` adds visible anomaly-candidate selection into the persistent Curation list and verifies ZIP bytes match the untouched original image; candidates remain human-review items. Full Streamlit numerical/UI parity, NativeApp folder grant, and cross-worker score-run persistence remain unverified.

`LV-WEB-10-PRELABEL-01` now reuses the existing frozen classification head and M11 gating logic to preview predictions, confidence, acceptance, and object crops, then export accepted YOLO labels plus `classes.txt` through the existing C6 source-protection helper. API and visible fixture journeys read back the files, confirm source immutability, and verify the no-head message. Same-input Streamlit/React numerical parity and NativeApp folder-grant evidence remain unverified.

`LV-WEB-07-SAMPLE-01` now builds a YOLO-object or whole-image search sample bank directly from the local feature model, without first creating an anomaly model, and saves it under `LV_CACHE_DIR` behind an opaque ID. API and visible Chromium journeys cover both source semantics, readback from a fresh worker, and unchanged source files. The separate `LV-WEB-07-SCAN-01` React slice scans target folders, previews candidates, applies a similarity threshold, supports acceptance and class correction, appends selected embeddings to the bank, and reuses that bank on a later scan. Alongside fixture API/Chromium coverage, a visible Chromium journey using the pinned DINOv2-small checkpoint now builds and reads a real bank, scans exact image copies, reviews two high-similarity candidates, appends them, reads the four-sample bank back, and scans again; it ran on Windows CPython 3.11.9 AMD64. `LV-WEB-07-EXPORT-01` exports accepted YOLO labels, classes, and a retrieval CSV to a separate folder, with optional image copies and a C6 source-write guard, and adds the local incremental monitor workspace, profile download, and human queue. Fixture API and visible Chromium journeys cover checkbox-driven export, monitor scan, preview, accept/skip, queue updates, and unchanged sources without creating an anomaly model. Same-input Streamlit numeric comparison and NativeApp Windows ARM64 worker/package/folder-grant evidence remain incomplete.

React currently implements **Visualize Embeddings → Image Classifier → whole image** (`LV-WEB-01-IC-01`) and **Object Detector → whole image/object** slices (`LV-WEB-01-OD-01/02`). Fixture-backed visible React flows also cover image-query search, Chinese-CLIP text-query search, dHash and embedding duplicate/leakage review, outlier/disagreement ranking with rank-ordered image navigation and CSV export, and the selection viewer's box/lasso selection, ordered thumbnail gallery, image navigation, and detector annotation overlay. SEARCH-02's text and duplicate analysis remains fixture-verified because real Chinese-CLIP weights are unavailable in this checkout. The Curation → Labeling → ANnoTation handoff now passes a full cross-repository React browser journey, including service restarts and LV readback, locally and in [Windows GitHub Actions run 41431459367](https://github.com/hctsaik/LV/actions/runs/41431459367); duplicate-pair keep/drop readback also passes locally on Windows ARM64 and in [hosted run 41434997697](https://github.com/hctsaik/LV/actions/runs/41434997697), with two SHA-matched decisions returned through ANnoTation React and LV. Ranking scores reuse the existing Python helpers and are explicitly presented as review clues, not quality or mislabel verdicts; fixed-model numeric comparison with the Streamlit UI remains unverified. These flows do not yet have NativeApp package evidence or full real-model parity evidence. Visible React runs with the pinned real DINOv2-small model pass on Windows ARM64 for the classifier, whole-image detector, and per-object detector fixtures. Object mode uses YOLO boxes, adjustable crop policy, and content-aware cache; the visible fixture journey also carries two selected objects into curation with their original object indices and bounding boxes intact. The classifier's fixed UMAP reference-frame slice has deterministic API and visible UI coverage for fit, reuse, added/removed images, and explicit rebuild; real `umap-learn` integration also passes in an isolated CPython 3.11 AMD64 environment. Windows ARM64 runtime parity and numerical comparison with Streamlit remain unverified. Exact Streamlit numerical comparison, NativeApp folder-grant evidence, NativeApp packaging, and the remaining nested flows are not complete.

`LV-WEB-01-IC-02` also has an opt-in real-model API comparison: two local ignored DINOv2 weights run all five projections, and all 10 model/projection coordinate sets match the existing Streamlit calculations within `rtol=2e-4, atol=2e-4`. It ran on CPython 3.11.9 AMD64 with CPU PyTorch; the weights are not committed, and this does not establish Windows ARM64 or NativeApp package parity. Re-run it with `LV_TEST_MODELS_DIR` pointing to a local `models/` directory:

```powershell
$env:LV_TEST_MODELS_DIR = "C:\path\to\models"
python -m pytest -q tests/test_react_web_shell.py::test_case_lv_web_01_ic_02_real_models_match_streamlit_projection_values
```

The separate visible React journey covers both model selectors, all five projections, class/split filters, and selection identity through model, projection, and 2D/3D changes. Its fixture-backed UMAP uses real `umap-learn 0.5.12`.

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

This installs the Python runtime used by React Web. The legacy Streamlit UI is
kept as a behavior reference while parity work is in progress; install
`requirements-streamlit-reference.txt` only when running Streamlit reference
tests.

External ONNX investigation is optional. Install its CPU runtime and executable test-fixture dependency only when using that workflow:

```powershell
pip install -r requirements-external-models.txt
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

## Legacy Streamlit behavior reference

The legacy Streamlit implementation is retained as a behavior reference for
case-by-case React parity checks. It is not a supported end-user launch path.
Use the React Web launcher above. Current coverage and remaining parity work
are tracked in [the migration acceptance matrix](4_PM_Feedback/react_web_shell.md).

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
