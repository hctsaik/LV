# VisualLatent — Dataset Embedding Analysis Tools

A Streamlit web app for visualising and comparing image dataset distributions using deep feature embeddings.

> 📖 **怎麼使用？** 新手先看 **[使用指南](docs/usage_guide.md)**（逐工具教學）＋ **[五工具總覽圖](docs/tools_overview.html)**；完整文件索引見 **[文件中心](docs/index.html)**。

## Features

| Tool | Description |
|---|---|
| **Visualize Embeddings** | Extract features from one or more image folders, reduce to 2-D with PCA / t-SNE / UMAP, colour by class label |
| **Compare Distributions** | Compare two image folders via FID, KID, LPIPS, and SSIM scores, visualise the joint embedding space |

Both tools run **fully offline** — model architectures and weights are loaded from local files only.

---

## Requirements

- Python 3.10+
- Conda (recommended) or virtualenv

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Model weights

Weights are **not** in the repository (large binaries). Provision them once with the
idempotent setup script — the clean "clone → run → it works" flow (no manual file placement):

```bash
python scripts/setup_models.py                # core: DINOv2 + Chinese-CLIP
python scripts/setup_models.py --with-compare # + Compare Distributions (Inception, LPIPS)
```

It downloads into a **model-house** you can relocate via env vars (so a host platform
can keep its checkout/submodule thin and point everything at one writable dir):

| Env var | Overrides | Holds |
|---|---|---|
| `LV_MODELS_DIR` | `models/` | DINOv2 `.pth`, Chinese-CLIP, ResNet `.pth` |
| `LV_INCEPTION_DIR` | `model/` | clean-fid Inception (FID/KID) |

Unset → the package-local `models/` and `model/`. The script skips anything already present.
FID/KID Inception and LPIPS also auto-download on first use, so `--with-compare` is only
needed to pre-seed an offline machine. For ResNet, drop `resnet*.pth` into the models dir.

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

## Launch

```bash
streamlit run scripts/app.py
```

App opens at `http://localhost:8501`.

---

## Tool 1 — Visualize Embeddings

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
