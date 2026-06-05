# Dataset Visualization — Design Spec
**Date:** 2026-06-05

## Overview

Two standalone CLI scripts for dataset analysis, sharing a common utility module. Neither touches the existing `retrieval/` module's internals.

---

## File Structure

```
scripts/
  _utils.py                  # Shared: model loading, embedding extraction, figure saving
  visualize_embeddings.py    # Feature 1: PCA/t-SNE embedding visualization
  compare_distributions.py   # Feature 2: FID/LPIPS distribution comparison
output/                      # All outputs (created at runtime)
docs/superpowers/specs/      # This file
```

---

## New Dependencies (add to requirements.txt, install in visiondiy conda env)

| Package | Purpose |
|---|---|
| `scikit-learn` | PCA, t-SNE |
| `plotly` | Interactive HTML figures |
| `clean-fid` | FID computation (Inception v3) |
| `lpips` | LPIPS perceptual distance |

---

## `scripts/_utils.py` — Shared Utilities

Three public functions:

1. **`load_model(model_name: str)`**
   Reuses existing `retrieval/` module classes (DINOv2, Clip) to load the selected model.
   Supported: `dinov2`, `siglip2_base`, `clip`.

2. **`extract_embeddings(image_paths: list[Path], model) -> np.ndarray`**
   Batch-extracts embeddings, returns shape `(N, D)`.

3. **`save_figure(plotly_fig, mpl_fig, output_dir: Path, name: str)`**
   Saves `name.html` (Plotly) and `name.png` (Matplotlib) to `output_dir`.

---

## Feature 1 — `scripts/visualize_embeddings.py`

### CLI

```bash
python scripts/visualize_embeddings.py \
  --dataset-dir ./dataset \
  --model siglip2_base \          # siglip2_base | dinov2 | clip
  --classes apple banana orange \ # class ID → name mapping (0→apple, 1→banana, ...)
  --output-dir ./output
```

### Process

1. Scan all available splits (`train`, `test`, `valid`) under `<dataset-dir>/<split>/images/`
2. For each image, read the corresponding `<split>/labels/<name>.txt`
   - Single unique class ID → map to `--classes` name
   - Multiple unique class IDs → label as `mix`
   - Missing label file → label as `unknown`
3. Extract embeddings with the chosen model
4. Compute **both** PCA and t-SNE (2D) ahead of time
5. Build Plotly figure with:
   - One trace per `(class × split)` combination
   - `updatemenus` buttons to switch PCA ↔ t-SNE
   - Dropdown/buttons to filter by split (All / Train / Test / Valid)
   - Legend click to toggle individual class visibility
   - Hover: filename, class, split
6. Build Matplotlib static figure (all data, PCA + t-SNE side by side)

### Outputs

```
output/
  embeddings_visualization.html   # Interactive: PCA/t-SNE toggle, split filter, class toggle
  embeddings_pca.png              # Static PCA snapshot
  embeddings_tsne.png             # Static t-SNE snapshot
```

---

## Feature 2 — `scripts/compare_distributions.py`

### CLI

```bash
python scripts/compare_distributions.py \
  --folder-a ./dataset/train/images \
  --folder-b ./dataset/test/images \
  --model siglip2_base \
  --name train_vs_test \
  --output-dir ./output
```

### Process

1. Load all images from `--folder-a` and `--folder-b`
2. Extract embeddings for both groups using the chosen model
3. Concatenate, apply PCA to 2D; color by group (A vs B)
4. Build Plotly interactive scatter plot (hover shows filename + group)
5. Compute **FID** using `clean-fid` on the two folders
6. Compute **LPIPS**: randomly sample min(500, N_a × N_b) cross-group pairs, average perceptual distance
7. Display FID and LPIPS values in the HTML figure's title/annotation area

### Outputs

```
output/
  <name>_projection.html    # Interactive scatter plot with FID/LPIPS shown
  <name>_projection.png     # Static scatter plot
  <name>_metrics.json       # { "fid": 12.3, "lpips": 0.45, "n_a": 200, "n_b": 50 }
```

---

## Dataset Label Format

YOLO `.txt` format: each line is `<class_id> <cx> <cy> <w> <h>`.
Class IDs map to names via `--classes` argument (positional: 0=first name, 1=second, ...).
No `data.yaml` exists in the current dataset; the `--classes` argument is required.

Current dataset default: `--classes apple banana orange`
