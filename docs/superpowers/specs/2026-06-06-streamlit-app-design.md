# Streamlit App Design — 2026-06-06

## Overview

Add `scripts/app.py` as a single Streamlit entry point that covers all functionality
previously available via the two CLI scripts (`visualize_embeddings.py` and
`compare_distributions.py`). Existing scripts are not modified.

## Architecture

```
scripts/
  app.py                  ← new: Streamlit entry point
  visualize_embeddings.py ← unchanged
  compare_distributions.py← unchanged
  _utils.py               ← unchanged
  models.py               ← unchanged
```

**Launch:**
```
streamlit run scripts/app.py
```

**Import strategy:** `app.py` imports figure-builder functions directly from the
existing scripts. No logic is duplicated.

```python
from visualize_embeddings import build_plotly_figure, discover_images
from compare_distributions import build_projection_figure, compute_fid, compute_lpips_score, get_image_paths
from _utils import available_models, load_model, extract_embeddings
```

## Page Layout

```
Sidebar
├── st.radio: [Visualize Embeddings | Compare Distributions]
├── ── tool-specific inputs ──
└── [▶ Run] button

Main area
├── Tool title
├── (after run) st.plotly_chart(fig, use_container_width=True)
├── (after run, Compare only) st.metric() for FID and LPIPS
└── (after run) st.download_button(s) for HTML [+ JSON for Compare]
```

## Tool: Visualize Embeddings

### Sidebar Inputs

| Input | Widget | Default | Notes |
|---|---|---|---|
| Folders | `st.text_area` | — | One path per line |
| Models | `st.multiselect` | all | Populated from `available_models()` |
| Output dir | `st.text_input` | `./output` | Used for HTML download |
| Classes | auto-detected | — | See Classes Detection below |

### Classes Detection

Triggered when the user fills in folder paths (on Run click):

1. Parse the first valid folder path from the text area.
2. Look for `classes.txt` in its **parent directory** (e.g., `dataset/train` → `dataset/classes.txt`).
3. If found: read lines as class names; display with `st.success("Detected N classes: ...")`.
4. If not found: show `st.warning()` and fall back to a `st.text_input` (comma-separated) for manual entry.

### Main Area (after Run)

- `st.plotly_chart(fig, use_container_width=True)`
- `st.download_button` — exports the Plotly figure as HTML (via `fig.to_html()`)

## Tool: Compare Distributions

### Sidebar Inputs

| Input | Widget | Default | Notes |
|---|---|---|---|
| Folder A | `st.text_input` | — | Required |
| Folder B | `st.text_input` | — | Required |
| Model | `st.selectbox` | — | Populated from `available_models()` |
| Name | `st.text_input` | `comparison` | Output filename prefix |
| Output dir | `st.text_input` | `./output` | Used for downloads |
| LPIPS pairs | `st.number_input` | `500` | min=1 |

### Main Area (after Run)

- `st.metric()` side-by-side: FID and LPIPS scores
- `st.plotly_chart(fig, use_container_width=True)`
- `st.download_button` — HTML (projection figure)
- `st.download_button` — JSON (metrics: fid, lpips, n_a, n_b, folders, model)

## Error Handling

- Missing folders: `st.error()` before running any computation.
- No models found (`available_models()` returns `[]`): `st.error()` with explanation.
- Computation errors (e.g., bad paths, model load failure): wrap in `try/except`, show `st.error(str(e))`.
- Long-running computation: wrap with `st.spinner("Computing...")`.

## Dependencies

Add `streamlit` to `requirements.txt`. No other new dependencies.

## Out of Scope

- Authentication or multi-user support.
- Persisting session state across browser refreshes.
- Modifying the existing CLI scripts or their `main()` functions.
