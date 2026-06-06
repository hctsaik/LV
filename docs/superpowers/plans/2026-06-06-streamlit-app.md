# Streamlit App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scripts/app.py` as a single Streamlit entry point covering all functionality of the two existing CLI scripts.

**Architecture:** Single `app.py` imports figure-builder and utility functions from existing scripts (unchanged). Two pure helper functions (`read_classes_txt`, `parse_folder_paths`) are extracted and unit-tested. The embedding/PCA/t-SNE orchestration lives in `app.py` using imported utilities from `_utils`. Streamlit sidebar provides tool selection and per-tool inputs; results render inline with `st.plotly_chart` and download buttons.

**Tech Stack:** Streamlit ≥ 1.32, Plotly, scikit-learn (PCA, t-SNE), existing `_utils`, `visualize_embeddings`, `compare_distributions` modules.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `requirements.txt` | Add `streamlit>=1.32.0` |
| Create | `scripts/app.py` | Streamlit entry point + helper functions |
| Create | `tests/test_app.py` | Unit tests for `read_classes_txt` and `parse_folder_paths` |

Existing scripts (`visualize_embeddings.py`, `compare_distributions.py`, `_utils.py`) are **not modified**.

---

### Task 1: Add streamlit to requirements

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add streamlit to requirements.txt**

Append to `requirements.txt`:
```
streamlit>=1.32.0
```

- [ ] **Step 2: Install streamlit**

Run: `conda run -n visiondiy pip install "streamlit>=1.32.0"`

Expected: `Successfully installed streamlit-...`

- [ ] **Step 3: Verify import**

Run: `conda run -n visiondiy python -c "import streamlit; print(streamlit.__version__)"`

Expected: prints a version string >= 1.32.0

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add streamlit dependency"
```

---

### Task 2: Helper functions with tests

**Files:**
- Create: `tests/test_app.py`
- Create: `scripts/app.py` (skeleton with helper functions only)

- [ ] **Step 1: Write failing tests**

Create `tests/test_app.py`:

```python
from pathlib import Path
import pytest
from app import parse_folder_paths, read_classes_txt


def test_read_classes_txt_found(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    (tmp_path / "classes.txt").write_text("apple\nbanana\norange\n")
    assert read_classes_txt(folder) == ["apple", "banana", "orange"]


def test_read_classes_txt_not_found(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    assert read_classes_txt(folder) is None


def test_read_classes_txt_empty_file(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    (tmp_path / "classes.txt").write_text("\n\n")
    assert read_classes_txt(folder) is None


def test_parse_folder_paths_basic(tmp_path):
    text = f"{tmp_path}/train\n{tmp_path}/test"
    result = parse_folder_paths(text)
    assert result == [Path(f"{tmp_path}/train"), Path(f"{tmp_path}/test")]


def test_parse_folder_paths_ignores_blank_lines(tmp_path):
    text = f"{tmp_path}/train\n\n  \n{tmp_path}/test"
    result = parse_folder_paths(text)
    assert len(result) == 2


def test_parse_folder_paths_empty():
    assert parse_folder_paths("") == []
    assert parse_folder_paths("   \n  ") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n visiondiy python -m pytest tests/test_app.py -v`

Expected: `ImportError` — `app.py` doesn't exist yet.

- [ ] **Step 3: Create scripts/app.py with helper functions**

Create `scripts/app.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent))

from _utils import available_models, extract_embeddings, load_model
from compare_distributions import (
    build_projection_figure,
    compute_fid,
    compute_lpips_score,
    get_image_paths,
)
from visualize_embeddings import build_plotly_figure, discover_images


def read_classes_txt(folder: Path) -> list[str] | None:
    """Return class names from <folder-parent>/classes.txt, or None if absent/empty."""
    classes_file = folder.parent / "classes.txt"
    if not classes_file.exists():
        return None
    lines = [ln.strip() for ln in classes_file.read_text().splitlines() if ln.strip()]
    return lines if lines else None


def parse_folder_paths(text: str) -> list[Path]:
    """Return a Path for each non-blank line in text."""
    paths = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            paths.append(Path(line))
    return paths


def _visualize_embeddings_ui() -> None:
    pass  # Task 3


def _compare_distributions_ui() -> None:
    pass  # Task 4


def main() -> None:
    st.set_page_config(page_title="Dataset Analysis", layout="wide")
    st.title("Dataset Analysis Tools")

    tool = st.sidebar.radio(
        "Tool",
        ["Visualize Embeddings", "Compare Distributions"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    if tool == "Visualize Embeddings":
        _visualize_embeddings_ui()
    else:
        _compare_distributions_ui()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n visiondiy python -m pytest tests/test_app.py -v`

Expected: 6 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/app.py tests/test_app.py
git commit -m "feat: add app.py skeleton with helper functions and tests"
```

---

### Task 3: Visualize Embeddings UI

**Files:**
- Modify: `scripts/app.py` — replace `_visualize_embeddings_ui` stub

- [ ] **Step 1: Replace the stub with the full implementation**

In `scripts/app.py`, replace:

```python
def _visualize_embeddings_ui() -> None:
    pass  # Task 3
```

with:

```python
def _visualize_embeddings_ui() -> None:
    st.header("Visualize Embeddings")

    with st.sidebar:
        folders_text = st.text_area(
            "Folders (one per line)",
            placeholder="dataset/train\ndataset/test\ndataset/valid",
        )
        all_models = available_models()
        if not all_models:
            st.error("No .pth models found in ./models/. Add a model file and restart.")
            return
        selected_models = st.multiselect("Models", all_models, default=all_models)
        class_input = st.text_input(
            "Class names — fallback if classes.txt not found",
            value="apple,banana,orange",
        )
        run = st.button("▶ Run", use_container_width=True, key="run_viz")

    if not run:
        st.info("Configure inputs in the sidebar and click ▶ Run.")
        return

    folders = parse_folder_paths(folders_text)
    if not folders:
        st.error("Enter at least one folder path.")
        return

    missing = [str(f) for f in folders if not (f / "images").exists()]
    if missing:
        st.error(f"Folder(s) missing 'images/' subdirectory: {', '.join(missing)}")
        return

    if not selected_models:
        st.error("Select at least one model.")
        return

    detected = read_classes_txt(folders[0])
    if detected is not None:
        class_names = detected
        st.success(
            f"Auto-detected {len(class_names)} classes from classes.txt: "
            + ", ".join(class_names)
        )
    else:
        class_names = [c.strip() for c in class_input.split(",") if c.strip()]
        if not class_names:
            st.error("Enter at least one class name.")
            return
        st.info(f"Using manually entered classes: {', '.join(class_names)}")

    with st.spinner("Extracting embeddings & reducing dimensions…"):
        records = discover_images(folders, class_names)
        if not records:
            st.error("No images found in the specified folders.")
            return

        embeddings_per_model: dict[str, dict[str, np.ndarray]] = {}
        for model_name in selected_models:
            embed_fn = load_model(model_name)
            all_embs = []
            for folder in folders:
                folder_records = [r for r in records if r["split"] == folder.name]
                folder_paths = [r["path"] for r in folder_records]
                cache_path = (
                    folder / f"embeddings_{model_name}" / "embeddings.npz"
                )
                all_embs.append(
                    extract_embeddings(folder_paths, embed_fn, cache_path=cache_path)
                )
            embeddings = np.vstack(all_embs)

            pca = PCA(n_components=2, random_state=42)
            pca_2d = pca.fit_transform(embeddings)

            perplexity = min(30, max(5, len(records) - 1))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_2d = tsne.fit_transform(embeddings)

            embeddings_per_model[model_name] = {"pca": pca_2d, "tsne": tsne_2d}

    fig = build_plotly_figure(records, embeddings_per_model)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "⬇ Download HTML",
        data=fig.to_html(include_plotlyjs="cdn"),
        file_name="embeddings_visualization.html",
        mime="text/html",
    )
```

- [ ] **Step 2: Verify tests still pass**

Run: `conda run -n visiondiy python -m pytest tests/test_app.py tests/test_visualize_embeddings.py -v`

Expected: all PASSED (no regressions).

- [ ] **Step 3: Manually launch and test**

Run: `conda run -n visiondiy streamlit run scripts/app.py`

Open http://localhost:8501 in a browser.

1. Select "Visualize Embeddings" in the sidebar (should be default).
2. Enter `dataset/train` and `dataset/test` (one per line) in "Folders".
3. Leave "Models" as default (all selected).
4. Click ▶ Run.
5. Verify: green success message if `classes.txt` found, or blue info if using manual classes.
6. Verify: Plotly scatter chart renders with model·PCA / model·t-SNE buttons.
7. Verify: "⬇ Download HTML" button appears and downloads a valid HTML file.

Stop with Ctrl+C.

- [ ] **Step 4: Commit**

```bash
git add scripts/app.py
git commit -m "feat: implement Visualize Embeddings Streamlit UI"
```

---

### Task 4: Compare Distributions UI

**Files:**
- Modify: `scripts/app.py` — replace `_compare_distributions_ui` stub

- [ ] **Step 1: Replace the stub with the full implementation**

In `scripts/app.py`, replace:

```python
def _compare_distributions_ui() -> None:
    pass  # Task 4
```

with:

```python
def _compare_distributions_ui() -> None:
    st.header("Compare Distributions")

    with st.sidebar:
        folder_a = st.text_input("Folder A (direct image folder)", placeholder="dataset/train/images")
        folder_b = st.text_input("Folder B (direct image folder)", placeholder="goal/images")
        all_models = available_models()
        if not all_models:
            st.error("No .pth models found in ./models/. Add a model file and restart.")
            return
        selected_model = st.selectbox("Model", all_models)
        name = st.text_input("Output name prefix", value="comparison")
        lpips_pairs = st.number_input("LPIPS pairs", min_value=1, value=500, step=50)
        run = st.button("▶ Run", use_container_width=True, key="run_cmp")

    if not run:
        st.info("Configure inputs in the sidebar and click ▶ Run.")
        return

    path_a = Path(folder_a.strip()) if folder_a.strip() else None
    path_b = Path(folder_b.strip()) if folder_b.strip() else None

    if not path_a or not path_b:
        st.error("Enter both Folder A and Folder B paths.")
        return
    if not path_a.exists():
        st.error(f"Folder A not found: {path_a}")
        return
    if not path_b.exists():
        st.error(f"Folder B not found: {path_b}")
        return

    paths_a = get_image_paths(path_a)
    paths_b = get_image_paths(path_b)

    if not paths_a:
        st.error(f"No images found in Folder A: {path_a}")
        return
    if not paths_b:
        st.error(f"No images found in Folder B: {path_b}")
        return

    with st.spinner("Computing embeddings, FID, and LPIPS…"):
        embed_fn = load_model(selected_model)
        cache_a = path_a.parent / f"embeddings_{selected_model}" / "embeddings.npz"
        cache_b = path_b.parent / f"embeddings_{selected_model}" / "embeddings.npz"
        emb_a = extract_embeddings(paths_a, embed_fn, cache_path=cache_a)
        emb_b = extract_embeddings(paths_b, embed_fn, cache_path=cache_b)

        combined = np.vstack([emb_a, emb_b])
        pca = PCA(n_components=2, random_state=42)
        pca_2d = pca.fit_transform(combined)

        fid_score = compute_fid(str(path_a), str(path_b))
        lpips_score = compute_lpips_score(paths_a, paths_b, n_pairs=int(lpips_pairs))

    col1, col2 = st.columns(2)
    col1.metric("FID", f"{fid_score:.4f}")
    col2.metric("LPIPS", f"{lpips_score:.4f}")

    fig = build_projection_figure(
        paths_a, paths_b, pca_2d,
        name_a=path_a.name,
        name_b=path_b.name,
        fid_score=fid_score,
        lpips_score=lpips_score,
    )
    st.plotly_chart(fig, use_container_width=True)

    metrics = {
        "fid": round(fid_score, 4),
        "lpips": round(lpips_score, 4),
        "n_a": len(paths_a),
        "n_b": len(paths_b),
        "folder_a": str(path_a),
        "folder_b": str(path_b),
        "model": selected_model,
    }

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "⬇ Download HTML",
        data=fig.to_html(include_plotlyjs="cdn").encode(),
        file_name=f"{name}_projection.html",
        mime="text/html",
    )
    dl2.download_button(
        "⬇ Download JSON",
        data=json.dumps(metrics, indent=2).encode(),
        file_name=f"{name}_metrics.json",
        mime="application/json",
    )
```

- [ ] **Step 2: Verify all tests pass**

Run: `conda run -n visiondiy python -m pytest tests/ -v`

Expected: all PASSED.

- [ ] **Step 3: Manually launch and test Compare Distributions**

Run: `conda run -n visiondiy streamlit run scripts/app.py`

Open http://localhost:8501.

1. Select "Compare Distributions" in the sidebar radio.
2. Enter direct image folder paths in "Folder A" and "Folder B" (e.g., `dataset/train/images` and `goal/images`).
3. Select a model, leave LPIPS pairs at 500.
4. Click ▶ Run.
5. Verify: FID and LPIPS metric cards appear.
6. Verify: Plotly scatter chart renders showing two groups.
7. Verify: "⬇ Download HTML" and "⬇ Download JSON" buttons appear and produce valid files.

Stop with Ctrl+C.

- [ ] **Step 4: Commit**

```bash
git add scripts/app.py
git commit -m "feat: implement Compare Distributions Streamlit UI"
```
