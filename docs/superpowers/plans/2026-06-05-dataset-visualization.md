# Dataset Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增兩個 CLI 腳本：`visualize_embeddings.py`（PCA/t-SNE 互動視覺化）與 `compare_distributions.py`（FID/LPIPS 分布比較），共用 `_utils.py`。

**Architecture:** 三支腳本放在 `scripts/`。`_utils.py` 重用既有 `retrieval/` 模組的 Dinov2、Clip、ImagePreprocessor 類別。兩支 CLI 腳本各自獨立，互動式輸出用 Plotly（.html），靜態輸出用 Matplotlib（.png），統計指標輸出為 JSON。

**Tech Stack:** PyTorch, scikit-learn (PCA/t-SNE), Plotly, Matplotlib, clean-fid (FID), lpips (LPIPS), pytest

---

## File Map

| 檔案 | 動作 | 責任 |
|---|---|---|
| `requirements.txt` | Modify | 新增 scikit-learn, plotly, clean-fid, lpips, torchvision |
| `scripts/_utils.py` | Create | load_model, extract_embeddings, save_figure |
| `scripts/visualize_embeddings.py` | Create | Feature 1：label 解析、image 探索、PCA/t-SNE 圖表、CLI |
| `scripts/compare_distributions.py` | Create | Feature 2：FID/LPIPS 計算、投影圖表、CLI |
| `tests/conftest.py` | Create | sys.path 設定，讓 scripts/ 與 retrieval/ 皆可 import |
| `tests/test_utils.py` | Create | _utils.py 單元測試 |
| `tests/test_visualize_embeddings.py` | Create | label 解析、圖表建構測試 |
| `tests/test_compare_distributions.py` | Create | 圖表建構、get_image_paths 測試 |

---

## Task 1：安裝相依套件

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1：更新 requirements.txt**

在 `requirements.txt` 現有內容後附加：
```
scikit-learn>=1.3.0
plotly>=5.18.0
clean-fid>=0.1.35
lpips>=0.1.4
torchvision>=0.17.0
pandas>=2.0.0
```

- [ ] **Step 2：安裝至 visiondiy 環境**

```bash
conda run -n visiondiy pip install scikit-learn plotly clean-fid lpips torchvision pandas
```

Expected：所有套件安裝成功，無版本衝突。

- [ ] **Step 3：驗證安裝**

```bash
conda run -n visiondiy python -c "import sklearn, plotly, cleanfid, lpips, torchvision, pandas; print('OK')"
```

Expected output：`OK`

- [ ] **Step 4：Commit**

```bash
git add requirements.txt
git commit -m "deps: add scikit-learn, plotly, clean-fid, lpips, torchvision for dataset visualization"
```

---

## Task 2：tests/conftest.py + scripts/_utils.py（TDD）

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_utils.py`
- Create: `scripts/_utils.py`

- [ ] **Step 1：建立 tests/conftest.py**

```python
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
```

- [ ] **Step 2：撰寫失敗測試**

`tests/test_utils.py`：
```python
import numpy as np
import pytest
from pathlib import Path


def test_extract_embeddings_shape(tmp_path):
    from _utils import extract_embeddings

    def mock_embed(path: Path) -> np.ndarray:
        return np.ones(64)

    paths = [tmp_path / f"{i}.jpg" for i in range(5)]
    for p in paths:
        p.write_bytes(b"x")

    result = extract_embeddings(paths, mock_embed)
    assert result.shape == (5, 64)


def test_extract_embeddings_preserves_values(tmp_path):
    from _utils import extract_embeddings

    def mock_embed(path: Path) -> np.ndarray:
        return np.array([float(path.stem)])

    paths = [tmp_path / f"{i}.jpg" for i in range(3)]
    for p in paths:
        p.write_bytes(b"x")

    result = extract_embeddings(paths, mock_embed)
    assert result[0, 0] == 0.0
    assert result[1, 0] == 1.0
    assert result[2, 0] == 2.0


def test_save_figure_creates_html_and_png(tmp_path):
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from _utils import save_figure

    pf = go.Figure(go.Scatter(x=[1], y=[2]))
    mf, ax = plt.subplots()
    ax.plot([1], [2])

    save_figure(pf, mf, tmp_path, "out")
    plt.close(mf)

    assert (tmp_path / "out.html").exists()
    assert (tmp_path / "out.png").exists()


def test_save_figure_creates_output_dir(tmp_path):
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from _utils import save_figure

    out_dir = tmp_path / "new_dir"
    pf = go.Figure(go.Scatter(x=[1], y=[2]))
    mf, _ = plt.subplots()

    save_figure(pf, mf, out_dir, "out")
    plt.close(mf)

    assert (out_dir / "out.html").exists()
```

- [ ] **Step 3：執行確認失敗**

```bash
conda run -n visiondiy pytest tests/test_utils.py -v
```

Expected：`ImportError: No module named '_utils'`

- [ ] **Step 4：建立 scripts/_utils.py**

```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import Clip, Dinov2, ImagePreprocessor

MULTIMODAL_REGISTRY = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip2_base": "google/siglip2-base-patch16-224",
}


def load_model(model_name: str) -> Callable[[Path], np.ndarray]:
    """Return embed_fn(image_path) -> np.ndarray."""
    preprocessor = ImagePreprocessor()
    if model_name == "dinov2":
        model = Dinov2(model_size="small")

        def embed_fn(path: Path) -> np.ndarray:
            return model(preprocessor.preprocess(path))
    else:
        hf_name = MULTIMODAL_REGISTRY.get(model_name)
        if hf_name is None:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose: dinov2, siglip2_base, clip"
            )
        model = Clip(model_name=hf_name)

        def embed_fn(path: Path) -> np.ndarray:
            return model.extract_image_features(preprocessor.preprocess(path))

    return embed_fn


def extract_embeddings(
    image_paths: list[Path], embed_fn: Callable[[Path], np.ndarray]
) -> np.ndarray:
    """Extract embeddings for all images. Returns shape (N, D)."""
    return np.stack([
        embed_fn(p) for p in tqdm(image_paths, desc="Extracting embeddings")
    ])


def save_figure(plotly_fig, mpl_fig, output_dir: Path, name: str) -> None:
    """Save Plotly figure as .html and Matplotlib figure as .png."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plotly_fig.write_html(str(output_dir / f"{name}.html"))
    mpl_fig.savefig(str(output_dir / f"{name}.png"), dpi=150, bbox_inches="tight")
```

- [ ] **Step 5：執行確認通過**

```bash
conda run -n visiondiy pytest tests/test_utils.py -v
```

Expected：4 PASSED

- [ ] **Step 6：Commit**

```bash
git add scripts/_utils.py tests/conftest.py tests/test_utils.py
git commit -m "feat: add _utils module (load_model, extract_embeddings, save_figure)"
```

---

## Task 3：visualize_embeddings.py — label 解析與圖片探索（TDD）

**Files:**
- Create: `tests/test_visualize_embeddings.py`
- Create: `scripts/visualize_embeddings.py`（含 parse_label_file、discover_images，其餘函式待 Task 4 加入）

- [ ] **Step 1：撰寫失敗測試**

`tests/test_visualize_embeddings.py`：
```python
import numpy as np
import pytest
from pathlib import Path


def test_parse_label_single_class(tmp_path):
    from visualize_embeddings import parse_label_file

    (tmp_path / "img.txt").write_text("0 0.5 0.5 0.3 0.4\n")
    assert parse_label_file(tmp_path / "img.txt", ["apple", "banana", "orange"]) == "apple"


def test_parse_label_class_id_1(tmp_path):
    from visualize_embeddings import parse_label_file

    (tmp_path / "img.txt").write_text("1 0.5 0.5 0.3 0.4\n")
    assert parse_label_file(tmp_path / "img.txt", ["apple", "banana", "orange"]) == "banana"


def test_parse_label_multi_class_is_mix(tmp_path):
    from visualize_embeddings import parse_label_file

    (tmp_path / "img.txt").write_text("0 0.5 0.5 0.3 0.4\n1 0.2 0.3 0.1 0.1\n")
    assert parse_label_file(tmp_path / "img.txt", ["apple", "banana"]) == "mix"


def test_parse_label_missing_file_is_unknown(tmp_path):
    from visualize_embeddings import parse_label_file

    assert parse_label_file(tmp_path / "missing.txt", ["apple"]) == "unknown"


def test_parse_label_empty_file_is_unknown(tmp_path):
    from visualize_embeddings import parse_label_file

    (tmp_path / "empty.txt").write_text("")
    assert parse_label_file(tmp_path / "empty.txt", ["apple"]) == "unknown"


def test_discover_images_finds_multiple_splits(tmp_path):
    from visualize_embeddings import discover_images

    for split in ("train", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        (tmp_path / split / "images" / "apple_1.jpg").write_bytes(b"fake")
        (tmp_path / split / "labels" / "apple_1.txt").write_text("0 0.5 0.5 0.3 0.4\n")

    records = discover_images(tmp_path, ["apple", "banana"])
    assert len(records) == 2
    assert {r["split"] for r in records} == {"train", "test"}


def test_discover_images_skips_absent_split(tmp_path):
    from visualize_embeddings import discover_images

    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "images" / "x.jpg").write_bytes(b"fake")
    (tmp_path / "train" / "labels" / "x.txt").write_text("0 0.5 0.5 0.3 0.4\n")

    records = discover_images(tmp_path, ["apple"])
    assert all(r["split"] == "train" for r in records)


def test_discover_images_label_assigned(tmp_path):
    from visualize_embeddings import discover_images

    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "images" / "img.jpg").write_bytes(b"fake")
    (tmp_path / "train" / "labels" / "img.txt").write_text("2 0.5 0.5 0.3 0.4\n")

    records = discover_images(tmp_path, ["apple", "banana", "orange"])
    assert records[0]["label"] == "orange"
```

- [ ] **Step 2：執行確認失敗**

```bash
conda run -n visiondiy pytest tests/test_visualize_embeddings.py -v
```

Expected：`ImportError: No module named 'visualize_embeddings'`

- [ ] **Step 3：建立 scripts/visualize_embeddings.py（只含 parsing 函式）**

```python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from _utils import extract_embeddings, load_model, save_figure

_COLORS = ["#e74c3c", "#f39c12", "#2ecc71", "#9b59b6", "#3498db", "#1abc9c", "#95a5a6"]
_SYMBOLS = {"train": "circle", "test": "square", "valid": "diamond"}


def parse_label_file(label_path: Path, class_names: list[str]) -> str:
    """YOLO label → class name | 'mix' | 'unknown'."""
    if not label_path.exists():
        return "unknown"
    lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
    if not lines:
        return "unknown"
    class_ids = {int(ln.split()[0]) for ln in lines}
    if len(class_ids) > 1:
        return "mix"
    cid = next(iter(class_ids))
    return class_names[cid] if cid < len(class_names) else f"class_{cid}"


def discover_images(dataset_dir: Path, class_names: list[str]) -> list[dict]:
    """探索 train/test/valid 下的影像，回傳 list of {path, split, label}。"""
    records = []
    for split in ("train", "test", "valid"):
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.exists():
            continue
        for img_path in sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png") for p in images_dir.glob(ext)
        ):
            label_path = labels_dir / f"{img_path.stem}.txt"
            records.append({
                "path": img_path,
                "split": split,
                "label": parse_label_file(label_path, class_names),
            })
    return records
```

- [ ] **Step 4：執行確認通過**

```bash
conda run -n visiondiy pytest tests/test_visualize_embeddings.py -v
```

Expected：8 PASSED

- [ ] **Step 5：Commit**

```bash
git add scripts/visualize_embeddings.py tests/test_visualize_embeddings.py
git commit -m "feat: add visualize_embeddings label parsing and image discovery"
```

---

## Task 4：visualize_embeddings.py — 圖表建構與 main

**Files:**
- Modify: `scripts/visualize_embeddings.py`（附加圖表函式與 main）
- Modify: `tests/test_visualize_embeddings.py`（附加圖表測試）

- [ ] **Step 1：在 scripts/visualize_embeddings.py 結尾附加圖表函式與 main**

```python
def _label_color_map(labels: list[str]) -> dict[str, str]:
    unique = sorted(set(labels))
    return {lbl: _COLORS[i % len(_COLORS)] for i, lbl in enumerate(unique)}


def build_plotly_figure(
    records: list[dict], pca_2d: np.ndarray, tsne_2d: np.ndarray
) -> go.Figure:
    """互動式圖表：PCA/t-SNE 切換 + split 篩選按鈕 + legend 類別切換。"""
    unique_labels = sorted({r["label"] for r in records})
    unique_splits = sorted({r["split"] for r in records})
    color_map = _label_color_map(unique_labels)

    traces: list[go.Scatter] = []
    trace_meta: list[dict] = []

    for label in unique_labels:
        for split in unique_splits:
            idx = [i for i, r in enumerate(records)
                   if r["label"] == label and r["split"] == split]
            if not idx:
                continue
            traces.append(go.Scatter(
                x=[pca_2d[i, 0] for i in idx],
                y=[pca_2d[i, 1] for i in idx],
                mode="markers",
                name=f"{label} ({split})",
                legendgroup=label,
                marker=dict(
                    color=color_map[label],
                    symbol=_SYMBOLS.get(split, "circle"),
                    size=7, opacity=0.8,
                ),
                text=[records[i]["path"].name for i in idx],
                hovertemplate="%{text}<br>"
                              + f"Label: {label}<br>Split: {split}"
                              + "<extra></extra>",
            ))
            trace_meta.append({
                "label": label, "split": split,
                "pca_x": [pca_2d[i, 0] for i in idx],
                "pca_y": [pca_2d[i, 1] for i in idx],
                "tsne_x": [tsne_2d[i, 0] for i in idx],
                "tsne_y": [tsne_2d[i, 1] for i in idx],
            })

    method_buttons = [
        dict(method="restyle", label="PCA", args=[{
            "x": [m["pca_x"] for m in trace_meta],
            "y": [m["pca_y"] for m in trace_meta],
        }]),
        dict(method="restyle", label="t-SNE", args=[{
            "x": [m["tsne_x"] for m in trace_meta],
            "y": [m["tsne_y"] for m in trace_meta],
        }]),
    ]

    split_buttons = [
        dict(method="restyle", label="All Splits",
             args=[{"visible": [True] * len(traces)}])
    ]
    for s in unique_splits:
        split_buttons.append(dict(
            method="restyle", label=s.capitalize(),
            args=[{"visible": [m["split"] == s for m in trace_meta]}],
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Dataset Embedding Visualization",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        legend=dict(title="Class (Split)", groupclick="toggleitem"),
        updatemenus=[
            dict(type="buttons", direction="right", x=0.0, y=1.12,
                 showactive=True, buttons=method_buttons,
                 bgcolor="#f0f0f0", bordercolor="#ccc"),
            dict(type="buttons", direction="right", x=0.38, y=1.12,
                 showactive=True, buttons=split_buttons,
                 bgcolor="#e8f4fd", bordercolor="#aad4f0"),
        ],
    )
    return fig


def build_matplotlib_figures(
    records: list[dict], pca_2d: np.ndarray, tsne_2d: np.ndarray
) -> tuple:
    """回傳 (pca_fig, tsne_fig)，每個類別一種顏色，所有 split 合併。"""
    unique_labels = sorted({r["label"] for r in records})
    color_map = _label_color_map(unique_labels)

    figs = []
    for coords, method_name in [(pca_2d, "PCA"), (tsne_2d, "t-SNE")]:
        fig, ax = plt.subplots(figsize=(10, 8))
        for label in unique_labels:
            idx = [i for i, r in enumerate(records) if r["label"] == label]
            ax.scatter(
                coords[idx, 0], coords[idx, 1],
                c=color_map[label], label=label, alpha=0.7, s=30,
            )
        ax.set_title(f"Dataset Embeddings — {method_name}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()
        figs.append(fig)
    return figs[0], figs[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize dataset embeddings (PCA/t-SNE)")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="資料集根目錄（含 train/test/valid 子資料夾）")
    parser.add_argument("--model", default="siglip2_base",
                        choices=["dinov2", "siglip2_base", "clip"])
    parser.add_argument("--classes", nargs="+", default=["apple", "banana", "orange"],
                        help="YOLO class ID 順序對應的類別名稱（0-indexed）")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    args = parser.parse_args()

    records = discover_images(args.dataset_dir, args.classes)
    if not records:
        print(f"No images found in {args.dataset_dir}")
        return

    print(f"Found {len(records)} images across {sorted({r['split'] for r in records})} splits")

    embed_fn = load_model(args.model)
    embeddings = extract_embeddings([r["path"] for r in records], embed_fn)

    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(embeddings)

    perplexity = min(30, max(5, len(records) - 1))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_2d = tsne.fit_transform(embeddings)

    plotly_fig = build_plotly_figure(records, pca_2d, tsne_2d)
    pca_fig, tsne_fig = build_matplotlib_figures(records, pca_2d, tsne_2d)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotly_fig.write_html(str(args.output_dir / "embeddings_visualization.html"))
    pca_fig.savefig(str(args.output_dir / "embeddings_pca.png"), dpi=150, bbox_inches="tight")
    tsne_fig.savefig(str(args.output_dir / "embeddings_tsne.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    print(f"\nSaved to {args.output_dir}/")
    print("  embeddings_visualization.html")
    print("  embeddings_pca.png")
    print("  embeddings_tsne.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2：在 tests/test_visualize_embeddings.py 結尾附加圖表測試**

```python
import numpy as np  # 附加至檔案頭部（若尚未有此 import）


def _make_records():
    return [
        {"path": Path("a.jpg"), "split": "train", "label": "apple"},
        {"path": Path("b.jpg"), "split": "train", "label": "banana"},
        {"path": Path("c.jpg"), "split": "test",  "label": "apple"},
        {"path": Path("d.jpg"), "split": "test",  "label": "mix"},
    ]


def test_build_plotly_figure_has_traces():
    from visualize_embeddings import build_plotly_figure

    records = _make_records()
    fig = build_plotly_figure(records, np.random.rand(4, 2), np.random.rand(4, 2))
    assert len(fig.data) > 0


def test_build_plotly_figure_has_two_updatemenus():
    from visualize_embeddings import build_plotly_figure

    records = _make_records()
    fig = build_plotly_figure(records, np.random.rand(4, 2), np.random.rand(4, 2))
    assert len(fig.layout.updatemenus) == 2


def test_build_matplotlib_figures_returns_two():
    import matplotlib.pyplot as plt
    from visualize_embeddings import build_matplotlib_figures

    records = _make_records()
    pca_fig, tsne_fig = build_matplotlib_figures(
        records, np.random.rand(4, 2), np.random.rand(4, 2)
    )
    assert pca_fig is not None
    assert tsne_fig is not None
    plt.close("all")
```

- [ ] **Step 3：執行所有測試**

```bash
conda run -n visiondiy pytest tests/test_utils.py tests/test_visualize_embeddings.py -v
```

Expected：全部 PASSED

- [ ] **Step 4：Commit**

```bash
git add scripts/visualize_embeddings.py tests/test_visualize_embeddings.py
git commit -m "feat: add visualize_embeddings figure builders and CLI entry point"
```

---

## Task 5：compare_distributions.py（TDD）

**Files:**
- Create: `tests/test_compare_distributions.py`
- Create: `scripts/compare_distributions.py`

- [ ] **Step 1：撰寫失敗測試**

`tests/test_compare_distributions.py`：
```python
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _make_images(folder: Path, n: int) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = folder / f"img_{i}.jpg"
        Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        ).save(p)
        paths.append(p)
    return paths


def test_get_image_paths_finds_jpg(tmp_path):
    from compare_distributions import get_image_paths

    _make_images(tmp_path, 3)
    paths = get_image_paths(tmp_path)
    assert len(paths) == 3
    assert all(p.suffix == ".jpg" for p in paths)


def test_get_image_paths_empty_folder(tmp_path):
    from compare_distributions import get_image_paths

    assert get_image_paths(tmp_path) == []


def test_build_projection_figure_has_two_traces():
    from compare_distributions import build_projection_figure

    paths_a = [Path(f"a{i}.jpg") for i in range(3)]
    paths_b = [Path(f"b{i}.jpg") for i in range(2)]
    pca_2d = np.random.rand(5, 2)
    fig = build_projection_figure(paths_a, paths_b, pca_2d, "train", "test", 10.5, 0.35)
    assert len(fig.data) == 2


def test_build_projection_figure_title_contains_metrics():
    from compare_distributions import build_projection_figure

    paths_a = [Path(f"a{i}.jpg") for i in range(2)]
    paths_b = [Path(f"b{i}.jpg") for i in range(2)]
    pca_2d = np.random.rand(4, 2)
    fig = build_projection_figure(paths_a, paths_b, pca_2d, "A", "B", 12.34, 0.56)
    assert "12.34" in fig.layout.title.text
    assert "0.56" in fig.layout.title.text


def test_build_matplotlib_figure_returns_figure():
    from compare_distributions import build_matplotlib_figure

    paths_a = [Path(f"a{i}.jpg") for i in range(2)]
    paths_b = [Path(f"b{i}.jpg") for i in range(2)]
    pca_2d = np.random.rand(4, 2)
    fig = build_matplotlib_figure(paths_a, paths_b, pca_2d, "A", "B", 5.0, 0.3)
    assert fig is not None
    plt.close("all")
```

- [ ] **Step 2：執行確認失敗**

```bash
conda run -n visiondiy pytest tests/test_compare_distributions.py -v
```

Expected：`ImportError: No module named 'compare_distributions'`

- [ ] **Step 3：建立 scripts/compare_distributions.py**

```python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA

from _utils import extract_embeddings, load_model, save_figure


def get_image_paths(folder: Path) -> list[Path]:
    return sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png") for p in folder.glob(ext)
    )


def compute_fid(folder_a: str, folder_b: str) -> float:
    from cleanfid import fid as cleanfid
    return float(cleanfid.compute_fid(folder_a, folder_b))


def compute_lpips_score(
    paths_a: list[Path], paths_b: list[Path], n_pairs: int = 500
) -> float:
    import lpips

    loss_fn = lpips.LPIPS(net="alex")
    loss_fn.eval()

    n = min(n_pairs, len(paths_a), len(paths_b))
    sampled_a = random.sample(paths_a, n)
    sampled_b = random.sample(paths_b, n)

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    total = 0.0
    with torch.no_grad():
        for pa, pb in zip(sampled_a, sampled_b):
            ia = transform(Image.open(pa).convert("RGB")).unsqueeze(0)
            ib = transform(Image.open(pb).convert("RGB")).unsqueeze(0)
            total += loss_fn(ia, ib).item()
    return total / n


def build_projection_figure(
    paths_a: list[Path],
    paths_b: list[Path],
    pca_2d: np.ndarray,
    name_a: str,
    name_b: str,
    fid_score: float,
    lpips_score: float,
) -> go.Figure:
    n_a = len(paths_a)
    fig = go.Figure(data=[
        go.Scatter(
            x=pca_2d[:n_a, 0].tolist(), y=pca_2d[:n_a, 1].tolist(),
            mode="markers", name=name_a,
            marker=dict(color="#3498db", size=6, opacity=0.7),
            text=[p.name for p in paths_a],
            hovertemplate="%{text}<br>Group: " + name_a + "<extra></extra>",
        ),
        go.Scatter(
            x=pca_2d[n_a:, 0].tolist(), y=pca_2d[n_a:, 1].tolist(),
            mode="markers", name=name_b,
            marker=dict(color="#e74c3c", size=6, opacity=0.7),
            text=[p.name for p in paths_b],
            hovertemplate="%{text}<br>Group: " + name_b + "<extra></extra>",
        ),
    ])
    fig.update_layout(
        title=(
            f"Distribution Comparison: {name_a} vs {name_b}<br>"
            f"<sub>FID: {fid_score:.2f} | LPIPS: {lpips_score:.4f}</sub>"
        ),
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        legend=dict(title="Group"),
    )
    return fig


def build_matplotlib_figure(
    paths_a: list[Path],
    paths_b: list[Path],
    pca_2d: np.ndarray,
    name_a: str,
    name_b: str,
    fid_score: float,
    lpips_score: float,
) -> plt.Figure:
    n_a = len(paths_a)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pca_2d[:n_a, 0], pca_2d[:n_a, 1],
               c="#3498db", label=name_a, alpha=0.6, s=25)
    ax.scatter(pca_2d[n_a:, 0], pca_2d[n_a:, 1],
               c="#e74c3c", label=name_b, alpha=0.6, s=25)
    ax.set_title(
        f"{name_a} vs {name_b}  |  FID: {fid_score:.2f}, LPIPS: {lpips_score:.4f}"
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare image distribution between two folders (FID/LPIPS)"
    )
    parser.add_argument("--folder-a", type=Path, required=True)
    parser.add_argument("--folder-b", type=Path, required=True)
    parser.add_argument("--model", default="siglip2_base",
                        choices=["dinov2", "siglip2_base", "clip"])
    parser.add_argument("--name", default="comparison", help="輸出檔名前綴")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--lpips-pairs", type=int, default=500,
                        help="LPIPS 最大 cross-group pair 數")
    args = parser.parse_args()

    paths_a = get_image_paths(args.folder_a)
    paths_b = get_image_paths(args.folder_b)

    if not paths_a:
        raise ValueError(f"No images found in {args.folder_a}")
    if not paths_b:
        raise ValueError(f"No images found in {args.folder_b}")

    print(f"Group A ({args.folder_a.name}): {len(paths_a)} images")
    print(f"Group B ({args.folder_b.name}): {len(paths_b)} images")

    embed_fn = load_model(args.model)
    emb_a = extract_embeddings(paths_a, embed_fn)
    emb_b = extract_embeddings(paths_b, embed_fn)

    combined = np.vstack([emb_a, emb_b])
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(combined)

    print("Computing FID...")
    fid_score = compute_fid(str(args.folder_a), str(args.folder_b))
    print(f"  FID: {fid_score:.4f}")

    print("Computing LPIPS...")
    lpips_score = compute_lpips_score(paths_a, paths_b, n_pairs=args.lpips_pairs)
    print(f"  LPIPS: {lpips_score:.4f}")

    name_a = args.folder_a.name
    name_b = args.folder_b.name
    plotly_fig = build_projection_figure(
        paths_a, paths_b, pca_2d, name_a, name_b, fid_score, lpips_score
    )
    mpl_fig = build_matplotlib_figure(
        paths_a, paths_b, pca_2d, name_a, name_b, fid_score, lpips_score
    )

    save_figure(plotly_fig, mpl_fig, args.output_dir, f"{args.name}_projection")
    plt.close("all")

    metrics = {
        "fid": round(fid_score, 4),
        "lpips": round(lpips_score, 4),
        "n_a": len(paths_a),
        "n_b": len(paths_b),
        "folder_a": str(args.folder_a),
        "folder_b": str(args.folder_b),
        "model": args.model,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.name}_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print(f"\nSaved to {args.output_dir}/")
    print(f"  {args.name}_projection.html")
    print(f"  {args.name}_projection.png")
    print(f"  {args.name}_metrics.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4：執行確認通過**

```bash
conda run -n visiondiy pytest tests/test_compare_distributions.py -v
```

Expected：5 PASSED

- [ ] **Step 5：全部測試一次執行**

```bash
conda run -n visiondiy pytest tests/ -v
```

Expected：全部 PASSED（共 17 tests）

- [ ] **Step 6：Commit**

```bash
git add scripts/compare_distributions.py tests/test_compare_distributions.py
git commit -m "feat: add compare_distributions CLI for FID/LPIPS distribution comparison"
```

---

## Task 6：Smoke test（實際資料驗證）

- [ ] **Step 1：執行 Feature 1**

```bash
conda run -n visiondiy python scripts/visualize_embeddings.py \
  --dataset-dir ./dataset \
  --model siglip2_base \
  --classes apple banana orange \
  --output-dir ./output
```

Expected：
```
Found N images across ['test', 'train'] splits
Extracting embeddings: 100%|...
Saved to output/
  embeddings_visualization.html
  embeddings_pca.png
  embeddings_tsne.png
```

確認：用瀏覽器開啟 `./output/embeddings_visualization.html`，應可見散點圖、PCA/t-SNE 切換按鈕、Split 篩選按鈕、Legend 類別切換。

- [ ] **Step 2：執行 Feature 2**

```bash
conda run -n visiondiy python scripts/compare_distributions.py \
  --folder-a ./dataset/train/images \
  --folder-b ./dataset/test/images \
  --model siglip2_base \
  --name train_vs_test \
  --output-dir ./output
```

Expected：
```
Group A (images): N images
Group B (images): M images
Computing FID...
  FID: X.XXXX
Computing LPIPS...
  LPIPS: X.XXXX

Saved to output/
  train_vs_test_projection.html
  train_vs_test_projection.png
  train_vs_test_metrics.json
```

確認：`cat ./output/train_vs_test_metrics.json` 應包含 `fid`、`lpips`、`n_a`、`n_b` 四個 key。

- [ ] **Step 3：新增 .gitignore 與 output placeholder**

在 `.gitignore` 中加入：
```
output/
```

```bash
echo "" > output/.gitkeep
git add .gitignore output/.gitkeep
git commit -m "chore: gitignore output/, add placeholder"
```
