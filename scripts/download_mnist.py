"""
下載 MNIST 並整理成 train/valid/test 結構，每個 split 內按數字分子資料夾。

輸出結構：
  dataset_mnist/
    train/   (50,000 張)
      0/ 1/ ... 9/
    valid/   (10,000 張，從原始訓練集切出)
      0/ 1/ ... 9/
    test/    (10,000 張，原始測試集)
      0/ 1/ ... 9/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import MNIST

DEST = Path(__file__).parent.parent / "dataset_mnist"
VALID_RATIO = 1 / 6  # 60000 * 1/6 ≈ 10000


def save_split(images: np.ndarray, labels: np.ndarray, split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for digit in range(10):
        (split_dir / str(digit)).mkdir(exist_ok=True)

    counters = [0] * 10
    for img_arr, label in zip(images, labels):
        digit = int(label)
        idx = counters[digit]
        counters[digit] += 1
        out_path = split_dir / str(digit) / f"{idx:05d}.png"
        Image.fromarray(img_arr).save(out_path)

    total = sum(counters)
    print(f"  {split_dir.name}: {total} 張 | " + " ".join(f"{d}:{counters[d]}" for d in range(10)))


def main() -> None:
    tmp = DEST / "_raw"
    tmp.mkdir(parents=True, exist_ok=True)

    print("下載 MNIST（首次需要網路，之後會使用快取）…")
    train_ds = MNIST(str(tmp), train=True, download=True)
    test_ds = MNIST(str(tmp), train=False, download=True)

    train_images = train_ds.data.numpy()
    train_labels = train_ds.targets.numpy()
    test_images = test_ds.data.numpy()
    test_labels = test_ds.targets.numpy()

    n_total = len(train_images)
    n_valid = round(n_total * VALID_RATIO)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_total)

    valid_idx = perm[:n_valid]
    train_idx = perm[n_valid:]

    print(f"\n切分比例：train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_images)}")
    print("寫入圖片中…")

    save_split(train_images[train_idx], train_labels[train_idx], DEST / "train")
    save_split(train_images[valid_idx], train_labels[valid_idx], DEST / "valid")
    save_split(test_images, test_labels, DEST / "test")

    print(f"\n完成。資料集位於：{DEST.resolve()}")


if __name__ == "__main__":
    main()
