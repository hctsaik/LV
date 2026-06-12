"""Download Chinese-CLIP (ViT-B/16) into models/ for fully offline use (F7).

Usage:  python scripts/download_chinese_clip.py

Weights are ~750MB and are NOT committed to git (see .gitignore) — run this
once per machine. After that the app loads them with local_files_only=True.
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "OFA-Sys/chinese-clip-vit-base-patch16"
TARGET = Path(__file__).parent.parent / "models" / "chinese-clip-vit-base-patch16"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO_ID,
        local_dir=str(TARGET),
        ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "flax*", "tf_*"],
    )
    print(f"done → {TARGET}")


if __name__ == "__main__":
    main()
