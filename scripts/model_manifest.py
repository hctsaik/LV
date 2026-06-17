"""Single source of truth for LV model weights — driven by MODELS.md.

MODELS.md (repo root) carries a fenced ```json block listing every weight the
app needs: which FEATURE uses it, where it lives, how to fetch it. This module
parses that block so provisioning (scripts/setup_models.py) and the runtime
error messages (debugging) agree on ONE list — edit MODELS.md, both follow.

Kept torch-free and stdlib-only so importing it just to format an error message
stays cheap.

CLI:
    python scripts/model_manifest.py     # print ✅/❌ status per model + feature
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
MANIFEST_PATH = _REPO_ROOT / "MODELS.md"

# Prefer the block fenced between the MANIFEST:BEGIN/END anchors; fall back to
# the first ```json block anywhere in the file.
_ANCHORED = re.compile(r"MANIFEST:BEGIN.*?```json\s*(.*?)```", re.DOTALL)
_ANY_JSON = re.compile(r"```json\s*(.*?)```", re.DOTALL)


# Every weight lives under ONE root: models/. Each model gets its own same-named
# subfolder (see MODELS.md). LV_MODELS_DIR relocates the whole root.
def _models_base() -> Path:
    return Path(os.environ.get("LV_MODELS_DIR") or (_REPO_ROOT / "models"))


class ModelSpec:
    """One row of the manifest."""

    def __init__(self, d: dict[str, Any]) -> None:
        self.id: str = d["id"]
        self.title: str = d.get("title", d["id"])
        self.features: list[str] = d.get("features", [])
        self.tier: str = d.get("tier", "core")
        self.path: str = d.get("path", "")
        self.sentinel: str = d.get("sentinel", self.path)
        self.acquire: str = d.get("acquire", "url")
        self.url: str | None = d.get("url")
        self.repo: str | None = d.get("repo")
        self.ignore: list[str] = d.get("ignore", [])
        self.size_mb = d.get("size_mb")

    @property
    def base_dir(self) -> Path:
        return _models_base()

    @property
    def base_env(self) -> str:
        return "LV_MODELS_DIR"

    @property
    def folder(self) -> Path:
        """The model's own subfolder under models/ (the named placeholder)."""
        return self.base_dir / self.path.split("/", 1)[0]

    @property
    def target(self) -> Path:
        return self.base_dir / self.path

    @property
    def sentinel_path(self) -> Path:
        return self.base_dir / self.sentinel

    def is_present(self) -> bool:
        return self.sentinel_path.exists()

    @property
    def features_label(self) -> str:
        return " / ".join(self.features) if self.features else "(未標註功能)"


def load_specs(manifest_path: Path = MANIFEST_PATH) -> list[ModelSpec]:
    """Parse the json block out of MODELS.md. Returns [] if absent/unparseable."""
    if not manifest_path.exists():
        return []
    text = manifest_path.read_text(encoding="utf-8")
    m = _ANCHORED.search(text) or _ANY_JSON.search(text)
    if not m:
        return []
    data = json.loads(m.group(1))
    return [ModelSpec(d) for d in data.get("models", [])]


def spec_by_id(model_id: str) -> ModelSpec | None:
    for s in load_specs():
        if s.id == model_id:
            return s
    return None


def explain(
    model_id: str,
    *,
    feature: str | None = None,
    expected: Path | str | None = None,
) -> str:
    """Build a debug-friendly message naming the FEATURE + MODEL that's missing,
    the path we looked at, and the one-line fix. Drop it straight into an
    exception so a failure reads e.g. 「功能〔Compare · LPIPS〕讀不到 model …」.

    Resilient: works even if MODELS.md is missing (falls back to a generic
    message) so error reporting never crashes on its own.
    """
    s = spec_by_id(model_id)
    if s is None:
        feat = feature or "(未知功能)"
        loc = f"\n  預期路徑 : {expected}" if expected else ""
        return (
            f"功能〔{feat}〕讀不到 model〔{model_id}〕。{loc}\n"
            f"  修法     : 見 MODELS.md;或執行 python scripts/setup_models.py"
        )
    feat = feature or s.features_label
    tier_flag = "" if s.tier == "core" else " --with-compare"
    return (
        f"功能〔{feat}〕讀不到 model〔{s.title}〕。\n"
        f"  預期路徑 : {expected or s.sentinel_path}\n"
        f"  根目錄   : {s.base_dir}  (可用環境變數 {s.base_env} 覆寫)\n"
        f"  修法     : python scripts/setup_models.py{tier_flag}\n"
        f"  細節     : 見 MODELS.md(model id: {s.id});"
        f"或 python scripts/model_manifest.py 看全部到位狀態"
    )


def main(argv: list[str] | None = None) -> int:
    specs = load_specs()
    if not specs:
        print(f"[err] 找不到或無法解析 manifest: {MANIFEST_PATH}")
        return 1

    print(f"manifest      : {MANIFEST_PATH}")
    print(f"models 根目錄 : {_models_base()}  (LV_MODELS_DIR 可覆寫)")
    print()

    width = max(len(s.title) for s in specs)
    missing_core = 0
    for s in specs:
        ok = s.is_present()
        mark = "OK " if ok else "MISS"
        print(f"[{mark}] [{s.tier:7}] {s.title:<{width}}  <- {s.features_label}")
        if not ok:
            print(f"          缺檔: {s.sentinel_path}")
            if s.tier == "core":
                missing_core += 1
    print()

    if all(s.is_present() for s in specs):
        print("[ok] 全部 model 到位。")
        return 0
    print("[!] 有 model 缺漏 — core: python scripts/setup_models.py;"
          "compare: 加 --with-compare")
    return 1 if missing_core else 0


if __name__ == "__main__":
    raise SystemExit(main())
