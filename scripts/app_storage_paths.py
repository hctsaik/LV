from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping


def resolve_output_dir(source_output_dir: Path, environ: Mapping[str, str] | None = None) -> Path:
    """Use LV_OUTPUT_DIR when supplied; otherwise keep the standalone App default."""
    values = os.environ if environ is None else environ
    configured = values.get("LV_OUTPUT_DIR")
    return Path(configured) if configured else Path(source_output_dir)


def read_jsonl_logs(paths: Iterable[Path], *, newest_first: bool = False) -> list[dict]:
    """Read valid JSONL rows from unique files, optionally ordered by timestamp."""
    rows = []
    seen = set()
    for value in paths:
        path = Path(value)
        identity = path.resolve()
        if identity in seen:
            continue
        seen.add(identity)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if newest_first:
        ordered = sorted(
            enumerate(rows),
            key=lambda pair: (
                str(pair[1].get("ts") or "") if isinstance(pair[1], dict) else "",
                pair[0],
            ),
            reverse=True,
        )
        rows = [row for _, row in ordered]
    return rows
