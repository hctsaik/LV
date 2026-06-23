"""壞檔防呆:統一的「安全讀取」入口。

App 任何「讀使用者影像 / YOLO label / classes」的地方都應走這裡,讓**單一**損壞
或格式錯誤的檔案被『跳過』,而不是讓整個工具崩潰。

鐵則:**只讀不寫。** 壞檔只是被略過 + 回報,**絕不刪除或改寫使用者資料集**
(見 CLAUDE.md / 記憶 no-dataset-writes)。原始檔案原封不動。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

# 解碼壞檔時 PIL 會丟出的家族:截斷 JPEG / 0-byte / 格式錯 / 超大「炸彈」/ 個別損壞掃描。
# UnidentifiedImageError ⊂ OSError;DecompressionBombError 另列(⊂ Exception)。
_DECODE_ERRORS = (
    OSError, ValueError, SyntaxError,
    UnidentifiedImageError, Image.DecompressionBombError,
)


def safe_open_image(path, mode: str | None = "RGB") -> Image.Image | None:
    """開啟並『完整解碼』一張影像;成功回傳 PIL.Image,任何損壞/格式問題回 ``None``。

    PIL 是惰性解碼(``Image.open`` 不讀像素),截斷檔要到 ``.load()`` / ``.convert()``
    才爆 —— 這裡主動 ``.load()`` 把錯誤收斂在本函式,呼叫端只需判 ``None`` 即可跳過。
    刻意**不設** ``LOAD_TRUNCATED_IMAGES`` → 截斷檔誠實當壞檔跳過,而非餵半張灰圖。

    ``mode`` 給定時轉色彩模式(預設 ``"RGB"``;灰階流程傳 ``"L"``);``None`` 表不轉。
    """
    try:
        im = Image.open(path)
        im.load()                       # 強制解碼:壞檔在此爆,被下方接住
        if mode:
            im = im.convert(mode)
        return im
    except _DECODE_ERRORS:
        return None
    except Exception:                   # 防 PIL 外掛丟非預期型別;一律當壞檔跳過,不讓它逃出去崩潰
        return None


def is_readable_image(path) -> bool:
    """這張影像能否被完整解碼(不保留結果,純探測)。"""
    im = safe_open_image(path, mode=None)
    if im is None:
        return False
    im.close()
    return True


def safe_read_text(path, default: str = "") -> str:
    """讀文字檔(YOLO label / classes.txt / data.yaml)。

    壞編碼以 ``errors="ignore"`` 容忍,讀不到(缺檔/權限/IO 錯)回 ``default``。
    **永不拋例外** —— 讓上層的逐行解析照常進行,壞行自然被略過。
    """
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return default


def partition_readable(image_paths: Iterable) -> tuple[list[Path], list[Path]]:
    """把影像清單分成 ``(可讀, 壞檔)`` 兩串。壞檔 = 無法被完整解碼者。

    供工具載入資料夾時做**前置過濾**:壞檔連同其對應 label 自然被排除在後續流程外
    (下游一律以 ``image_path`` 為鍵),壞檔清單回報給 GUI 顯示「已略過 N 個壞檔」。
    代價是對每張圖做一次完整解碼;呼叫端可依資料夾+mtime 在 session 內快取結果。
    """
    good: list[Path] = []
    bad: list[Path] = []
    for p in image_paths:
        (good if is_readable_image(p) else bad).append(Path(p))
    return good, bad
