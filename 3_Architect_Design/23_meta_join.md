# 23 — meta_join 技術設計(metadata CSV ↔ 影像對照 + 分組索引)

對應 PRD:[2_PO_PRD/dataset_audit_prd.md](../2_PO_PRD/dataset_audit_prd.md) · 里程碑 M20 · **Tier B(僅 CSV 讀檔一點 I/O,其餘純函式)**

## 1. 目的

把使用者的「檔名 ↔ 機台/配方/批號」CSV 掛到影像清單上:雙鍵匹配(sha256 優先、
檔名次之)、**匹配失敗明講不默略**(雙向:哪些 CSV 列沒配到、哪些影像沒資料)、
並提供分組索引給報告切片。這是 08 文件 P0-A「最小 metadata」的落地,
manifest sha256 是天然掛點。

## 2. I/O 契約

### `scripts/meta_join.py`(依賴:csv/stdlib;禁 import 其他專案模組)

```python
def load_metadata_csv(csv_path) -> list[dict]
# utf-8-sig 讀入(容 BOM)。空檔或無表頭 → ValueError(訊息含路徑)。
# 回 DictReader 列(值皆 str,strip 過;全空列跳過)。

def join_metadata(rows, paths, sha_of=None, *,
                  filename_col=None, sha_col=None) -> dict
# rows: list[dict];paths: list[str](影像絕對路徑,list 索引=影像 idx);
# sha_of: dict[int, str] | None(idx → sha256)。
# 欄名偵測(不分大小寫、strip):
#   sha_col   未指定 → 在 {"sha256","sha"} 找;
#   filename_col 未指定 → 在 {"filename","file","image","image_name","path","filepath","檔名"} 找;
#   兩者皆偵測不到 → ValueError(訊息列出 CSV 實際欄位)。
# 匹配優先序:① sha256 完全比對(有 sha_col 且 sha_of 給定時)② 檔名 basename
#   不分大小寫比對(CSV 值取 basename,容其含路徑)。sha 配到就不再用檔名。
# 一列配多張影像(同 basename 多路徑)→ 全部套用;
# 多列配同一影像 → 後列覆蓋前列,並在 warnings 記一句。
# 回:{"fields": [除鍵欄外的欄名,保持 CSV 順序],
#     "per_image": {idx: {field: value}},
#     "n_rows": int, "n_matched_images": int, "n_unmatched_images": int,
#     "unmatched_row_idx": [int](0-based,配不到任何影像的列),
#     "warnings": [str]}

def group_counts(per_image, n_images, field, *, max_groups=20) -> dict
# 依欄位值分組:{"groups": {value: [idx...]}(依組大小降冪;值超過 max_groups 種時
#   只留前 max_groups-1 組,其餘合併鍵 "(其他)"), "missing": [idx...](無該欄位值
#   的影像,含未匹配者)}。組內索引遞增。
```

## 3. 邊界條件

| 情況 | 行為 |
|---|---|
| CSV 空 / 無表頭 | ValueError |
| 鍵欄偵測不到 | ValueError(列出實際欄位) |
| sha 與檔名同時可比但指向不同影像 | sha 勝(內容身分優先) |
| CSV 檔名含路徑(`a/b/img.png`) | 取 basename 比對 |
| 值全空的列 | 跳過,不入 n_rows |
| max_groups=1 | 全部進 "(其他)" 之前至少留 0 組 → 定義:留 max_groups-1=0 組,全進 "(其他)" |

## 4. Acceptance Criteria(`tests/test_meta_join.py`;tmp CSV + 合成路徑清單)

- **AC1(檔名匹配手算)**:4 圖(其中 2 張同 basename 不同資料夾)+ 3 列 CSV
  (1 列命中該同名 basename)→ 該列套到 2 張;`n_matched_images==3`?
  ——fixture 釘死:paths=[a/x.png, b/x.png, c/y.png, d/z.png],rows 檔名=[x.png, y.png]
  → per_image 覆蓋 idx{0,1,2},n_matched_images==3,n_unmatched_images==1(z)。
- **AC2(sha 優先)**:CSV 檔名亂寫但 sha256 正確 → 仍配上;同列 sha 指 idx0、
  檔名指 idx1 → 只套 idx0。
- **AC3(未匹配雙向明講)**:5 列 bogus 檔名 → `unmatched_row_idx` 長度 5;
  2 張影像無任何列 → `n_unmatched_images==2`。
- **AC4(欄名偵測)**:表頭 `FileName,Tool`(大小寫混)可用;表頭 `SHA256,Tool` 配
  sha_of 可用;表頭 `foo,bar` → ValueError 訊息含 "foo"。
- **AC5(group_counts 上限)**:25 種 tool 值、max_groups=20 → 19 組 + "(其他)";
  "(其他)" 內索引 = 其餘 6 組聯集;`missing` 含未匹配影像。
- **AC6(BOM/空檔)**:utf-8-sig 寫入可讀;空檔 ValueError。
- **AC7(多列同圖後勝)**:兩列同檔名不同 Tool → per_image 取後列值,warnings 非空。

## 5. 給 /pm 的提示

- AC1 的 n_matched_images 數的是**影像**(不是列);同名 2 張都算配到。
- 本模組不碰影像檔案系統(paths 只是字串),CSV 是唯一 I/O。
