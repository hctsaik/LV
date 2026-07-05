# 設計:13 `sample_bank`(few-shot 樣本集:build/save/load/append + 模型綁定)

> 對應 PRD:[2_PO_PRD/fewshot_search_prd.md](../2_PO_PRD/fewshot_search_prd.md)(M13)。Tier B(檔案 I/O)。
> **只出契約 + AC,不寫實作。** 復用 `anomaly_tool._object_embeddings`、`anomaly_bank_store._atomic_*`。

## 1. 目的
把使用者的少量樣本(4 類×5~10 張的 YOLO 物件)轉成帶 provenance 的 few-shot 樣本集,持久化到
`.lv_cache`,供海掃 multi_ref_similarity 比對。**綁定凍結模型**(embedding 空間一致才可比)。
> 與 anomaly bank 的差異:anomaly bank 是 patch coreset、**無 provenance**;sample_bank 從第一天帶
> provenance(image_path/bbox/label),且存**物件級** embedding(非 patch)。

## 2. I/O 契約(`scripts/sample_bank.py`)
資料結構 `bank`:`{"vectors": (N,D) float32, "labels": (N,) str, "provenance": [{"image_path","bbox":[cx,cy,w,h],"label"}], "model": str, "target_res": int}`。

```python
def build_sample_bank(sample_dirs, *, model, target_res, class_names=None,
                      object_source="yolo", embed_fn=None, extractor=None) -> dict:
    """樣本資料夾 → discover objects(yolo/whole_image)→ _object_embeddings → bank dict。
    無物件 → ValueError。embed_fn 可注入(測試)。"""

def save_sample_bank(bank_dir, bank) -> None:
    """emb.npz(vectors+labels)+ meta.json(model/target_res/provenance/schema_version)。原子寫。"""

def load_sample_bank(bank_dir) -> dict:
    """回上述 bank dict。缺 meta.json → FileNotFoundError/ValueError(訊息含 meta)。"""

def append_sample(bank_dir, *, vectors, labels, provenance) -> dict:
    """把新物件接到既有樣本集尾端(N→N+k),原子寫,回更新後 bank。維度/長度不符 → ValueError。"""

def assert_model_compatible(bank, *, model, target_res) -> None:
    """bank['model']!=model 或 bank['target_res']!=target_res → ValueError(不可跨 embedding 空間比對)。"""
```

## 3. 資料流
build:`sample_dirs → discover_yolo_objects/discover_whole_images → meta → _object_embeddings(meta,model,embed_fn) → vectors`;
provenance 逐物件記 `image_path/bbox/label`。save/load 走 emb.npz + meta.json;append 讀舊 → concat → 原子覆寫。

## 4. 邊界與錯誤
| 情境 | 行為 |
|------|------|
| build 找不到物件(無 labels/、空) | `ValueError`(訊息含「物件」或「樣本」) |
| load 缺 meta.json | raise(訊息含 meta) |
| append 的 vectors 維度 ≠ 既有 D,或 len(labels)≠len(vectors) | `ValueError` |
| assert_model_compatible 模型/解析度不符 | `ValueError`(訊息含模型名或解析度) |
| 空樣本集 save(N=0) | 允許(回空 bank);但 build 不產空(無物件即 raise) |

## 5. Acceptance Criteria(釘死;給 `/pm`,tests/test_sample_bank.py)
> 持久化/append/相容(AC-B1~B5)用**合成向量**(不觸真模型);build(AC-B6~B7)用小 YOLO 資料夾 + **注入 embed_fn**。

- **AC-B1(save→load 往返)**:save `{vectors:(3,4) 已知, labels:["a","a","b"], provenance:[3 筆], model:"m", target_res:224}`
  → load:vectors allclose(<1e-6)、labels 逐一相等、provenance 逐欄相等、model=="m"、target_res==224。
- **AC-B2(append)**:3 筆 bank → `append_sample(+2 筆)` → load 後 N==5;第 4/5 筆 labels/provenance == 新增值(接尾)。
- **AC-B3(模型相容)**:`assert_model_compatible(bank, model="m", target_res=224)` 不 raise;
  `model="other"` → raise;`target_res=336` → raise。
- **AC-B4(load 缺檔)**:空目錄 load → raise(含 meta)。
- **AC-B5(append 維度不符)**:對 D=4 的 bank append D=5 向量 → ValueError。
- **AC-B6(build 注入 embed_fn)**:小 YOLO 資料夾(2 物件,label scratch/stain)+ 假 embed_fn(回固定維向量)
  → build 回 vectors.shape==(2, D)、labels==["scratch"?...](依 YOLO)、provenance 2 筆含 image_path/bbox/label、
  model/target_res 帶入。
- **AC-B7(build 無物件)**:無 labels/ 的資料夾 → `ValueError`。
- **AC-B8(衍生:往返後可直接餵 multi_ref)**:load 出的 vectors+labels 丟 `similarity.multi_ref_similarity(query, vectors, labels)`
  不 raise、回形狀正確(證明樣本集與檢索介面契約相容)。
