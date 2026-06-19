# Model 清單(單一真相來源)

這個檔案是 LV 需要哪些 model 權重的**唯一清單**。人讀這份表就知道「哪個功能要哪個
model、放哪、從哪來」;程式([scripts/model_manifest.py](scripts/model_manifest.py))
讀下方的 `json` 區塊就能自動檢查 / 下載。**改這裡,setup 與錯誤訊息都會跟著改。**

權重是大型二進位檔,**不進 git**;由 `python scripts/setup_models.py` 依本檔下載。

## 單一根目錄:`models/`

所有 model 都放在 **`models/`** 底下,**每個 model 一個同名資料夾**(不再有單數
`model/`)。`LV_MODELS_DIR` 可整個搬到別處。資料夾長這樣:

```
models/
  dinov2_vits14/                   DINOv2;放 dinov2_vits14.pth
  chinese-clip-vit-base-patch16/   Chinese-CLIP(整包 HF 檔)
  inception-fid-kid/               Compare·FID/KID;放 inception-2015-12-05.pt
  inception-score/                 Compare·Inception Score;放 checkpoints/inception_v3_google-0cc3c7bd.pth
  lpips/                           Compare·LPIPS;放 checkpoints/alexnet-owt-7be5be79.pth + v0.1/alex.pth
```

## 需要哪些 model

| Model | 功能 | 層級 | 資料夾 (models/…) | 大小 |
|---|---|---|---|---|
| DINOv2 ViT-S/14 | 影像 embedding / 視覺化 / 相似搜尋 | core | `dinov2_vits14/dinov2_vits14.pth` | ~85 MB |
| Chinese-CLIP ViT-B/16 | 以文搜圖 (F7) + 影像 embedding | core | `chinese-clip-vit-base-patch16/` | ~750 MB |
| Inception (clean-fid) | Compare · FID / KID | compare | `inception-fid-kid/inception-2015-12-05.pt` | ~92 MB |
| InceptionV3 (torchvision) | Compare · Inception Score | compare | `inception-score/checkpoints/inception_v3_google-0cc3c7bd.pth` | ~104 MB |
| AlexNet backbone | Compare · LPIPS(骨幹) | compare | `lpips/checkpoints/alexnet-owt-7be5be79.pth` | ~234 MB |
| LPIPS 校正頭 (alex v0.1) | Compare · LPIPS(校正頭) | compare | `lpips/v0.1/alex.pth` | ~6 KB |

## 怎麼用

```bash
python scripts/setup_models.py                # 只裝 core(DINOv2 + Chinese-CLIP)
python scripts/setup_models.py --with-compare # 再裝 Compare 的 Inception / LPIPS
python scripts/model_manifest.py              # 檢視每個 model 到位狀態(✅/❌)
```

`model_manifest.py` 會逐項印出「✅/❌ + 功能 + 預期路徑」,缺什麼一眼看到。
App 執行期若讀不到某個 model,錯誤訊息也會明講**哪個功能讀不到哪個 model、預期路徑、修法**。

---

<!-- MANIFEST:BEGIN — 下方 json 由 scripts/model_manifest.py 解析,請保持合法 JSON(勿加註解/尾逗號) -->

```json
{
  "schema": 1,
  "models": [
    {
      "id": "dinov2_vits14",
      "title": "DINOv2 ViT-S/14",
      "features": ["影像 embedding / 視覺化 / 相似搜尋"],
      "tier": "core",
      "path": "dinov2_vits14/dinov2_vits14.pth",
      "sentinel": "dinov2_vits14/dinov2_vits14.pth",
      "acquire": "url",
      "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
      "size_mb": 85
    },
    {
      "id": "dinov2_vitb14",
      "title": "DINOv2 ViT-B/14 (768-d, 更大容量)",
      "features": ["影像 embedding / 視覺化 / 相似搜尋（比 ViT-S 分離度更高、較慢）"],
      "tier": "core",
      "path": "dinov2_vitb14/dinov2_vitb14.pth",
      "sentinel": "dinov2_vitb14/dinov2_vitb14.pth",
      "acquire": "url",
      "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
      "size_mb": 331
    },
    {
      "id": "chinese-clip-vit-base-patch16",
      "title": "Chinese-CLIP ViT-B/16",
      "features": ["以文搜圖 (F7)", "影像 embedding"],
      "tier": "core",
      "path": "chinese-clip-vit-base-patch16",
      "sentinel": "chinese-clip-vit-base-patch16/config.json",
      "acquire": "hf",
      "repo": "OFA-Sys/chinese-clip-vit-base-patch16",
      "ignore": ["*.msgpack", "*.h5", "*.onnx", "flax*", "tf_*", "clip_cn_vit-b-16.pt"],
      "size_mb": 750
    },
    {
      "id": "inception-2015-12-05",
      "title": "Inception (clean-fid)",
      "features": ["Compare · FID / KID"],
      "tier": "compare",
      "path": "inception-fid-kid/inception-2015-12-05.pt",
      "sentinel": "inception-fid-kid/inception-2015-12-05.pt",
      "acquire": "cleanfid",
      "size_mb": 92
    },
    {
      "id": "inception_v3_google",
      "title": "InceptionV3 (torchvision)",
      "features": ["Compare · Inception Score"],
      "tier": "compare",
      "path": "inception-score/checkpoints/inception_v3_google-0cc3c7bd.pth",
      "sentinel": "inception-score/checkpoints/inception_v3_google-0cc3c7bd.pth",
      "acquire": "url",
      "url": "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",
      "size_mb": 104
    },
    {
      "id": "alexnet_lpips_backbone",
      "title": "AlexNet backbone (LPIPS)",
      "features": ["Compare · LPIPS（骨幹）"],
      "tier": "compare",
      "path": "lpips/checkpoints/alexnet-owt-7be5be79.pth",
      "sentinel": "lpips/checkpoints/alexnet-owt-7be5be79.pth",
      "acquire": "url",
      "url": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
      "size_mb": 234
    },
    {
      "id": "lpips_alex_head",
      "title": "LPIPS 校正頭 (alex v0.1)",
      "features": ["Compare · LPIPS（校正頭）"],
      "tier": "compare",
      "path": "lpips/v0.1/alex.pth",
      "sentinel": "lpips/v0.1/alex.pth",
      "acquire": "lpips_pkg",
      "size_mb": 0.006
    }
  ]
}
```

<!-- MANIFEST:END -->
