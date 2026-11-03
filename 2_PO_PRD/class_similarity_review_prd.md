# PRD:類別相似度分布＋Threshold 候選＋人工評估 PR — M23

- 需求來源:[1_user_needs/class_similarity_review.md](../1_user_needs/class_similarity_review.md)
- 設計契約:[docs/class-similarity-review-implementation-plan.md](../docs/class-similarity-review-implementation-plan.md) **r3**
- 審查紀錄:[docs/class-similarity-review-design-audit.md](../docs/class-similarity-review-design-audit.md)
- 立案:2026-09-14;狀態:PRD 完成,放行實作(T0–T10)

## 三行自問

1. **誰會用**:在 Visualize 做標籤健檢的資料／製程工程師，要從分布挑出該檢查的 image／object。
2. **不做會怎樣**:只能看混淆熱圖與「鄰居多數異類」清單，無法用同一套原始 cosine Top-K 規則預覽工作量，也無法在有人工真值後誠實計算找出誤標的 P／R。
3. **成功的可觀察判準**:手算 fixture 與獨立 oracle 一致;門檻 `<` 不含相等;PR 正例=人工確認原標錯誤;來源資料不被寫入;image 與 object 真實 E2E 可完成計算→證據→選取／下載。

## 範圍(MoSCoW)

### Must
1. 擴充 `_render_label_quality`，不新增頂層工具。保留混淆熱圖、疑似標錯、相似搜尋、購物車。
2. 同一模型原始高維 embedding;同類=排除自己的 Top-K cosine 平均;異類=逐異類 Top-K 平均再取最高一類;margin=same−other。
3. Box／散點／表格／鄰居證據;threshold 預覽;人工評估模板／匯入;有效評估集才顯示 PR;操作點保存／恢復;schema 2 報告。
4. 分塊 exact、≤5,000 列;身份／crop 衝突／readback／來源變更使舊結果失效。

### Won't(本輪)
頂層新工具、灰帶入口、自動改標、Labeling schema 變更、Top-N、每類獨立門檻、ANN、自動最佳門檻、跨 split 模型泛化驗證。

## 模組分解(Tier)

| 模組 | Tier | 職責 |
|------|------|------|
| `class_similarity` | A | 分塊 exact Top-K、分數、box summary |
| `class_similarity_records` | B | 穩定 ID、analysis key、label_status、選取驗證、generation snapshot |
| `class_similarity_export` | A | CSV／JSON bytes,不寫資料集 |
| `class_similarity_threshold` | A | mask／counts／曲線／評估驗證／PR |
| `class_similarity_ui` + app 接線 | B | 面板、draft/commit、選取／購物車、E2E |

## Appetite
一輪交付 T0–T10 / AC01–AC34。工作區既有未提交變更必須保留。
