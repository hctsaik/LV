# PRD:AL Loop Infrastructure(把主動學習串成迴圈)

對應需求:[1_user_needs/al_loop_infrastructure.md](../1_user_needs/al_loop_infrastructure.md)
日期:2026-07-04 · 狀態:**規格已備、未放行開發**(User 明示先寫到弱 AI 可接手,不開工)

## 1. 問題(一句話)

LV 的選樣訊號很富,但迴圈沒有「記憶」與「模型參與」:重複送標無防呆、回合歷史蒸發、
預測/熱力圖不能當預標、選樣不能混合策略、回讀後無便宜的效益估計。

## 2. 模組分解(feature → module)

| # | Module | Tier | MoSCoW | 一句話 | 相依 |
|---|--------|------|--------|--------|------|
| 08 | `pool_registry` | A(核心)→B(接線) | **Must** | sha256 全域樣本狀態機,永不重複送標 | 無 |
| 09 | `round_ledger` | A(核心)→B(接線) | **Must** | 回合帳本 + 回合時間軸 UI | 08(共用身分鍵)、apply_readback(已有) |
| 10 | `gt_pred_diff` | B | **Must** | GT×預測比對(FN/FP/混淆/低conf 佇列)+ 預測框預標 seed | parse_yolo_boxes_conf(已有) |
| 13 | `heatmap_to_boxes` | A(核心)→B(按鈕) | **Should** | anomaly 熱力圖→YOLO 6 欄預標框,餵進 10 的管線 | patch score map(已有)、10 |
| 11 | `hybrid_sampler` | A(核心)→B(接線) | **Should** | 不確定×多樣性×類配額混合選樣 | farthest_point_sampling(已有) |
| 12 | `probe_eval` | B | **Could** | frozen-embedding linear probe 回合代理評估 + 報酬遞減視圖 | 09、dino_head 模式(已有) |

**Won't(本 feature 不做,再痛再起輪)**:gold 樣本品質哨兵、漂移哨兵排程、多標註者指派/仲裁、
LV 內訓練重模型、自動停止(只給建議)。

## 3. 建議里程碑切法(appetite)

- **M9(地基)= 08 + 09**:兩個純邏輯核心都小,一輪做完。沒有它們,其餘模組沒地方記帳。
- **M10(模型參與)= 10 + 13**:共用「預測資料夾(YOLO 6 欄 txt)」這一個輸入介面;13 把 anomaly
  熱力圖轉成同介面,一次設計兩個來源。
- **M11(聰明化)= 11 + 12**:疊在地基上,各自獨立可再拆。

每輪照 U-Net 五階段;Tier A 核心走極簡設計+完整測試,GUI 接線一律 Tier B(單元綠 AND 真實 E2E 綠)。

## 4. 成功門檻(feature 級)

- 同一張圖(同 sha256)第二次被任何工具送標時,UI 明確攔截並顯示先前批次;預設不重送。
- 跑完一輪送標→回讀,回合時間軸出現一張含「策略/送出數/回讀數/變更數」的回合卡,重開 app 仍在。
- 給一對 GT+預測資料夾,FN 佇列裡的圖用肉眼驗證確實缺框;預標 seed 後,Labeling 端打開能看到預測框。
- anomaly patch 模式跑完,一鍵產出的預標 txt 能被 `parse_yolo_boxes_conf` 讀回且框住合成瑕疵。
- 混合選樣在造好的「叢聚不確定」fixture 上,近重複對數 < 純 top-K 的一半。

## 5. 全域鐵則(每個模組設計都要遵守)

1. **絕不寫使用者資料集**:pool/round 記錄進 `.lv_cache/`(池)與 `output/`(回合);
   熱力圖預標 txt 進 `.lv_cache/`。
2. **不靜默決策**:攔截/建議/預標一律附理由與數量;預設行為保守但可被使用者覆寫。
3. **身分鍵 = 影像 sha256**(與 handoff/export 既有身分一致);路徑只是 hint。
4. **設計文件**:`3_Architect_Design/08…13_*.md`,AC 編號沿用 `# ACn` 慣例,PM 落
   `tests/test_<module>.py` + E2E 落 `tests/e2e/test_<module>_e2e.py`。
