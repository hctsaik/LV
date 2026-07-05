# AC ↔ 測試對應表:M14 以樣搜樣特徵器解耦(B1 GUI)

設計:[3_Architect_Design/M14_extractor_decouple.md](../3_Architect_Design/M14_extractor_decouple.md)。
E2E:`tests/e2e/test_fewshot_decouple_e2e.py`(+ `tests/e2e/test_fewshot_gui_e2e.py::test_f3_no_model_usable` 改契約)。

| AC | 說明 | 測試 |
|----|------|------|
| AC-D1 | 乾淨 session、從未建 anomaly 模型 → 以樣搜樣①可用、預設特徵器建成樣本集(死路拆掉) | `test_ac_d1_d2_no_model_full_flow` |
| AC-D2 | 解耦後海掃佇列仍含建議類別+相似度(相似物件排前)+ 來源零寫入(C6) | 同上 |
| AC-D3(可選) | 換 object_source 重掃不吃 stale shard | **由單元 AC-RL-4 覆蓋**(引擎層確定性;E2E 成本高故標可選) |
| — | 舊「無模型→導引先建模」契約已由解耦取代 | `test_f3_no_model_usable`(原 `test_f3_no_model_graceful` 改寫) |

> 後端核心(retrieve 免 bank、model_version 綁特徵器身分)由單元 AC-RL/AC-OS/AC-SVC-DEC 覆蓋;
> E2E 只驗 GUI 旗艦真實行為(無模型全程可用 + 檢索正確 + C6),不進 PG 自主修綠迴圈,交 /ux-test 或人觸發。
