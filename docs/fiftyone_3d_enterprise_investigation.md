# FiftyOne 3D 視覺化與 Enterprise 功能調查 — 為什麼 3D 用不了、哪些值得自己重做

> 2026-06-13 · 多代理調查（3 視角：架構考據 / 心智模型挑戰 / 工程取捨）＋ FiftyOne 1.16 原始碼與官方文件查證
> 狀態：**調查與討論，尚未開發**（依使用者要求「先不要實際的開發，而是先討論」）。
> 範圍：回答三個問題——(1) 為什麼 3D 用不了？(2) 能不能請你開發？(3) 我有哪些 Enterprise 功能不能用、值不值得重做？

---

## TL;DR — 三個直接答案

1. **為什麼 3D 用不了？** 因為 **plotly 的 3D 散點（WebGL `scatter3d`）原生不支援 box/lasso 區域選取**——3D 場景的滑鼠拖曳被綁成「旋轉視角」。FiftyOne 在 `num_dims==3` 時只能 `warnings.warn("Interactive selection is only supported in 2D")` 並回傳一張**不連動的靜態 3D 圖**。這是**繪圖函式庫的天花板，不是 Enterprise 的鎖**。

2. **「3D 沒有 App 內 lasso 是 Enterprise 功能」這個前提，是錯的。** 我查遍 FiftyOne 官方的 OSS↔Enterprise 功能對照，**沒有任何一項**把「3D embeddings 選取」列為 Enterprise 功能。Enterprise 付費版**同樣沒有** 3D in-scene lasso，因為它沿用同一套 plotly 後端。你把一個**技術限制**誤讀成了**商業限制**，再用這個誤讀去推論「值得重做」——推論的前提就不成立。

3. **「一個功能用 Enterprise，就代表它重要有價值」——這個心智模型站不住。** FiftyOne 的真正切割線**不是「價值高低」，而是「服務誰」**：OSS = 單人分析者的完整能力（curation／evaluation／視覺化／Brain／embeddings／2D lasso 全在）；Enterprise = 多人／雲端／合規／規模的**營運基礎設施**（SSO、RBAC、版本治理、cloud media、orchestration、十億級規模、白手套支援）。對一個**單人、離線**的你，Enterprise 清單裡幾乎每一項的價值訊號都是失真的——它們服務的是「買單的企業 IT／合規部門」，不是「想把分析做好的個人」。

> **最重要的一句**：你其實已經把 FiftyOne 裡你覺得有價值的東西（2D 框選看圖）重做進了 visuallatent，而且**已經撞到同一道 3D 牆**（你的 [app.py](../scripts/app.py) 裡就寫著「3D 模式不支援框選；切回 2D 後選取仍會保留」）。那道牆跟著你過來，是因為它是 plotly 的牆，不是 FiftyOne 的商業決策。

---

## §1 為什麼 3D 用不了（原始碼層級查證）

我直接讀了 VIX venv 裡安裝的 FiftyOne 1.16 原始碼，逐條驗證你的主張：

| 你的主張 | 查證結果 | 出處 |
|---|---|---|
| `num_dims` 可設 3 | ✅ 對 | `core/plots/plotly.py` 接受 2D/3D |
| `points_field`+lasso 硬性要求 `num_dims==2` | ✅ **完全屬實** | `brain/visualization.py:79-80`：`raise ValueError("points_field is only supported when num_dims=2")` |
| 3D 沒有 App 內 lasso/選取 | ✅ 對，但**原因不是 Enterprise** | `base.py:437-438`、`plotly.py:962-963`、`matplotlib.py:622-623` 全寫「interactive point selection is only available in 2D」 |
| 3D 只能離線出靜態圖 | ✅ 對 | `plotly.py:1144-1147`：`num_dims==3` 且有 samples 時只 `warn`，回傳非連動圖 |

**真正的技術根因（三位 agent 一致）**：

- **2D 散點** 跑在 SVG/Canvas 上，plotly 提供 `lasso`／`box` 工具，會回傳「被框中的點索引」，FiftyOne（與你的 app）才能據此連動看圖。
- **3D 散點（`scatter3d`）** 跑在 WebGL `scene` 上，**plotly 根本沒有「在投影平面圈一塊區域」的選取工具**——3D 的滑鼠拖曳被綁成旋轉相機。Streamlit 的 `st.plotly_chart(on_select=...)` 在 3D 同樣拿不到框選事件。
- `points_field`（讓 lasso 高效查詢的空間索引）因此也只允許 2D。

所以這不是「FiftyOne 把 3D 選取藏進付費版」，而是「**上游繪圖函式庫到 3D 就沒有區域選取這個能力**」。你的 app 與 FiftyOne 都同樣受限，正是獨立印證。

---

## §2 我目前不能用的 Enterprise 功能有哪些（逐項對「單人離線」分類）

來源：FiftyOne 官方 [why-upgrade](https://voxel51.com/fiftyone/why-upgrade) 與 [Enterprise overview](https://docs.voxel51.com/enterprise/overview.html)。OSS 套件**沒有** `fiftyone/management` 模組（那是 Enterprise SDK）即一個旁證。

| Enterprise 獨有功能 | 對「單人、離線」相關？ | OSS／自建等價物 |
|---|---|---|
| 多人協作／RBAC／SSO（OIDC/OAuth2/SAML） | ❌ 否（單人無多人概念） | — |
| 雲端／地端／air-gapped 部署 | ❌ 否（本機跑就是 air-gapped） | 本機執行 |
| Dataset versioning + 稽核 | 🟡 弱相關 | git／parquet 快照＋內容雜湊 |
| Cloud-backed media（不需複製媒體） | ❌ 否 | 直接讀本地檔 |
| Delegated operations／orchestration／scheduler | ❌ 多半否 | 單機本地 job queue |
| Management SDK | ❌ 否（管理對象是團隊資源） | — |
| Query performance／索引優化 | 🟡 視資料規模 | faiss／hnswlib（你的 app 已用 hnswlib） |
| Enterprise plugins／白手套支援 | ❌ 否 | — |
| 十億級樣本規模、雲廠商整合 | ❌ 否（單人資料量遠低於此） | — |
| Auto-labeling／quality scoring（foundation models） | ✅ **唯一真正可惜處** | OSS 可自接模型；自建 app 可自行整合 |
| **Curation／Evaluation／Visualization／Brain（embeddings、2D lasso）** | — | ✅ **全部 OSS 已有，無需 Enterprise** |

**結論**：Enterprise 清單裡，對單人離線者真正「有價值但用不到」的，幾乎只有 **auto-labeling／quality scoring 的現成服務化**——而連它的底層能力 OSS 都能自行拼裝（只是要花工）。其餘全是「組織治理／規模／合規」，與你的情境零相關。

---

## §3 心智模型修正：「Enterprise = 有價值」站不站得住

這是這次調查最重要的部分，因為它會反覆影響你的決策。

**判斷準則只有一條**：這個 paywall 是因為「**功能難做／稀缺**」，還是因為「**易於計費、且只在多人/組織情境才有意義**」？

- 前者（獨家演算法、難複製的模型）→ paywall 確實是價值訊號。
- 後者（SSO、RBAC、版本治理、雲端規模）→ paywall 訊號的是「**客戶是誰**」，而非「東西多好」。

功能被放進 Enterprise tier 的常見真實原因，以及它與「對單一使用者有價值」的相關性：

| 放進 Enterprise 的原因 | 與「對單人有價值」的相關性 |
|---|---|
| 協作才有意義（多人、共享、權限） | 零 |
| 合規剛需（SSO、稽核、RBAC） | 零（是企業 IT／法務在買單） |
| 規模成本（雲端媒體、十億級、orchestration） | 零（廠商替你扛基礎設施） |
| 支援成本（白手套、SLA） | 零（賣的是「人」不是功能） |
| 價格歧視（企業掏得出錢） | 零 |

五個原因，跟「功能本身對單人有價值」相關的**接近零個**。Enterprise tier 的設計目標，本來就是抓「為合規與規模付錢的買家」，不是「想做好分析的個人」。

**這個心智模型如何傷害你**：把「別人鎖起來」當「我需要」，會讓你把工程資源投向**模仿一個不是為你設計的產品邊界**——去重做治理、版本、權限這類單人用不到的殼，而不是去解你真正卡住的分析問題。而且這跟你在 [瑕疵問題重定義](defect_problem_redefinition.md) 裡「用 Enterprise 的存在替代自己判斷需求」是**同一個根**：讓賣方的 SKU 結構替你思考「我需要什麼」。

**該問的不是「別人收費的是哪一步」，而是「我這個工作流此刻卡在哪一步」。**

**誠實的反例（別把話說死）**：Enterprise 裡確實可能藏著對單人也有用、只是被搭售的東西——例如自動版本快照／媒體去重的底層機制，或當「十億級」不只是規模、而是讓某分析在你的資料量上「跑得動 vs 跑不動」時，它就跨過質變線變成真價值。準則是：**逐項剝掉「多人/合規/規模」的外殼，問剩下什麼對單人還有用**；剩很多 → 值得自建，剩空殼 → 那是別人的客戶的需求。

---

## §4 能不能「重新做一版」？務實工程取捨

使用者說「先不要實際的開發，而是先討論」，所以這裡只談**值不值得、難在哪、價值多少**。

### 4.1 3D 內選取：四條技術途徑的取捨

| 途徑 | 工程難度 | 真實使用價值 | 取捨 |
|---|---|---|---|
| 3D 僅 click 單選 | 極低（plotly 原生） | 低：一次一點、無法群選 | 可順手保留，解決不了核心需求 |
| **2D 投影選、3D 看（雙視圖聯動）** | 低-中 | **高**：框選在 2D、gestalt 看 3D | ✅ **唯一划算解** |
| 旋轉到某視角→投影成 2D 框選 | 中-高（自管相機矩陣＋投影） | 中：前後遮擋，框到的是視覺重疊非空間鄰近 | 遮擋問題本質無解，不划算 |
| 自建 WebGL 選取錐體（frustum 投回 3D） | 高（離開 plotly，自寫 three.js 拾取＋橋接） | 中-高（最完整） | 對單人離線者**嚴重過度工程**，維護成本吞掉收益 |

### 4.2 「3D 選取」其實多半是偽需求

降維到 2D 的整個目的，就是把高維結構壓到「能用平面互動操作」的維度——**2D 是為了操作，第三維只買到一點額外的整體形狀感**。而在 3D 投影下精確選取還會被前後遮擋污染，反而比 2D 差。「我要在 3D 裡框選」通常是把「我想要更好的整體感」誤譯成了選取需求。正解是**分工：3D 負責看、2D 負責選**。

### 4.3 真正值得自建的 Enterprise 等價物（單人離線排序）

| 功能 | 價值 | 工程成本 | 判定 |
|---|---|---|---|
| Query 索引／效能（大集合近鄰、過濾加速） | 高（決定 10 萬+ 點還能不能互動） | 中（你的 app 已用 hnswlib，可延伸） | 該做（最優先，視資料規模） |
| Dataset versioning（快照／diff 篩選結果） | 中-高（單人也常需「回到上週的篩選」） | 低-中（parquet 快照＋內容雜湊，你已有 manifest sha256） | 該做（輕量版） |
| Delegated／背景運算 | 中（嵌入／降維跑久時有用） | 中（單機只需本地 job queue） | 可做，用最小方案 |
| 多人／雲端／SSO／RBAC | 0 | 高 | ❌ 不該做 |

排序：**Query 效能 > 輕量 versioning > 本地背景運算 > 其餘忽略**。

---

## §5 結論與建議的下一步（討論層級，未開發）

1. **3D 框選不要做**「真正的 3D 內 lasso」——它是 plotly 天花板、且多半是偽需求、Enterprise 也沒有。
2. **若你真想要 3D 的價值**，唯一划算解是「**2D 選、3D 看**」聯動——而你的 app 其實已經有 90% 的零件（2D 框選、selection state 跨 rerun 保存、右欄看圖）。把 3D 圖接成「唯讀的 gestalt 視圖 + 高亮目前 2D 選取」即可，零新技術風險。
3. **不要再用「Enterprise 有沒有」當需求清單**。改用「我的工作流卡在哪」當清單。
4. **若要投入工程**，CP 值排序是：query 效能（視規模）→ 輕量 versioning → 本地背景運算。3D 選取墊底。

> 你問「我能請你開發相關的功能就好嗎？」——能。但這次調查的誠實建議是：**先別開發 3D lasso**（投錯方向），如果你要，我可以做的是「2D 選 / 3D 看」這條低成本聯動，或上面 CP 值更高的 query/versioning。等你決定方向，再進開發。

---

## §6 誠實邊界（這份調查可能錯在哪）

1. **FiftyOne 版本演進快**：OSS↔Enterprise 界線會移動（Teams→Enterprise 改名、部分功能曾下放）。本文以你環境裡的 **1.16 原始碼**與當前官網為準，未複查每個更新版。
2. **plotly 的 3D 限制若未來改版新增區域選取**，§1 的「技術天花板」結論會鬆動；以目前原始碼成立。
3. **query 效能的痛點取決於你真實的資料規模**——若只有幾千點，plotly/pandas 直接夠用，自建 ANN 是過度準備。這是我沒有的資訊。
4. **「auto-labeling 底層 OSS 可自建」是能力判斷，不代表工程成本低**，未量化。
5. 「2D 選、3D 看」聯動在 Streamlit rerun 模型下的狀態同步，可能比「低成本」更黏手一點（兩視圖 index 對齊、選取持久化）。

---

## 附錄：多代理調查歷程與來源

| 視角 | 結論 |
|---|---|
| 架構考據者 | 「3D=Enterprise」是把技術限制誤讀成商業分層；逐項分類 Enterprise 功能對單人離線多半零相關；切割線是「服務個人 vs 服務組織」。 |
| 心智模型挑戰者 | paywall 是「客戶是誰」的訊號、非「東西多好」；五個分層原因與單人價值相關性接近零；與「用 Enterprise 存在替代自己判斷」同根。 |
| 工程取捨者 | 真 3D 選取不值得做；唯一划算是「2D 選 3D 看」；3D 選取多為偽需求；自建排序 query 效能 > 輕量 versioning > 背景運算。 |

**來源（已查證）**：
- FiftyOne 原始碼（1.16）：`fiftyone/brain/visualization.py:79-80`、`fiftyone/core/plots/plotly.py:1144-1147`、`fiftyone/core/plots/base.py:437-438`、`matplotlib.py:622-623`；OSS 無 `fiftyone/management` 模組。
- [When FiftyOne Enterprise is the right choice (why-upgrade)](https://voxel51.com/fiftyone/why-upgrade)
- [FiftyOne Enterprise Overview — docs](https://docs.voxel51.com/enterprise/overview.html)
- [Interactive Plots — FiftyOne docs](https://docs.voxel51.com/user_guide/plots.html)
- [Announcing FiftyOne 0.19: In-App Embeddings Visualization](https://voxel51.com/blog/announcing-fiftyone-0-19)
