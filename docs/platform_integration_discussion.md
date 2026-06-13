# LV 掛進 CIM 平台（像 AI4BI 一樣）— multi-agent 深度討論（先不寫程式）

> 目標：LV 維持**獨立 repo**，但能像 **AI4BI** 一樣**簡單掛進** `cim-hybrid-edge-platform`（nativeApp）。
> 本文是「怎麼做」的共識文件；有共識後才開工。

參與者：阿傑（LV 工程）、Kevin（平台架構，懂 plugin 契約）、老張（MLOps/infra）、Dr. Lin（LV/ML 重相依）、Vivian（PM）、小P（clone 後要能跑的人）。

---

## 0. 研究發現（事實基礎，全部讀過原始碼/文件）

**平台是什麼**：`cim-hybrid-edge-platform` = Electron monorepo（`apps/host-electron` + `apps/portal-react`）＋ **python-engine sidecar**（FastAPI 控制 + 一次跑一個 Streamlit 工具子程序）。

**整合模型（平台自己的 `docs/AI4BI_INTEGRATION.md` playbook）—「連結，不複製」**：
- AI4BI 在**自己的 repo** 開發，以 **git submodule** 放 `sidecar/python-engine/vendor/AI4BI`（鎖 commit，release 可重現）。
- **editable 安裝**進 engine 的 Python：`pip install -e vendor/AI4BI[llm]`。
- 平台端只加一個**很薄的 runner** `tools/bi_runner.py`（一行 `runpy.run_module("ai4bi.ui.app", run_name="__main__")`）。
- 宣告在 `plugins/bi/`：`plugin.manifest.yaml` + `modules/ai4bi/plugin.yaml`（`id: app-ai4bi`、`runner: bi`）。
- **`category: app`**（第 4 種工具類型：單一 iframe、app 自己掌管整頁）**已經為 AI4BI 做好了**，engine 不用再改。
- **更新 AI4BI = submodule `git pull`**（editable 即時生效），平台端不動。

**瘦平台原則**：frozen `engine.exe` **刻意不把 plugin 重相依（plotly/duckdb…）打進核心**；plugin 自帶相依走 **per-tool venv**（`plugin.yaml` 的 `requires:`，功能 #7，`core/tool_deps.py`）。有靜態守門 `test_packaging_guards.py` 禁止把 plugin 重相依加回核心 spec。

**LV 現況（剛盤點）**：
- Python **3.11.9 ✓**（與平台 host 直譯器一致）。
- streamlit **1.58**（平台 1.57，微差）。
- 有 `pyproject.toml` 但**只放 tooling**（pytest/black/ruff），**沒有 `[project]`、不是可安裝 package**。
- **flat `scripts/`**：`scripts/app.py` 用 `from interaction import`、`from _utils import`（同層 sibling import，無 package 命名空間）。
- **重相依**：torch 2.6 / torchvision / transformers / clean-fid / lpips / umap-learn / scikit-image / chroma-hnswlib / opencc（GB 級）。
- **大模型資產**：`models/`（dinov2_vits14.pth、chinese-clip、lpips）+ `model/inception-2015-12-05.pt`，**非 git**。AI4BI 沒這問題。

→ 結論先講：**整合骨架可以完全照抄 AI4BI**；LV 特有的麻煩有三個：**flat scripts（非 package）**、**torch 級重相依**、**大模型資產**。整場討論就圍著這三個。

---

## 1. 目標模型（鏡像 AI4BI）

**Kevin**：骨架沒有懸念，照 AI4BI 抄：
- LV 以 submodule 放 `sidecar/python-engine/vendor/LV`。
- 薄 runner `tools/lv_runner.py`。
- `plugins/<id>/`：`plugin.manifest.yaml` + `modules/<slug>/plugin.yaml`（`id: app-<slug>` → category `app` 由 `app-` 前綴推導；`runner: <id>` 由 runner-map fallback `f"{runner}_runner.py"` 對到 `tools/<id>_runner.py`）。
- app 類型、portal 的 AppPanel 都已存在 → **engine / portal 幾乎不用改**。
- 更新 = submodule `git pull`。

**爭點全在底下三件 LV 特有的事。**

---

## 2. 大決策一：LV 要不要「重新打包成 package」？

**阿傑**：AI4BI 是正規 package（`ai4bi.ui.app`），所以 runner 一行 `runpy.run_module`。LV 是 flat scripts、`from interaction import`，**沒有 package 命名空間**。兩條路：
- **A 重新打包**：加 `[project]`、把 `scripts/` 變成 `visuallatent/` 套件、所有 sibling import 改成套件相對 import、`pip install -e`。乾淨、跟 AI4BI 一致、可被 pip 安裝/測試。**但動到全部 import + 測試 import 路徑**。
- **B 薄 runner + sys.path**：runner 把 `scripts/` 塞 `sys.path`、`runpy.run_path(app.py, "__main__")`。**LV 幾乎不動**。但「editable install」對 LV 沒意義（沒 package）。

**老張**：B 有個真風險——LV 的 `models.py`/`manifest.py`/`interaction.py` 名字很**通用**。把 `scripts/` 塞**全域** `sys.path`，萬一別的 plugin 也有 `models.py`…撞名。

**Kevin**：等一下——關鍵事實：engine `_start_app` 是 **spawn 一個獨立的 streamlit 子程序**跑 `lv_runner.py`。所以 runner 的 `sys.path` 注入**只活在那個子程序內**，不會洩漏到別的 plugin 子程序（它們是各自的程序）。**程序隔離把撞名風險消掉了。**

**老張**：那 B 的撞名疑慮解除。frozen 端呢？per-tool venv 裝的是**第三方相依**（torch…）；LV **自己的程式碼在 submodule、已在磁碟上**，runner 用 sys.path import 即可——**LV 自己的碼根本不需要 pip-install**。所以 B 在 frozen 也成立。

**共識（決策一）**：**走 B（薄 runner + sys.path），LV 幾乎不動**。理由：程序隔離已避撞名；LV 本來就能獨立跑，不必為平台大改；最快掛上。**但**留一條「之後可選 A」的路（給 LV 補最小 `[project]`，讓它也能被 pip 測試/重用）——非阻塞，看時程。

---

## 3. 大決策二：torch 級重相依 → per-tool venv + wheelhouse

**Dr. Lin**：LV 的 torch/transformers/clean-fid/lpips 是 GB 級。

**Kevin**：那**絕對不能進 core `engine.exe`**（packaging guard 會擋，也違反瘦平台）。走 **per-tool venv（#7）**：`plugin.yaml` 的 `requires:` 宣告，框架在隔離 venv 裝、注入子程序 `PYTHONPATH`。

**老張**：真風險是**首次啟動 LV 很慢**——裝 torch 是分鐘級、~2GB。`_prewarm_deps_and_timeout` 有放寬 timeout 不會 500，但 UX 上第一次點 LV 會乾等。緩解：
- 出貨前**預建**該 per-tool venv，或
- **wheelhouse**（`CIM_WHEELHOUSE`，離線 `--no-index --find-links`）預放 torch 輪子。
- CPU torch（**CUDA 是平台非目標**）。

**Dr. Lin**：LV 推論 CPU 可接受（demo 全 CPU 跑）。要 GPU 是另一條線，不在這次。

**細節**：`requires:` 是 inline list。LV 已有 `requirements.txt`，兩邊別各寫一份漂移——讓 setup/runner 直接 `pip install -r vendor/LV/visuallatent/requirements.txt` 進該 venv（或 plugin.yaml `requires:` 指向它）。**單一真相：LV 的 requirements.txt。**

**共識（決策二）**：torch 等走 **per-tool venv**，相依**單一真相 = LV requirements.txt**；出貨用 **wheelhouse / 預建 venv** 把「首次很慢」變「首次秒過」；CPU only。

---

## 4. 大決策三：大模型資產（LV 特有，AI4BI 沒有）

**Dr. Lin**：LV 要 `dinov2_vits14.pth`、`chinese-clip`、`inception-2015-12-05.pt`、lpips weights——大二進位、非 git。現在 LV **寫死相對 `models/` / `model/`**。

**老張**：平台已有同類模式（`.venv-xanylabeling`、`LabelMe_Dino/.venv` 把大東西放**可寫資料夾**）。模型也該放「可寫的 model-house」，不是塞進 submodule（submodule 該瘦）。

**方案**：
- (a) 首次啟動自動下載（LV 已有 `download_chinese_clip.py` 可借）。
- (b) wheelhouse 式的 **model-house** 預放（離線/工廠內網）。
- (c) LV 的模型路徑**參數化**（環境變數覆寫，例如 `LV_MODELS_DIR`），由平台指到可寫 model-house。

**共識（決策三）**：**LV 要改的一小點 = 模型路徑參數化（環境變數覆寫）**；模型放平台可寫的 model-house，首次下載或預載；submodule 保持瘦（不含模型）。

---

## 5. 小決策：版本對齊 / 命名 / portal / MCP

**版本**：
- Python 3.11 已對齊✓。
- streamlit 1.58 vs 1.57：**per-tool venv 是隔離的，LV 可自帶 1.58，不影響平台 1.57**（隔離 venv 的好處——版本不必對齊）。dev 若共用平台 env，要確認 1.57 也能跑 LV（pin `>=1.32`，應可）。

**命名（Vivian）**：plugin id 建議 `curation`（或 `lv`）；domain `data-quality`；module `id: app-visuallatent`（`app-` 前綴 → category app）；portal 名稱「🔬 資料策展 (LV)」之類。

**portal（Kevin）**：AppPanel 已支援 app 類型，LV 自動出現一個 iframe；除非要 icon/排序，**portal 不用改**。

**MCP（小P/Kevin）**：可選，**v1 不做**。之後把 `interaction.py` 的策展函式包成 MCP server（跟 labeling/bi 一致），讓別的 plugin（如 annotation）能呼叫 LV。

---

## 6. 共識：落地配方（分階段）

**P0 — 掛上去、能跑（dev）**
1. nativeApp 加 submodule `sidecar/python-engine/vendor/LV`（鎖 commit）。
2. 新增 `tools/lv_runner.py`：把 `vendor/LV/visuallatent/scripts` 塞 sys.path → `runpy.run_path(scripts/app.py, "__main__")`。
3. 新增 `plugins/<id>/`：`plugin.manifest.yaml`（仿 bi）+ `modules/<slug>/plugin.yaml`（`id: app-visuallatent`、`runner: <id>`、`requires:` 指 LV requirements）。
4. **LV 改一小點**：模型路徑參數化（`LV_MODELS_DIR` 等環境變數覆寫；預設維持現狀，向後相容）。
5. dev 設定：`pip install -r vendor/LV/visuallatent/requirements.txt` 進平台 3.11 env（或讓 per-tool venv 處理）。
6. 測試：仿 `tests/test_app_tool_type.py` 驗 LV app 接線；status endpoint 單程序存活（防假「Output 已停止」橫幅）。

**P1 — 出貨健康（frozen）**
- per-tool venv + **wheelhouse 預載 torch**（首次秒過）。
- **model-house 預載**模型。
- packaging guard 確認 LV 重相依**不進 core**。
- frozen 端到端（`/package-build`）：起 LV、確認目標機有外部 Python 3.11。

**P2 — 深整合（可選）**
- LV 的 MCP server（策展函式對外）。
- cross-plugin：annotation 的資料一鍵送 LV 策展、LV 的決策回送 annotation。

---

## 7. 風險 / 非目標

**風險**
- 首次 torch 安裝慢 → wheelhouse / 預建 venv。
- sys.path 撞名 → **程序隔離已緩解**；runner 仍只做局部注入。
- 模型資產大 → model-house（不進 submodule）。
- streamlit 版本 → 隔離 venv 化解。

**非目標（這次不做）**
- GPU/CUDA。
- 把 LV 重相依塞進 core engine.exe。
- 跨機分發相依（Fleet #1）。
- LV 大重構成 package（B 路線免了；A 留作之後可選）。

---

## 8. 一句話總結

> **骨架完全照抄 AI4BI 的「連結不複製」playbook（submodule + 薄 runner + `plugins/<id>` + 已存在的 app 類型，engine/portal 不用改）；LV 特有的三件事各有低侵入解：flat scripts → 薄 runner 用 sys.path（靠 streamlit 子程序的程序隔離避撞名，免重打包）、torch 級重相依 → per-tool venv + wheelhouse、大模型 → 路徑參數化 + 可寫 model-house。LV 端只需改「模型路徑可被環境變數覆寫」這一小點。**
