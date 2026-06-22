#!/usr/bin/env python3
"""U-Net 角色感知硬隔離:PreToolUse(Write|Edit)hook。
讀 stdin 的工具呼叫 JSON 與 .unet/role(目前角色),擋下兩類越界(exit 2):
  (1) 任何 active 角色去寫「基礎設施檔」(閘門/守門/稽核腳本、settings、.unet);
  (2) 某角色去寫「不屬於它的層」。
fail-open:無 role 檔 / 解析失敗 / 非受管路徑一律放行(exit 0)——所以無角色的一般模式可自由維護基礎設施。

本 repo 為「規格疊加」模式,層擁有權對應:
  1_user_needs→user  2_PO_PRD→po  3_Architect_Design→architect  4_PM_Feedback→pm
  scripts/(實作)→pg   tests/(驗收測試+測試基礎設施)→pm   ROADMAP.md→po
注意:只擋「用 Write/Edit 直接寫檔」,擋不住 Bash 寫檔或語義繞過,後者靠 gate.py 的 hash 自查 + 人審。
要切回無限制的一般維護模式:刪掉 .unet/role(rm .unet/role)。
"""
import sys, os, json

def allow():
    sys.exit(0)

def deny(msg):
    sys.stderr.write(msg + "\n")
    sys.exit(2)

try:
    data = json.load(sys.stdin)
except Exception:
    allow()

if data.get("tool_name") not in ("Write", "Edit"):
    allow()

fp = (data.get("tool_input") or {}).get("file_path", "")
if not fp:
    allow()

root = data.get("cwd") or os.getcwd()
role_file = os.path.join(root, ".unet", "role")
if not os.path.exists(role_file):
    allow()                       # 無 active 角色 → 不限制(設定/維護模式)
role = open(role_file, encoding="utf-8").read().strip()
if not role:
    allow()

try:
    rel = os.path.relpath(os.path.abspath(fp), os.path.abspath(root)).replace("\\", "/")
except Exception:
    allow()

# (1) 基礎設施:任何角色都不可改(請切回無角色模式維護)
INFRA = {"verify/gate.py", "verify/role_guard.py", "verify/unet_status.py",
         ".claude/settings.json"}
if rel in INFRA or rel.startswith(".unet/"):
    deny("U-Net 阻擋:[%s] 不可改基礎設施 %s。請先 `rm .unet/role`(切回無角色維護模式)再改。" % (role, rel))

# (2) 層擁有權(規格疊加模式:scripts/=實作=pg;tests/=驗收+測試基礎設施=pm)
LAYER = {
    "1_user_needs": "user",
    "2_PO_PRD": "po",
    "3_Architect_Design": "architect",
    "4_PM_Feedback": "pm",
    "scripts": "pg",
    "tests": "pm",
}
owner = None
for prefix, r in LAYER.items():
    if rel == prefix or rel.startswith(prefix + "/"):
        owner = r
        break
if owner is None:
    if rel == "ROADMAP.md":
        owner = "po"
    elif rel == "conftest.py" or rel.startswith("fixtures/"):
        owner = "pm"

if owner is None:
    allow()                       # 非受管路徑 → 放行

if owner != role:
    deny("U-Net 角色越界:目前角色 [%s] 不可寫入 %s(屬於 [%s] 的領域)。\n"
         "→ 若是契約有誤,走反向閘門停手回報;否則切換到對的角色再動。" % (role, rel, owner))
allow()
