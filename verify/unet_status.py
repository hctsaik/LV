#!/usr/bin/env python3
"""U-Net 唯讀稽核:三層同名對齊、AC↔測試雙向覆蓋缺口、總測試數。
只報客觀事實,絕不用「檔案存在」推斷「完成」。用法:python verify/unet_status.py

本 repo 為「規格疊加」模式:設計在 3_Architect_Design/、AC↔測試對應表在 4_PM_Feedback/、
驗收測試碼在 tests/(test_<module>.py 與 tests/e2e/)、實作在 scripts/。
下面同名對齊掃的是 3_設計 / 4_對應表 / scripts_實作。測試碼在 tests/,用 AC 對應表追溯。
"""
import os, re, glob, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def stem(p, prefix=""):
    n = re.sub(r"^\d+_", "", os.path.splitext(os.path.basename(p))[0])
    return n[len(prefix):] if prefix and n.startswith(prefix) else n

designs = {stem(p): p for p in glob.glob(os.path.join(ROOT, "3_Architect_Design", "*.md"))}
maps    = {stem(p, "test_"): p for p in glob.glob(os.path.join(ROOT, "4_PM_Feedback", "*.md"))}
mods = sorted(set(designs) | set(maps))

print("== 設計 / AC對應表 對齊 ==")
print("%-22s 設計 對應表" % "module")
for m in mods:
    f = lambda d: "v" if m in d else "X"
    print("%-22s  %s    %s" % (m, f(designs), f(maps)))

# 需含數字:AC1/AC10/AC-F4a;不誤抓 LV_CACHE_DIR 的 ACHE。
# `(?<!\d-)` 排除**跨模組引用**:設計裡寫「AC7(無重依賴):同 08-AC8」時,那個 AC8 是 08 號模組的,
# 不是本模組的 AC —— 沒排除的話會誤報「AC8 無對應測試」(2026-07-12 實際踩過的假缺口)。
AC = re.compile(r"(?<!\d-)\bAC[-_ ]?[A-Za-z]?\d\w*")
print("\n== AC <-> 測試 雙向覆蓋(只驗掛名,不驗行為正確)==")
for m in mods:
    d_ac = set(AC.findall(open(designs[m], encoding="utf-8").read())) if m in designs else set()
    t_ac = set(AC.findall(open(maps[m], encoding="utf-8").read())) if m in maps else set()
    miss = d_ac - t_ac; orphan = t_ac - d_ac
    if miss or orphan:
        print("  %s: 無對應測試的 AC=%s; 對應表多出的 AC=%s"
              % (m, sorted(miss) or "-", sorted(orphan) or "-"))

try:
    co = subprocess.run([sys.executable, "-m", "pytest", "--co", "-q", "tests"],
                        cwd=ROOT, capture_output=True, text=True)
    print("\n== 總測試數(pytest 收集 tests/)==", sum(1 for l in co.stdout.splitlines() if "::" in l))
except Exception as e:
    print("無法收集測試數:", e)

print("\n注意:本稽核不判斷『完成』。請用上表核對 ROADMAP 模組表是否漂移(模組數 / 測試數)。")
