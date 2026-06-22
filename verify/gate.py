#!/usr/bin/env python3
"""U-Net 綠燈閘門(語言中立):客觀判定 module 驗收是否全綠 + 防契約竄改。
測試命令由 .unet/gate.json 設定(缺檔則用內建 Python+pytest 預設)。用法:
  python verify/gate.py --snapshot      # 記錄 3_/4_ 契約檔雜湊(PG 動工前)
  python verify/gate.py <module>        # 跑該 module 驗收;只有全綠才印 GREEN
GREEN 條件:收集到的測試數>0(若有設 collect_cmd)、測試命令 exit==0、3_/4_ 契約檔未改動。
換語言:只改 .unet/gate.json 的 test_path / test_cmd / collect_cmd / collect_mark。
"""
import sys, os, subprocess, hashlib, json, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT_DIRS = ["3_Architect_Design", "4_PM_Feedback"]
SNAP = os.path.join(ROOT, ".unet", "contract_hashes.json")
CFG = os.path.join(ROOT, ".unet", "gate.json")

DEFAULT_CFG = {
    "test_path": "4_PM_Feedback/test_{module}.py",
    "test_cmd": '"{py}" -m pytest "{test}" -p no:cacheprovider --strict-markers -q',
    "collect_cmd": '"{py}" -m pytest --co -q "{test}"',
    "collect_mark": "::",
}

def load_cfg():
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CFG):
        try:
            cfg.update(json.load(open(CFG, encoding="utf-8")))
        except Exception as e:
            print("WARN: 讀 .unet/gate.json 失敗,改用預設:", e)
    return cfg

def contract_hashes():
    h = {}
    for d in CONTRACT_DIRS:
        for p in glob.glob(os.path.join(ROOT, d, "**", "*"), recursive=True):
            if os.path.isfile(p) and "__pycache__" not in p:
                with open(p, "rb") as f:
                    rel = os.path.relpath(p, ROOT).replace("\\", "/")
                    h[rel] = hashlib.sha256(f.read()).hexdigest()
    return h

def snapshot():
    os.makedirs(os.path.dirname(SNAP), exist_ok=True)
    with open(SNAP, "w", encoding="utf-8") as f:
        json.dump(contract_hashes(), f, ensure_ascii=False, indent=2)
    print("snapshot:", len(contract_hashes()), "個契約檔已記錄")

def tampered():
    if not os.path.exists(SNAP):
        return None
    old = json.load(open(SNAP, encoding="utf-8"))
    now = contract_hashes()
    return [k for k in set(old) | set(now) if old.get(k) != now.get(k)]

def run(mod):
    cfg = load_cfg()
    test = cfg["test_path"].format(module=mod)
    if not os.path.exists(os.path.join(ROOT, test)):
        print("RED: 找不到", test); return 1
    ctx = {"py": sys.executable, "test": test, "module": mod}
    n = None
    cc = cfg.get("collect_cmd")
    if cc:
        out = subprocess.run(cc.format(**ctx), shell=True, cwd=ROOT,
                             capture_output=True, text=True)
        mark = cfg.get("collect_mark", "::")
        n = sum(1 for ln in out.stdout.splitlines() if mark in ln)
        if n == 0:
            print("RED: 收集到 0 個測試(collection error 或空檔)\n", out.stdout, out.stderr)
            return 1
    r = subprocess.run(cfg["test_cmd"].format(**ctx), shell=True, cwd=ROOT)
    if r.returncode != 0:
        print("RED: 測試命令 exit =", r.returncode); return 1
    ch = tampered()
    if ch:
        print("RED: 偵測到契約檔被改動(任何角色不得改 3_/4_):")
        for c in ch: print("  -", c)
        return 1
    if ch is None:
        print("(提醒:無 baseline,未檢查契約竄改;請先跑 --snapshot)")
    print("GREEN: 測試全過%s,契約未被竄改" % ("" if n is None else "(%d 個)" % n))
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    if sys.argv[1] == "--snapshot":
        snapshot(); sys.exit(0)
    sys.exit(run(sys.argv[1]))
