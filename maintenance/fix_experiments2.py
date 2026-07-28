import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
with open(_os.path.join(_ROOT, "paper/sections/experiments.tex"), "r") as f:
    content = f.read()

# 1. Remove duplicate old Finding 3 text
old_finding3 = "We also evaluate operational cost via a base-stock inventory simulation (lead time = 7 days, holding cost = \\$1/unit, stockout cost = \\$10/unit). On 100 paired SKUs, total inventory cost is comparable (CD = $7.88 \\pm 11.60$ vs. Tuned XGBoost = $6.56 \\pm 9.55$, $p = 0.31$), but the $2.6\\times$ safety stock reduction means CD achieves the same service level with far less capital tied up in buffer inventory. "
if old_finding3 in content:
    content = content.replace(old_finding3, "")
    print("Removed old Finding 3 text.")

# 2. Remove duplicate "Finding 3b" line
dup_3b = "\\textbf{Finding 3b: Distributed Lag Baseline.}\n\n\\textbf{Finding 3b: Distributed Lag Baseline.}"
if dup_3b in content:
    content = content.replace(dup_3b, "\\textbf{Finding 3b: Distributed Lag Baseline.}")
    print("Removed duplicate Finding 3b.")

# 3. Remove stray table environment before Favorita figure
stray = "\\begin{table}[t]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{2pt}\n\\begin{figure}[t]"
if stray in content:
    content = content.replace(stray, "\\begin{figure}[t]")
    print("Removed stray table environment.")

with open(_os.path.join(_ROOT, "paper/sections/experiments.tex"), "w") as f:
    f.write(content)
print("Done.")
