import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
with open(_os.path.join(_ROOT, "paper/sections/experiments.tex"), "r") as f:
    content = f.read()

old = "Favorita Variance Ratio distribution (97 items, paired). CD wins on $97\\%$ of items. Median: CD $= 0.006$ vs.~Vanilla XGBoost $= 0.38$ (Wilcoxon $p = 3.5 \\times 10^{-15}$). Only $3\\%$ of items have CD VR $> 0.5$, confirming the median reduction is robust despite the long tail that inflates the mean."
new_str = "Favorita Variance Ratio distribution (953 items, paired CD vs.~Tuned XGBoost). CD wins on $82\\%$ of items. Median: CD $= 0.0077$ vs.~Tuned XGBoost $= 0.012$ (Wilcoxon $p < 10^{-15}$). Only $4\\%$ of items have CD VR $> 0.1$, confirming the median reduction is robust despite the long tail that inflates the mean."

if old in content:
    content = content.replace(old, new_str)
    with open(_os.path.join(_ROOT, "paper/sections/experiments.tex"), "w") as f:
        f.write(content)
    print("Fixed Favorita caption.")
else:
    print("Old caption not found.")
    # Try to find what the actual caption says
    import re
    matches = re.findall(r'Favorita Variance Ratio distribution[^}]+', content)
    for m in matches:
        print(f"Found: {m[:80]}...")
