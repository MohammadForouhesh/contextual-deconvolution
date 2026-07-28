import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
with open(_os.path.join(_ROOT, "paper/sections/experiments.tex"), "r") as f:
    content = f.read()

# Find the second Favorita table (without "; mean \pm std" in caption)
target = "\\caption{28-Day Forecast Performance and Bullwhip Mitigation on Favorita (953 Items, global pooled kernel)}"
idx = content.find(target)

if idx > 0:
    # Find the figure after this table
    fig = content.find("\\begin{figure}[t]", idx)
    if fig > 0:
        # Remove from idx to fig
        new_content = content[:idx] + content[fig:]
        with open(_os.path.join(_ROOT, "paper/sections/experiments.tex"), "w") as f:
            f.write(new_content)
        print(f"Removed duplicate table from {idx} to {fig}")
    else:
        print("Figure not found after second table")
else:
    print("Second table not found")
