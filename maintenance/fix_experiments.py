import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import re

with open(_os.path.join(_ROOT, "paper/sections/experiments.tex"), "r") as f:
    content = f.read()

# Find and fix the broken sentence that has \ref{tab:favorita} reports aggregate performance.
# The pattern: the sentence ends with "operational stability.\ref{tab:favorita} reports aggregate performance."
broken = "operational stability.\\ref{tab:favorita} reports aggregate performance."
fixed = "operational stability."

if broken in content:
    content = content.replace(broken, fixed)
    print("Fixed broken sentence.")
else:
    print("Broken sentence pattern not found.")

# Now remove the duplicate old table (the one without Tuned XGBoost)
# Find the second occurrence of the Favorita table
old_table_start = r"\\begin\{table\}\[t\]\n\\centering\n\\footnotesize\n\\setlength\{\\tabcolsep\}\{2pt\}\n\\caption\{28-Day Forecast Performance and Bullwhip Mitigation on Favorita \(953 Items, global pooled kernel\)\}\n\\label\{tab:favorita\}\n\\resizebox\{\\columnwidth\}\{!\}\{%\n\\begin\{tabular\}\{lccccc\}\n\\toprule\nMethod & wMAPE & RMSE & VR \(mean\\\$pm\\std\) & VR \(median\) & Safety Stock \\\\\n\\midrule\nVanilla XGBoost & \\$1\.31"

# Find all table starts and remove the second old-style one (without Tuned XGBoost)
# Actually, let me just find the duplicate "Finding 4" paragraph and remove it
old_finding4 = r"\\textbf\{Finding 4: Cross-Dataset Consistency\.\} The pattern on Favorita mirrors M5: CD achieves lower wMAPE \(\$1\.24\$ vs\. \$1\.31\$\) and dramatically lower Variance Ratio \(median \\textbf\{0\.0077\} vs\. \$0\.38\$, Wilcoxon \$p < 10\^\{-15\}\$\), with CD winning on \$95\%\$ of items\. The mean VR \(\$0\.022"

# Let me try a simpler approach: split by the first occurrence of the figure environment
# after the new table, and remove everything duplicate between the new Finding 4 and the figure

# Find the position after "This confirms that even when gradient boosting achieves competitive accuracy on less structured calendars, the structural decomposition still yields superior operational stability."
marker = "operational stability.\n\n\\begin{figure}[t]"
if marker in content:
    print("Found correct marker.")
else:
    # Try without the extra newline
    marker = "operational stability.\n\\begin{figure}[t]"
    if marker in content:
        print("Found marker with single newline.")
    else:
        print("Marker not found, checking...")
        # Find position of first \begin{figure} after new table
        idx = content.find("Tuned XGBoost (lags 1--28, rolling stats) & $1.20")
        if idx > 0:
            # Find the figure after this
            fig_idx = content.find("\\begin{figure}[t]", idx)
            print(f"Figure at position {fig_idx}")
            print(repr(content[fig_idx-100:fig_idx+50]))

# Find the duplicate table (without Tuned XGBoost row) 
# Pattern: after "operational stability." there should be \begin{figure}
# but currently there's a duplicate table in between

# Find the text between the first and second "Finding 4"
parts = content.split("\\textbf{Finding 4: Cross-Dataset Consistency.}")
print(f"Number of 'Finding 4' occurrences: {len(parts) - 1}")

if len(parts) == 3:
    # We have two Finding 4 paragraphs - keep the first, remove the second
    # The structure is: part0 + "Finding 4" + part1 + "Finding 4" + part2
    # part1 should end with the new Finding 4 text
    # We need to find where the duplicate starts (the old table)
    
    # Find the position of the second "Finding 4" in the original
    first_idx = content.find("\\textbf{Finding 4: Cross-Dataset Consistency.}")
    second_idx = content.find("\\textbf{Finding 4: Cross-Dataset Consistency.}", first_idx + 1)
    
    if second_idx > 0:
        # Find the figure after the second Finding 4
        fig_after_second = content.find("\\begin{figure}[t]", second_idx)
        if fig_after_second > 0:
            # Remove from second_idx to fig_after_second, replace with nothing
            # But keep the figure
            new_content = content[:second_idx] + content[fig_after_second:]
            print(f"Removed duplicate content from {second_idx} to {fig_after_second}")
            
            with open(_os.path.join(_ROOT, "paper/sections/experiments.tex"), "w") as f:
                f.write(new_content)
            print("File updated.")
        else:
            print("Figure after second Finding 4 not found.")
    else:
        print("Second Finding 4 not found.")
else:
    print(f"Unexpected number of splits: {len(parts)}")
