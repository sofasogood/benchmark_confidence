"""Calculate BCR Coverage score based on subject distribution entropy."""
from collections import Counter
import math
import pandas as pd
from src.utils.dataset import load_mmlu     # the same helper that parses choices

THRESHOLDS = [(0.90, 3), (0.75, 2), (0.50, 1)]

df  = load_mmlu("mmlu_test.csv")[0:1000]        # 14 079 rows
counts = Counter(df["subject"])
K = len(counts)

def entropy(counter):
    """Calculate Shannon entropy of a Counter object."""
    N = sum(counter.values())
    return -sum((c/N) * math.log2(c/N) for c in counter.values())

H      = entropy(counts)
H_max  = math.log2(K)
score_norm = H / H_max

def rubric(x):
    """Convert normalized entropy to BCR Coverage score (0-3)."""
    for thr, s in THRESHOLDS:
        if x >= thr:
            return s
    return 0

print(f"Unique constructs : {K}")
print(f"H / H_max         : {score_norm:.3f}")
print(f"BCR Coverage      : {rubric(score_norm)} / 3")
