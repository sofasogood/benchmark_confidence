"""Calculate BCR Coverage score based on subject distribution entropy."""
from collections import Counter
import math
import pandas as pd
import json
from src.utils.dataset import load_mmlu     # the same helper that parses choices

def main(data_path="data/raw/mmlu_test.csv", max_rows=None):
    thresholds = [(0.90, 3), (0.75, 2), (0.50, 1)]

    df = load_mmlu(data_path)
    if max_rows:
        df = df[0:max_rows]
    counts = Counter(df["subject"])
    k = len(counts)

    def entropy(counter):
        """Calculate Shannon entropy of a Counter object."""
        n = sum(counter.values())
        return -sum((c/n) * math.log2(c/n) for c in counter.values())

    h = entropy(counts)
    h_max = math.log2(k)
    score_norm = h / h_max

    def rubric(x):
        """Convert normalized entropy to BCR Coverage score (0-3)."""
        for thr, s in thresholds:
            if x >= thr:
                return s
        return 0

    score = rubric(score_norm)
    print(f"Unique constructs : {k}")
    print(f"H / H_max         : {score_norm:.3f}")
    print(f"BCR Coverage      : {score} / 3")
    
    return {
        "unique_constructs": int(k),
        "entropy": float(h),
        "max_entropy": float(h_max),
        "normalized_score": float(score_norm),
        "score": int(score)
    }

if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
