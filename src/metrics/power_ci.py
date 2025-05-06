"""Calculate confidence intervals and power scores for model predictions."""
import numpy as np
import pandas as pd

THRESHOLDS = [(0.02, 3), (0.05, 2), (0.10, 1)]   # CI width ⇒ rubric

def bootstrap_ci(correct, n_boot=1000, alpha=0.05):
    """Calculate the bootstrap confidence interval for the mean accuracy."""
    rng = np.random.default_rng(42)
    stats = [rng.choice(correct, size=len(correct)).mean() for _ in range(n_boot)]
    lower, upper = np.percentile(stats, [100*alpha/2, 100*(1-alpha/2)])
    return lower, upper

def score_from_width(ci_width):
    """Convert CI width to BCR power score."""
    for thresh, score in THRESHOLDS:
        if ci_width <= thresh:
            return score
    return 0

if __name__ == "__main__":
    df = pd.read_csv("gpt4_preds.csv")          
    acc  = df["is_correct"].mean()
    lo, hi = bootstrap_ci(df["is_correct"].values)
    width  = hi - lo
    POWER_SCORE = score_from_width(width)

    print(f"Accuracy      : {acc*100:6.2f} %")
    print(f"95 % CI       : [{lo*100:6.2f}, {hi*100:6.2f}] %")
    print(f"CI width      : {width*100:6.2f} pp")
    print(f"BCR Power     : {POWER_SCORE} / 3")
