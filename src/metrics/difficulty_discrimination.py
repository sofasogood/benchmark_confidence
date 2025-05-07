"""
difficulty_discrimination.py
Compute % ceiling / floor items using 4 correctness vectors and map
to BCR Difficulty & Discrimination score.
"""

import pandas as pd
import numpy as np
import json

def main(files=None):
    if files is None:
        files = [
            "data/predictions/gpt4_preds.csv",          # original
            "data/predictions/gpt4_paraphrase.csv",
            "data/predictions/gpt4_noise.csv",
            "data/predictions/gpt4_shuffle.csv",
        ]

    # ------------------------------------------------ load all vectors
    rows = [np.array(pd.read_csv(f)["is_correct"].values) for f in files]
    X    = np.vstack(rows)             # shape (P, N)
    P, N = X.shape

    # ------------------------------------------------ difficulty per item
    d     = X.mean(axis=0)             # length N
    ceils = (d == 1.0).sum()
    floors= (d == 0.0).sum()
    pct   = 100 * (ceils + floors) / N

    def rubric(p):
        """Convert percentage of ceiling/floor items to BCR Difficulty & Discrimination score (0-3)."""
        if p < 5:  
            return 3
        if p < 10: 
            return 2
        if p < 20: 
            return 1
        return 0

    score = rubric(pct)
    
    return {
        "total_items": int(N),
        "ceiling_items": int(ceils),
        "floor_items": int(floors),
        "ceiling_percentage": float(100 * ceils / N),
        "floor_percentage": float(100 * floors / N),
        "total_percentage": float(pct),
        "score": int(score)
    }

if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
