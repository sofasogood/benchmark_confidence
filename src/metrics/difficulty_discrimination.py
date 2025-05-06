"""
difficulty_discrimination.py
Compute % ceiling / floor items using 4 correctness vectors and map
to BCR Difficulty & Discrimination score.
"""

import pandas as pd
import numpy as np

FILES = [
    "gpt4_preds.csv",          # original
    "gpt4_paraphrase.csv",
    "gpt4_noise.csv",
    "gpt4_shuffle.csv",
]

# ------------------------------------------------ load all vectors
rows = [np.array(pd.read_csv(f)["is_correct"].values) for f in FILES]
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

print(f"Items        : {N}")
print(f"Ceiling      : {ceils}   ({100*ceils/N:.2f} %)")
print(f"Floor        : {floors}  ({100*floors/N:.2f} %)")
print(f"Ceil+Floor % : {pct:.2f} %")
print(f"BCR D&D score: {rubric(pct)} / 3")
