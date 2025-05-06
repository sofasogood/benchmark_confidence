"""
robust.py
Compare accuracy on original vs perturbed sets and print BCR Robustness score.
"""
import pandas as pd
import numpy as np

def rubric(delta):
    """Convert accuracy drop to BCR Robustness score (0-3)."""
    if abs(delta) <= 2:
        return 3
    if abs(delta) <= 5:
        return 2
    if abs(delta) <= 10:
        return 1
    return 0

if __name__ == '__main__':
    ORIG = "gpt4_preds.csv"
    PERT = "gpt4_paraphrase_preds.csv"

    orig = np.array(pd.read_csv(ORIG)["is_correct"].values)
    pert = np.array(pd.read_csv(PERT)["is_correct"].values)

    acc_orig = np.mean(orig)
    acc_pert = np.mean(pert)
    delta_pp = (acc_orig - acc_pert) * 100      # percentage-point drop

    print(f"Original accuracy : {acc_orig*100:.2f} %")
    print(f"Perturbed accuracy: {acc_pert*100:.2f} %")
    print(f"Drop              : {delta_pp:.2f} pp")
    print(f"BCR Robustness    : {rubric(delta_pp)} / 3")
