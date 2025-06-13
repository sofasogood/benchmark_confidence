"""
robust.py
Compare accuracy on original vs perturbed sets and print BCR Robustness score.
"""
import pandas as pd
import numpy as np
import json

def rubric(delta):
    """Convert accuracy drop to BCR Robustness score (0-3)."""
    if abs(delta) <= 2:
        return 3
    if abs(delta) <= 5:
        return 2
    if abs(delta) <= 10:
        return 1
    return 0

def main(orig_path="data/predictions/gpt4_preds.csv", pert_path="data/predictions/gpt4_paraphrase.csv"):
    orig = np.array(pd.read_csv(orig_path)["is_correct"].values)
    pert = np.array(pd.read_csv(pert_path)["is_correct"].values)

    acc_orig = np.mean(orig)
    acc_pert = np.mean(pert)
    delta_pp = (acc_orig - acc_pert) * 100      # percentage-point drop

    score = rubric(delta_pp)
    print(f"Original accuracy : {acc_orig*100:.2f} %")
    print(f"Perturbed accuracy: {acc_pert*100:.2f} %")
    print(f"Drop              : {delta_pp:.2f} pp")
    print(f"BCR Robustness    : {score} / 3")
    
    return {
        "original_accuracy": float(acc_orig * 100),
        "perturbed_accuracy": float(acc_pert * 100),
        "accuracy_drop": float(delta_pp),
        "score": int(score)
    }

if __name__ == '__main__':
    result = main()
    print(json.dumps(result, indent=2))
