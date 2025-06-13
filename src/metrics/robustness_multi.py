# robust_multi.py
import pandas as pd
import numpy as np
import json
from src.metrics.robustness import rubric          # reuse the function

def main(pairs=None):
    if pairs is None:
        pairs = [
            ("data/predictions/gpt4_preds.csv",        "Original"),
            ("data/predictions/gpt4_paraphrase.csv",   "Paraphrased"),
            ("data/predictions/gpt4_noise.csv",        "Surface noise"),
            ("data/predictions/gpt4_shuffle.csv",      "Distractor shuffle"),
        ]

    base = np.array(pd.read_csv(pairs[0][0])["is_correct"].values)
    acc_base = np.mean(base) * 100
    
    results = []
    for path, name in pairs[1:]:
        pert = np.array(pd.read_csv(path)["is_correct"].values)
        delta = acc_base - np.mean(pert)*100
        score = rubric(delta)
        print(f"{name:18}: drop {delta:5.2f} pp  →  score {score}/3")
        results.append({
            "name": name,
            "accuracy_drop": float(delta),
            "score": int(score)
        })
    
    return {
        "base_accuracy": float(acc_base),
        "perturbations": results
    }

if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
